"""
Deep GP baseline for the GAN experiment.

Variational Deep GP (configurable depth/hidden dims) with no hierarchy.
Uses the same tabular surrogate and noise model as vanilla_gan.py.

Parent space: (lr_gen, z_dim, lr_disc, dropout) on a 7^4 = 2401-point grid.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gpytorch
import torch
import numpy as np
import warnings
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

import gan_models as models
from gan_surrogate import load_gan_surrogate, lookup_surrogate


# ---------------------------------------------------------------------------
# Deep GP Architecture (identical to deep_gp_neural.py)
# ---------------------------------------------------------------------------

class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=32):
        if output_dims is None:
            inducing_points = torch.rand(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.rand(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)

        if output_dims is not None:
            self.mean_module = gpytorch.means.LinearMean(input_dims, batch_shape=batch_shape)
        else:
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepGPModel(DeepGP):
    def __init__(self, input_dims, depth=3, num_inducing=32, hidden_dims=4):
        super().__init__()

        hidden_layers = torch.nn.ModuleList()
        current_dims = input_dims
        for _ in range(depth - 1):
            hidden_layers.append(DeepGPHiddenLayer(current_dims, hidden_dims, num_inducing))
            current_dims = hidden_dims

        self.hidden_layers = hidden_layers
        self.output_layer = DeepGPHiddenLayer(current_dims, None, num_inducing)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, inputs):
        output = inputs
        for layer in self.hidden_layers:
            output = layer(output)
        return self.output_layer(output)


class _Prediction:
    """Thin wrapper so DeepGP predictions are compatible with gan_models.get_acquisition_map()."""
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev


# ---------------------------------------------------------------------------
# Helper: Parent R^2 from heatmap data
# ---------------------------------------------------------------------------

def heatmap_r_score(data, y_hier):
    """
    Compute R^2 between predicted heatmap and ground truth for each
    repetition and each query step.

    Parameters:
        data: ndarray of shape (n_reps, n_queries, 2401)
        y_hier: (49, 49) tensor -- ground truth

    Returns:
        r_scores: list of lists, shape (n_reps, n_queries)
    """
    z = y_hier.numpy().reshape(-1)  # flatten to (2401,)
    r_scores = []
    for datum in data:
        trial_score = []
        for q in range(len(datum)):
            trial_score.append((linregress(z, datum[q]).rvalue) ** 2)
        r_scores.append(trial_score)
    return r_scores


# ---------------------------------------------------------------------------
# Single repetition
# ---------------------------------------------------------------------------

def run_repetition(kappa, nbr_query, nbr_rand_init, training_iter,
                   depth, num_inducing, hidden_dims,
                   seed=True, noise=0.1, disable_tqdm=False, surrogate_path=None):
    """
    Run one full Deep GP BO repetition on the GAN surrogate.

    A variational Deep GP (depth layers, hidden_dims per hidden layer) with
    UCB acquisition over the 2401-point parent grid.

    Parameters:
        kappa: UCB exploration weight
        nbr_query: number of BO iterations
        nbr_rand_init: number of random initial points from the parent grid
        training_iter: number of Deep GP optimization steps per query
        depth: number of GP layers
        num_inducing: number of inducing points per layer
        hidden_dims: output dimensionality of hidden layers
        seed: random seed (int or False)
        noise: additive Gaussian noise scale for surrogate lookups
        disable_tqdm: whether to suppress progress bar
        surrogate_path: path to surrogate .pt file (default: gan_surrogate.pt)

    Returns:
        (exploration_scores, exploitation_scores, heatmap_data, best_fid_so_far)
    """
    warnings.filterwarnings('ignore')

    if seed is not False and seed is not True:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Load surrogate data
    if surrogate_path is None:
        surrogate_dir = os.path.dirname(__file__)
        surrogate_path = os.path.join(surrogate_dir, "gan_surrogate.pt")
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = \
        load_gan_surrogate(surrogate_path)

    # Convert to double for Deep GP compatibility
    test_x_hier = test_x_hier.double()
    y_hier_flat = y_hier.flatten().double()
    ground_truth_max = torch.max(y_hier_flat)

    n_parent = len(test_x_hier)  # 2401
    hier_qc = torch.ones(n_parent).double()

    better_exploration_score = []
    better_exploitation_score = []
    heatmap_rep = []
    best_fid_so_far = []

    train_x_hier = None
    train_y_hier = None
    deep_gp = None
    optimizer = None

    for q in tqdm(range(nbr_query), disable=disable_tqdm):

        if q == 0:
            # Random init from the 2401-point grid
            np.random.seed(seed if (seed is not False and seed is not True) else None)
            init_indices = np.random.randint(0, n_parent, size=nbr_rand_init)
            train_x_hier = test_x_hier[init_indices].clone()
            train_y_hier = y_hier_flat[init_indices].clone()

            if noise > 0:
                y_range = torch.max(y_hier_flat) - torch.min(y_hier_flat)
                train_y_hier = train_y_hier + torch.tensor(
                    np.random.normal(0, noise * y_range.item(), size=train_y_hier.shape),
                    dtype=train_y_hier.dtype)

            for i in range(len(train_x_hier)):
                for j in range(n_parent):
                    if torch.allclose(test_x_hier[j], train_x_hier[i], atol=1e-6):
                        hier_qc[j] += 1
                        break

            input_dims = train_x_hier.shape[-1]  # 4
            deep_gp = DeepGPModel(input_dims, depth=depth,
                                  num_inducing=num_inducing, hidden_dims=hidden_dims)
            deep_gp.double()
            optimizer = torch.optim.Adam(deep_gp.parameters(), lr=0.01)

        # Prediction (eval mode, 10 MC samples)
        deep_gp.eval()
        deep_gp.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(10):
            preds = deep_gp.likelihood(deep_gp(test_x_hier))
        pred_mean = preds.mean.mean(0)
        pred_std = (preds.variance.mean(0) + preds.mean.var(0)).sqrt()
        observed_pred = _Prediction(pred_mean, pred_std)

        # Acquisition
        acquisition_map, y_mu = models.get_acquisition_map(kappa, observed_pred, hier_qc)
        next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)

        # Surrogate lookup
        next_query_value_random, next_query_value_mean = lookup_surrogate(
            next_query_pins, test_x_hier, y_hier, noise=noise)

        response = torch.tensor(next_query_value_random).double()

        # Update query counter
        for j in range(n_parent):
            if torch.allclose(test_x_hier[j], next_query_pins, atol=1e-6):
                hier_qc[j] += 1
                break

        # Update training data
        train_x_hier, train_y_hier = models.update_training_data(
            train_x_hier, train_y_hier, next_query_pins, response)

        # Train (train mode, 32 MC samples)
        mll = DeepApproximateMLL(
            VariationalELBO(deep_gp.likelihood, deep_gp, num_data=len(train_x_hier))
        )
        deep_gp.train()
        deep_gp.likelihood.train()
        for _ in range(training_iter):
            optimizer.zero_grad()
            with gpytorch.settings.num_likelihood_samples(32):
                output = deep_gp(train_x_hier)
                loss = -mll(output, train_y_hier)
            loss.backward()
            optimizer.step()

        # Metrics
        instantaneous_regret, _ = models.get_instantaneous_regret(
            y_mu, ground_truth_max, test_x_hier, y_hier)
        better_exploration_score.append(instantaneous_regret)

        exploitation = models.get_exploitation_score(next_query_value_mean, ground_truth_max)
        better_exploitation_score.append(exploitation)

        heatmap_rep.append(pred_mean.detach().cpu().numpy())

        best_neg_fid = torch.max(train_y_hier).item()
        best_fid_so_far.append(-best_neg_fid)

    return better_exploration_score, better_exploitation_score, heatmap_rep, best_fid_so_far


# ---------------------------------------------------------------------------
# Training procedure (multi-seed loop)
# ---------------------------------------------------------------------------

def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter,
                       k_vals, multi, seed, noise=0.1, disable_tqdm=False,
                       depth=3, num_inducing=32, hidden_dims=4):
    """
    Run the full Deep GP BO experiment across kappa values and seeds.

    Creates output directories, runs repetitions, collects metrics, saves
    CSV results and plots.

    Parameters:
        nbr_query: number of BO iterations per repetition
        nbr_repetition: number of seeds to run
        nbr_rand_init: number of random initial points from the parent grid
        training_iter: Deep GP optimization steps per query
        k_vals: list of kappa values
        multi: bool -- reserved for multiprocessing (not implemented)
        seed: array of integer seeds
        noise: surrogate noise scale
        disable_tqdm: suppress progress bars
        depth: number of GP layers
        num_inducing: number of inducing points per layer
        hidden_dims: output dimensionality of hidden layers
    """
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = os.path.join(os.path.dirname(__file__), "gan_deep_gp", "deep_gp")
    folder_of_the_day = '/data-' + str(current_dateday)
    full_path = workspace + folder_of_the_day

    if os.path.exists(full_path):
        print('Data folder is ready')
    else:
        os.makedirs(full_path, exist_ok=True)
        print('Data folder created')
        os.makedirs(full_path + '/csv', exist_ok=True)
        print("CSV folder created")
        os.makedirs(full_path + '/differentiable_plots', exist_ok=True)
        print("Differentiable plots folder created")
        os.makedirs(full_path + '/png', exist_ok=True)
        print("PNG folder created")

    # Load surrogate once to get ground truth for plotting
    surrogate_dir = os.path.dirname(__file__)
    surrogate_path = os.path.join(surrogate_dir, "gan_surrogate.pt")
    if os.path.exists(surrogate_path):
        _, _, _, _, _, y_hier_gt, _, _ = load_gan_surrogate(surrogate_path)
    else:
        _, _, _, _, _, y_hier_gt, _, _ = load_gan_surrogate()

    list_models = []

    for kappa in k_vals:
        better_exploration_score = []
        better_exploitation_score = []
        heatmap_data = []
        best_fid_data = []

        for i in range(nbr_repetition):
            print(f"\n--- Repetition {i+1}/{nbr_repetition} "
                  f"(seed={seed[i]}) k={kappa} depth={depth} ---")
            try:
                (rep_explor, rep_exploit, h_rep, fid_rep) = run_repetition(
                    kappa, nbr_query, nbr_rand_init, training_iter,
                    depth, num_inducing, hidden_dims,
                    seed=seed[i], noise=noise, disable_tqdm=disable_tqdm)
            except Exception as e1:
                print(f"  Attempt 1 failed ({e1}), retrying seed+1")
                try:
                    (rep_explor, rep_exploit, h_rep, fid_rep) = run_repetition(
                        kappa, nbr_query, nbr_rand_init, training_iter,
                        depth, num_inducing, hidden_dims,
                        seed=seed[i] + 1, noise=noise, disable_tqdm=disable_tqdm)
                except Exception as e2:
                    print(f"  Attempt 2 failed ({e2}), retrying seed+2")
                    try:
                        (rep_explor, rep_exploit, h_rep, fid_rep) = run_repetition(
                            kappa, nbr_query, nbr_rand_init, training_iter,
                            depth, num_inducing, hidden_dims,
                            seed=seed[i] + 2, noise=noise, disable_tqdm=disable_tqdm)
                    except Exception as e3:
                        print(f"  Attempt 3 failed ({e3}), skipping rep {i}")
                        continue

            better_exploration_score.append(rep_explor)
            better_exploitation_score.append(rep_exploit)
            heatmap_data.append(h_rep)
            best_fid_data.append(fid_rep)

        # ---------- Post-processing and plotting ----------
        if len(better_exploration_score) == 0:
            print(f"WARNING: All repetitions failed for k={kappa}")
            continue

        n_completed = len(better_exploration_score)
        heatmap_data = np.array(heatmap_data)

        # String-safe HP labels for filenames
        k = str(kappa).replace('.', ',')
        noi = str(noise).replace('.', ',')
        tag = (f"{n_completed}_rep_init_{nbr_rand_init}_"
               f"train_iter_{training_iter}_k_{k}_noise_{noi}_depth_{depth}")

        # --- Prepend zeros for the initial random queries ---
        n_pad = nbr_rand_init

        # 1) Instantaneous Regret (Relative Optimum)
        y_explor = np.mean(better_exploration_score, axis=0)
        y_explor = np.insert(y_explor, 0, np.zeros(n_pad))[:nbr_query]
        std_explor = (np.std(better_exploration_score, axis=0)
                      / np.sqrt(n_completed))
        std_explor = np.insert(std_explor, 0, np.zeros(n_pad))[:nbr_query]

        plt.figure()
        plt.plot(y_explor, label='Instantaneous Regret')
        plt.fill_between(range(len(y_explor)),
                         y_explor - std_explor,
                         y_explor + std_explor, alpha=0.4)

        # 2) Parent R^2
        r2 = np.array(heatmap_r_score(heatmap_data, y_hier_gt))
        r2 = np.pad(r2, ((0, 0), (n_pad, 0)), mode='constant')[:, :nbr_query]
        r2_avg = np.mean(r2, axis=0)
        r2_std = np.std(r2, axis=0) / np.sqrt(n_completed)

        plt.plot(r2_avg, label='Parent R2')
        plt.fill_between(range(len(r2_avg)),
                         r2_avg - r2_std, r2_avg + r2_std, alpha=0.4)

        plt.legend()
        plt.ylim(-0.1, 1.1)
        plt.ylabel('Performance')
        plt.xlabel('Query Number')
        plt.title(f'Deep GP GAN (depth={depth}) {n_completed} reps  k={kappa}')
        plt.savefig(os.path.join(
            full_path, 'differentiable_plots', f'{tag}.svg'))
        plt.savefig(os.path.join(
            full_path, 'png', f'{tag}.png'))
        plt.close()

        # 3) Best FID convergence
        fid_arr = np.array(best_fid_data)
        fid_avg = np.mean(fid_arr, axis=0)
        fid_std = np.std(fid_arr, axis=0) / np.sqrt(n_completed)

        plt.figure()
        plt.plot(fid_avg, label='Best FID')
        plt.fill_between(range(len(fid_avg)),
                         fid_avg - fid_std,
                         fid_avg + fid_std, alpha=0.4)
        plt.legend()
        plt.ylabel('FID (lower is better)')
        plt.xlabel('Query Number')
        plt.title(f'Best FID convergence  k={kappa}  depth={depth}')
        plt.savefig(os.path.join(
            full_path, 'png', f'fid_convergence_{tag}.png'))
        plt.savefig(os.path.join(
            full_path, 'differentiable_plots', f'fid_convergence_{tag}.svg'))
        plt.close()

        # --- Save CSV summary ---
        auc = np.sum(y_explor + r2_avg)
        print(f"\n===== k={kappa} depth={depth} =====")
        print(f"  Instantaneous regret (last): {y_explor[-1]:.4f}")
        print(f"  Parent R2 (last):            {r2_avg[-1]:.4f}")
        print(f"  AUC:                         {auc:.4f}")
        print(f"  Best FID (last avg):         {fid_avg[-1]:.2f}")

        df = pd.DataFrame(
            [f'kappa_{k}_depth_{depth}_{nbr_query}q_init_{nbr_rand_init}',
             y_explor[-1], r2_avg[-1], auc, fid_avg[-1]])
        df.index = ['name', 'instantaneous_regret', 'parent_r2', 'auc', 'best_fid']
        df.to_csv(os.path.join(full_path, 'csv', f'{tag}.csv'))

        # Also save the full time-series for later analysis
        ts = pd.DataFrame({
            'query': np.arange(nbr_query),
            'regret_mean': y_explor,
            'regret_sem': std_explor,
            'parent_r2_mean': r2_avg,
            'parent_r2_sem': r2_std,
            'fid_mean': np.pad(fid_avg, (n_pad, 0),
                               mode='constant')[:nbr_query],
            'fid_sem': np.pad(fid_std, (n_pad, 0),
                              mode='constant')[:nbr_query],
        })
        ts.to_csv(os.path.join(
            full_path, 'csv', f'timeseries_{tag}.csv'), index=False)

        list_models.append([
            f'kappa_{k}_depth_{depth}',
            y_explor, r2_avg, fid_avg])

    return list_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    nbr_query = 100
    training_iter = 100
    nbr_repetition = 5
    k_vals = [8]
    nbr_rand_init = 3
    noise = 0.1
    multi = True
    depth = 3
    num_inducing = 32
    hidden_dims = 4

    seed = np.array([9049607, 2402697, 6510749, 758529, 3523986, 3224638, 9729091,
       5830471, 5343420, 2417321, 9891788, 9314146, 9488226, 2697408,
       5135059, 6813578, 430826, 6192331, 8026546, 6735254, 1112898,
       5609958, 4736968, 617977, 8500888, 4205117, 756214, 4283694,
       7449696, 9848369])

    training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter,
                       k_vals, multi, seed, noise=noise,
                       depth=depth, num_inducing=num_inducing, hidden_dims=hidden_dims)
