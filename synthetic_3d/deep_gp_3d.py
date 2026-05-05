import torch
import numpy as np
import gpytorch
import os
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import models_3d as models
from dataset_actions_3d import *
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
import warnings


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
    def __init__(self, input_dims, depth=2, num_inducing=32, hidden_dims=3):
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
    """Thin wrapper so DeepGP predictions are compatible with models.get_acquisition_map()."""
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev


def run_repetition(kappa, nbr_query, nbr_rand_init, dimension, training_iter,
                   data_creation_func, eps, depth, num_inducing, hidden_dims,
                   seed=False, final=False, disable_tqdm=False):
    warnings.filterwarnings('ignore')
    if seed is not False:
        torch.manual_seed(seed)
        np.random.seed(seed)

    x_sub1, y_sub1, x_sub2, y_sub2, x_sub3, y_sub3, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    ground_truth_max_hier = torch.max(y_hier)
    hier_qc = torch.ones(len(x_sub1) * len(x_sub2) * len(x_sub3))

    train_x = None
    train_y = None
    max_seen = 0
    deep_gp = None
    optimizer = None

    better_exploration_score = []
    better_exploitation_score = []
    heatmap_rep = []

    for q in tqdm(range(nbr_query), disable=disable_tqdm):
        if q == 0:
            _tx, _ty = hierarchical_select_random_queries(
                nbr_rand_init, x_hier, y_hier, seed=seed
            )
            train_x = _tx
            train_y = _ty
            max_seen = torch.max(_ty)

            input_dims = _tx.shape[-1]
            deep_gp = DeepGPModel(input_dims, depth=depth, num_inducing=num_inducing, hidden_dims=hidden_dims)
            deep_gp.double()
            optimizer = torch.optim.Adam(deep_gp.parameters(), lr=0.01)

        deep_gp.eval()
        deep_gp.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(10):
            preds = deep_gp.likelihood(deep_gp(test_x_hier))
        pred_mean = preds.mean.mean(0)
        pred_std = (preds.variance.mean(0) + preds.mean.var(0)).sqrt()
        observed_pred = _Prediction(pred_mean, pred_std)

        acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred, hier_qc)
        next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)
        next_query_value_random, next_query_value_mean = models.get_next_query_value(
            next_query_pins, test_x_hier, y_hier
        )
        next_query_value_random, max_seen = models.update_max_seen_response_no_norm(
            next_query_value_random, max_seen
        )

        mask = (
            (test_x_hier[:, 0] == next_query_pins[0]) &
            (test_x_hier[:, 1] == next_query_pins[1]) &
            (test_x_hier[:, 2] == next_query_pins[2])
        )
        hier_qc[mask] += 1

        response = torch.tensor(next_query_value_random)
        train_x, train_y = update_training_data(train_x, train_y, next_query_pins, response)

        mll = DeepApproximateMLL(
            VariationalELBO(deep_gp.likelihood, deep_gp, num_data=len(train_x))
        )
        deep_gp.train()
        deep_gp.likelihood.train()
        for _ in range(training_iter):
            optimizer.zero_grad()
            with gpytorch.settings.num_likelihood_samples(32):
                output = deep_gp(train_x)
                loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        deep_gp.eval()
        deep_gp.likelihood.eval()

        instantaneous_regret, _ = models.get_instantaneous_regret(
            hierar_y_mu, ground_truth_max_hier, test_x_hier, x_hier, y_hier
        )
        exploitation_score_3D = models.get_exploitation_score(next_query_value_mean, ground_truth_max_hier)

        better_exploration_score.append(instantaneous_regret)
        better_exploitation_score.append(exploitation_score_3D)
        heatmap_rep.append(pred_mean.detach().cpu().numpy())

    if final:
        return deep_gp, better_exploration_score, better_exploitation_score, heatmap_rep
    return better_exploration_score, better_exploitation_score, heatmap_rep


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
                       k_vals, data_name, data_creation_func, eps, seed=None,
                       depth=4, num_inducing=32, multi=False, disable_tqdm=False, hidden_dims=3):

    if seed is None:
        seed = [False] * nbr_repetition

    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/deep_gp"
    folder_of_the_day = '/data-' + str(current_dateday)

    os.makedirs(workspace, exist_ok=True)
    if not os.path.exists(workspace + folder_of_the_day):
        os.mkdir(workspace + folder_of_the_day)
        for subdir in ['/contour', '/differentiable_plots', '/plots', '/png', '/csv']:
            os.mkdir(workspace + folder_of_the_day + subdir)

    x_sub1, y_sub1, x_sub2, y_sub2, x_sub3, y_sub3, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    for kappa in k_vals:
        better_exploration_score = []
        better_exploitation_score = []
        heatmap_data = []

        if multi:
            with mp.Pool(processes=nbr_repetition - 1) as pool:
                processes = []
                for i in range(nbr_repetition - 1):
                    p = pool.apply_async(run_repetition, (
                        kappa, nbr_query, nbr_rand_init, dimension, training_iter,
                        data_creation_func, eps, depth, num_inducing, hidden_dims,
                        seed[i], False, disable_tqdm))
                    processes.append(p)

                try:
                    deep_gp, rep_exploration_score, rep_exploitation_score, heatmap_rep = run_repetition(
                        kappa, nbr_query, nbr_rand_init, dimension, training_iter,
                        data_creation_func, eps, depth, num_inducing, hidden_dims,
                        seed[-1], True, disable_tqdm)
                except Exception:
                    try:
                        deep_gp, rep_exploration_score, rep_exploitation_score, heatmap_rep = run_repetition(
                            kappa, nbr_query, nbr_rand_init, dimension, training_iter,
                            data_creation_func, eps, depth, num_inducing, hidden_dims,
                            seed[-1], True, disable_tqdm)
                    except Exception:
                        try:
                            deep_gp, rep_exploration_score, rep_exploitation_score, heatmap_rep = run_repetition(
                                kappa, nbr_query, nbr_rand_init, dimension, training_iter,
                                data_creation_func, eps, depth, num_inducing, hidden_dims,
                                seed[-1], True, disable_tqdm)
                        except Exception:
                            deep_gp = None
                            rep_exploration_score, rep_exploitation_score, heatmap_rep = [], [], []

                better_exploration_score.append(rep_exploration_score)
                better_exploitation_score.append(rep_exploitation_score)
                heatmap_data.append(heatmap_rep)

                for proc in processes:
                    try:
                        rep_exploration_score, rep_exploitation_score, heatmap_rep = proc.get()
                        better_exploration_score.append(rep_exploration_score)
                        better_exploitation_score.append(rep_exploitation_score)
                        heatmap_data.append(heatmap_rep)
                    except Exception:
                        continue

        else:
            for i in range(nbr_repetition):
                deep_gp, rep_exploration_score, rep_exploitation_score, heatmap_rep = run_repetition(
                    kappa, nbr_query, nbr_rand_init, dimension, training_iter,
                    data_creation_func, eps, depth, num_inducing, hidden_dims,
                    seed[i], True, disable_tqdm)
                better_exploration_score.append(rep_exploration_score)
                better_exploitation_score.append(rep_exploitation_score)
                heatmap_data.append(heatmap_rep)

        heatmap_data = np.array(heatmap_data)

        y = np.mean(better_exploration_score, axis=0)
        y = np.insert(y, 0, np.zeros(nbr_rand_init))[:nbr_query]
        std = np.std(better_exploration_score, axis=0) / np.sqrt(len(better_exploration_score))
        std = np.insert(std, 0, np.zeros(nbr_rand_init))[:nbr_query]
        plt.plot(y, label='Instantaneous Regret')
        plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)

        r2 = vi.heatmap_r_score(heatmap_data, y_hier)
        r2 = np.insert(r2, 0, np.zeros((nbr_rand_init, 1)), axis=1)[:, :nbr_query]
        r2_avg = np.mean(r2, axis=0)
        r2_std = np.std(r2, axis=0) / np.sqrt(len(r2))
        plt.plot(r2_avg, label="Parent R2")
        plt.fill_between(range(len(r2_avg)), r2_avg - r2_std, r2_avg + r2_std, alpha=0.4)
        plt.legend()
        plt.ylim(-0.1, 1.1)

        k = str(kappa).replace('.', ',')
        auc = np.sum(y + r2_avg)

        if not disable_tqdm:
            print(f"explor {y[-1] * 100:.2f} +/- {std[-1] * 100:.2f}")
            print(f"r2 avg {r2_avg[-1] * 100:.2f} +/- {r2_std[-1] * 100:.2f}")
            print(f"AUC {auc * 100:.2f}")
            print(f'\n{data_name} Deep GP 3D depth={depth} Kappa {k} complete!\n')

        plt.title(f'Deep GP 3D (depth={depth}) BO {nbr_repetition} repetitions with kappa value {k}')
        plt.savefig(
            f'{workspace}{folder_of_the_day}/differentiable_plots/'
            f'deep_gp_3d_{data_name}_BO_{nbr_repetition}_repetitions_kappa_{k}.svg'
        )
        plt.savefig(
            f'{workspace}{folder_of_the_day}/png/'
            f'deep_gp_3d_{data_name}_BO_{nbr_repetition}_repetitions_kappa_{k}.png'
        )
        plt.close()

        df = pd.DataFrame(
            [f'kappa_{k}_depth_{depth}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
             y[-1], r2_avg[-1], auc]
        )
        df.index = ['name', 'instantaneous_regret', 'parent_r2', 'auc']
        df.to_csv(
            f'{workspace}{folder_of_the_day}/csv/'
            f'kappa_{k}_depth_{depth}_model_state_{nbr_query}_queries_init_{nbr_rand_init}'
            f'_train_iter_{training_iter}_repetitions_{nbr_repetition}'
        )


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    depth = 4
    dimension = 10
    nbr_query = 100
    training_iter = 100
    nbr_repetition = 10
    nbr_rand_init = 1
    k_vals = [7.5]
    seed = [False] * nbr_repetition

    for dataset_num in [6]:
        data_name, data_creation_func, eps = get_dataset_info(dataset_num)
        training_procedure(
            nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
            k_vals, data_name, data_creation_func, eps, seed, depth=depth, hidden_dims=3, multi=True
        )
