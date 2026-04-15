"""
BIF experiment script for GAN hyperparameter optimization.

Follows the pattern of model_testing/efficient_general_2d.py, adapted for:
  - 2D children (49 points each on a 7x7 grid)
  - 4D parent (2401 points = 49 x 49)
  - Tabular surrogate data (noise applied at query time, not data creation)

The parent space is (lr_gen, z_dim, lr_disc, dropout).
  Child 1 (Generator):     dims [0, 1] => lr_gen, z_dim
  Child 2 (Discriminator): dims [2, 3] => lr_disc, dropout
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
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

import gan_models as models
import gan_hmodel as hmodel
from gan_surrogate import load_gan_surrogate, select_random_queries_2d, lookup_surrogate


# ---------------------------------------------------------------------------
# Helper: Parent R^2 from heatmap data
# ---------------------------------------------------------------------------

def heatmap_r_score(data, y_hier):
    """
    Compute R^2 between predicted heatmap and ground truth for each
    repetition and each query step.

    Parameters:
        data: ndarray of shape (n_reps, n_queries, 2401)
        y_hier: (49, 49) tensor — ground truth

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
# Helper: Child R^2
# ---------------------------------------------------------------------------

def compute_child_r2(sub_models, true_x, true_y):
    """
    Compute R^2 for each child model against ground truth marginals.

    Uses min-max normalization to [0,1] for both predictions and ground truth,
    avoiding sign mismatch when child preds are positive but ground truth is
    negative (neg-FID marginals).

    Parameters:
        sub_models: list of two child GP models
        true_x: list of two (49, 2) tensors — child test domains
        true_y: list of two (49,) tensors — child ground truth

    Returns:
        list of two R^2 values [child1_r2, child2_r2]
    """
    children = []
    for i, model in enumerate(sub_models):
        model.eval()
        test_x = true_x[i]
        likelihood = model.likelihood
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))

        mean = observed_pred.mean.detach().numpy()
        gt = true_y[i].numpy()

        # Min-max normalize both to [0, 1] to avoid sign mismatch
        # (child preds may be positive from [0,1] training, gt may be negative from neg-FID)
        mn, mx = mean.min(), mean.max()
        if mx - mn > 0:
            mean_norm = (mean - mn) / (mx - mn)
        else:
            mean_norm = np.zeros_like(mean)

        gn, gx = gt.min(), gt.max()
        if gx - gn > 0:
            gt_norm = (gt - gn) / (gx - gn)
        else:
            gt_norm = np.zeros_like(gt)

        r2 = linregress(gt_norm, mean_norm).rvalue ** 2
        children.append(r2)

    return children


# ---------------------------------------------------------------------------
# Single repetition
# ---------------------------------------------------------------------------

def run_repetition(kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, acq_func,
                   seed=True, noise=0.1, disable_tqdm=False, children=[],
                   final=False, surrogate_path=None):
    """
    Run one full BIF-BO repetition on the GAN surrogate.

    Parameters:
        kappa: UCB exploration weight for the parent
        gamma: UCB exploration weight for children (prior map construction)
        nu: Matern kernel smoothness
        nbr_query: number of BO iterations
        nbr_rand_init: number of random initial points per child
        training_iter: number of GP optimization steps
        seed: random seed (int or False)
        noise: additive Gaussian noise scale for surrogate lookups
        disable_tqdm: whether to suppress progress bar
        children: list of pretrained child GPs. [] = init from scratch,
                  [sub1] = pretrained gen child + fresh disc,
                  [sub1, sub2] = both pretrained
        final: if True, also return the child models
        surrogate_path: path to surrogate .pt file (default: gan_surrogate.pt)

    Returns:
        (exploration_scores, exploitation_scores, heatmap_data,
         child1_r2, child2_r2, best_fid_so_far)
        If final=True, prepends (master, sub1, sub2) to the tuple.
    """
    warnings.filterwarnings('ignore')

    # Load surrogate data
    if surrogate_path is None:
        surrogate_dir = os.path.dirname(__file__)
        surrogate_path = os.path.join(surrogate_dir, "gan_surrogate.pt")
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = \
        load_gan_surrogate(surrogate_path)

    n_child = len(x_sub1)    # 49
    n_parent = len(test_x_hier)  # 2401

    prior_map = torch.zeros(n_child, n_child).double()

    ground_truth_max_hier = torch.max(y_hier)

    sub1_qc = torch.ones(n_child).double()
    sub2_qc = torch.ones(n_child).double()
    hier_qc = torch.ones(n_parent).double()

    max_seen_resp_2D = torch.tensor(0.0).double()
    max_seen_resp_1_1D = torch.tensor(0.0).double()
    max_seen_resp_2_1D = torch.tensor(0.0).double()

    better_exploration_score = []
    better_exploitation_score = []
    child_1_r2 = []
    child_2_r2 = []
    heatmap_rep = []
    best_fid_so_far = []

    for q in tqdm(range(nbr_query), disable=disable_tqdm):

        if q == 0:
            if len(children) == 0:
                # ----- Initialize both children with random points -----
                train_x_sub1, train_y_sub1 = select_random_queries_2d(
                    nbr_rand_init, x_sub1, y_sub1, seed=seed, noise=noise)
                train_x_sub2, train_y_sub2 = select_random_queries_2d(
                    nbr_rand_init, x_sub2, y_sub2,
                    seed=seed + 1 if isinstance(seed, int) else seed,
                    noise=noise)

                max_seen_resp_1_1D = torch.max(train_y_sub1)
                max_seen_resp_2_1D = torch.max(train_y_sub2)

                sub1_like = gpytorch.likelihoods.GaussianLikelihood()
                sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1, sub1_like,
                                           query_counter=sub1_qc, nu=nu)

                sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like,
                                           query_counter=sub2_qc, nu=nu)

                for i in range(len(train_x_sub1)):
                    sub1_qc = sub1.increment_q_n(sub1_qc, train_x_sub1[i], x_sub1)
                for i in range(len(train_x_sub2)):
                    sub2_qc = sub2.increment_q_n(sub2_qc, train_x_sub2[i], x_sub2)

            elif len(children) == 1:
                # ----- Pretrained child 1 (gen), fresh child 2 (disc) -----
                sub1 = children[0]
                sub1_like = sub1.likelihood
                sub1_qc = sub1.query_counter if sub1.query_counter is not None else sub1_qc
                train_x_sub1 = sub1.train_inputs[0]
                # Use raw_train_y (not train_targets which is rescaled)
                train_y_sub1 = sub1.raw_train_y.clone()
                max_seen_resp_1_1D = torch.max(train_y_sub1)

                train_x_sub2, train_y_sub2 = select_random_queries_2d(
                    nbr_rand_init, x_sub2, y_sub2,
                    seed=seed + 1 if isinstance(seed, int) else seed,
                    noise=noise)
                max_seen_resp_2_1D = torch.max(train_y_sub2)
                sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like,
                                           query_counter=sub2_qc, nu=nu)
                for i in range(len(train_x_sub2)):
                    sub2_qc = sub2.increment_q_n(sub2_qc, train_x_sub2[i], x_sub2)

            elif len(children) == 2:
                # ----- Both children pretrained -----
                sub1 = children[0]
                sub1_like = sub1.likelihood
                sub1_qc = sub1.query_counter if sub1.query_counter is not None else sub1_qc
                train_x_sub1 = sub1.train_inputs[0]
                train_y_sub1 = sub1.raw_train_y.clone()
                max_seen_resp_1_1D = torch.max(train_y_sub1)

                sub2 = children[1]
                sub2_like = sub2.likelihood
                sub2_qc = sub2.query_counter if sub2.query_counter is not None else sub2_qc
                train_x_sub2 = sub2.train_inputs[0]
                train_y_sub2 = sub2.raw_train_y.clone()
                max_seen_resp_2_1D = torch.max(train_y_sub2)

            # Get child predictions
            sub1.eval(); sub1_like.eval()
            sub2.eval(); sub2_like.eval()

            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
                observed_pred2 = models.make_prediction(sub2, x_sub2, sub2_like)

            y_mu1, y_conf1 = observed_pred1.mean, observed_pred1.stddev
            y_mu2, y_conf2 = observed_pred2.mean, observed_pred2.stddev

            p1 = models.compute_acq_value(y_mu1, y_conf1, sub1_qc, gamma, acq_func, max_seen_resp_1_1D)
            p2 = models.compute_acq_value(y_mu2, y_conf2, sub2_qc, gamma, acq_func, max_seen_resp_2_1D)

            for i in range(n_child):
                for j in range(n_child):
                    prior_map[i, j] = (p1[i] + p2[j]) / 2

            prior_map_max = torch.max(prior_map)

            # First query from prior map
            next_query_pins = models.get_next_query_pins(
                torch.flatten(prior_map), test_x_hier)
            next_query_value_random, next_query_value_mean = lookup_surrogate(
                next_query_pins, test_x_hier, y_hier, noise=noise)

            response = torch.tensor(next_query_value_random).double()
            train_x_hier = next_query_pins.unsqueeze(0)
            train_y_hier = response.unsqueeze(0)

            max_seen_resp_2D = torch.max(train_y_hier)

            # Build hierarchical kernel and parent model
            prior_h_kernel = hmodel.hierarchical_kernel("add_kernel", sub1, sub2)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

            master = hmodel.Lossless_Efficient_UCB_Hierarchical_GP(
                train_x_hier,
                train_y_hier - torch.mean(train_y_hier),
                test_x_hier,
                likelihood,
                prior_h_kernel,
                prior_map / prior_map_max if prior_map_max != 0 else prior_map,
                kernel_op='add_kernel',
                sub_models=[sub1, sub2],
                kappa=kappa,
                query_counter=hier_qc
            )

            for i in range(len(train_x_hier)):
                hier_qc = master.increment_q_n(hier_qc, train_x_hier[i], test_x_hier)

            master.eval(); likelihood.eval()
            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred = hmodel.make_Hierarchique_prediction(
                    master, test_x_hier, likelihood)

        # --- Compute child R^2 ---
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            c1_r2, c2_r2 = compute_child_r2(
                master.sub_models, [x_sub1, x_sub2], [y_sub1, y_sub2])
        child_1_r2.append(c1_r2)
        child_2_r2.append(c2_r2)

        # --- Acquisition ---
        acquisition_map, hierar_y_mu = models.get_acquisition_map(
            kappa, observed_pred, hier_qc)
        next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)

        # --- Surrogate lookup ---
        next_query_value_random, next_query_value_mean = lookup_surrogate(
            next_query_pins, test_x_hier, y_hier, noise=noise)

        # --- BIF contribution decomposition ---
        # Extract 2D child coords from 4D parent query
        child1_coord = next_query_pins[0:2]
        child2_coord = next_query_pins[2:4]

        y_mu_point_a = hmodel.get_y_mu_point_value_2d(child1_coord, y_mu1, x_sub1)
        y_mu_point_b = hmodel.get_y_mu_point_value_2d(child2_coord, y_mu2, x_sub2)

        y_conf_point_a = hmodel.get_y_mu_point_value_2d(child1_coord, y_conf1, x_sub1)
        y_conf_point_b = hmodel.get_y_mu_point_value_2d(child2_coord, y_conf2, x_sub2)

        y_qc_a = hmodel.get_y_mu_point_value_2d(child1_coord, sub1_qc, x_sub1)
        y_qc_b = hmodel.get_y_mu_point_value_2d(child2_coord, sub2_qc, x_sub2)

        next_query_value_random, max_seen_resp_2D = models.update_max_seen_response_no_norm(
            next_query_value_random, max_seen_resp_2D)

        response = torch.tensor(next_query_value_random).double()
        cont1 = models.compute_acq_value(y_mu_point_a, y_conf_point_a, y_qc_a, gamma, acq_func, max_seen_resp_1_1D)
        cont2 = models.compute_acq_value(y_mu_point_b, y_conf_point_b, y_qc_b, gamma, acq_func, max_seen_resp_2_1D)

        cont1_scaled = torch.nan_to_num(
            cont1 / torch.max(
                models.compute_acq_value(y_mu1, y_conf1, sub1_qc, gamma, acq_func, max_seen_resp_1_1D)))

        cont2_scaled = torch.nan_to_num(
            cont2 / torch.max(
                models.compute_acq_value(y_mu2, y_conf2, sub2_qc, gamma, acq_func, max_seen_resp_2_1D)))

        div = torch.exp(cont1_scaled) + torch.exp(cont2_scaled)
        contribution1 = torch.nan_to_num(response * torch.exp(cont1_scaled) / div)
        contribution2 = torch.nan_to_num(response * torch.exp(cont2_scaled) / div)

        # Update max seen for children
        response_1 = sub1.update_max_seen_response_no_norm(
            contribution1, max_seen_resp_1_1D)
        response_2 = sub2.update_max_seen_response_no_norm(
            contribution2, max_seen_resp_2_1D)

        # Update query counters
        sub1_qc = sub1.increment_q_n(sub1_qc, child1_coord, x_sub1)
        sub2_qc = sub2.increment_q_n(sub2_qc, child2_coord, x_sub2)
        hier_qc = master.increment_q_n(hier_qc, next_query_pins, test_x_hier)

        # Update children with contributions
        sub1, sub1_like, train_x_sub1, train_y_sub1 = hmodel.update_model_2d_max_seen(
            sub1, sub1_like, train_x_sub1, train_y_sub1,
            child1_coord, contribution1, False, training_iter=training_iter)

        sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model_2d_max_seen(
            sub2, sub2_like, train_x_sub2, train_y_sub2,
            child2_coord, contribution2, False, training_iter=training_iter)

        # Update child predictions
        sub1.eval(); sub1_like.eval()
        sub2.eval(); sub2_like.eval()

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
            observed_pred2 = models.make_prediction(sub2, x_sub2, sub2_like)

        y_mu1, y_conf1 = observed_pred1.mean, observed_pred1.stddev
        y_mu2, y_conf2 = observed_pred2.mean, observed_pred2.stddev

        # Rebuild prior map
        p1 = models.compute_acq_value(y_mu1, y_conf1, sub1_qc, gamma, acq_func, max_seen_resp_1_1D)
        p2 = models.compute_acq_value(y_mu2, y_conf2, sub2_qc, gamma, acq_func, max_seen_resp_2_1D)

        for i in range(n_child):
            for j in range(n_child):
                prior_map[i, j] = (p1[i] + p2[j]) / 2

        prior_map_max = torch.max(prior_map)
        if prior_map_max == 0:
            master.mean_module.map = torch.nn.Parameter(prior_map)
        else:
            master.mean_module.map = torch.nn.Parameter(prior_map / prior_map_max)

        # Update parent kernel
        master = hmodel.update_kernel_parameters(master, sub1, sub2)

        # Update parent training data
        train_x_hier, train_y_hier = models.update_training_data(
            train_x_hier, train_y_hier, next_query_pins, response)

        normalized_y = (train_y_hier - torch.mean(train_y_hier))
        if torch.std(train_y_hier) > 0:
            normalized_y = normalized_y / torch.std(train_y_hier)
        master.set_train_data(train_x_hier, normalized_y, strict=False)

        # Retrain parent
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            master.train(); likelihood.train()
            master, likelihood = master.Hoptimize(
                likelihood, training_iter, train_x_hier, normalized_y,
                verbose=False)

            master.eval(); likelihood.eval()
            observed_pred = hmodel.make_Hierarchique_prediction(
                master, test_x_hier, likelihood)

        # Record metrics
        instantaneous_regret, _ = models.get_instantaneous_regret(
            hierar_y_mu, ground_truth_max_hier, test_x_hier, y_hier)
        better_exploration_score.append(instantaneous_regret)

        exploitation_score = models.get_exploitation_score(
            next_query_value_mean, ground_truth_max_hier)
        better_exploitation_score.append(exploitation_score)

        heatmap_rep.append(observed_pred.mean.detach().cpu().numpy())

        # Track best FID found so far (neg_fid: max is best => convert back to positive FID)
        best_neg_fid = torch.max(train_y_hier).item()
        best_fid_so_far.append(-best_neg_fid)

    if final:
        return (master, sub1, sub2, better_exploration_score, better_exploitation_score,
                heatmap_rep, child_1_r2, child_2_r2, best_fid_so_far)
    return (better_exploration_score, better_exploitation_score, heatmap_rep,
            child_1_r2, child_2_r2, best_fid_so_far)


# ---------------------------------------------------------------------------
# Training procedure (multi-seed loop)
# ---------------------------------------------------------------------------

def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter,
                       k_vals, g_vals, nu_vals, multi, seed, noise=0.1, acq_func='ucb',
                       disable_tqdm=False):
    """
    Run the full BIF experiment across kappa / gamma / nu grids and seeds.

    Creates output directories, runs repetitions, collects metrics, saves
    CSV results and plots.

    Parameters:
        nbr_query: number of BO iterations per repetition
        nbr_repetition: number of seeds to run
        nbr_rand_init: number of random initial points per child
        training_iter: GP optimization steps
        k_vals: list of kappa values
        g_vals: list of gamma values
        nu_vals: list of nu (Matern smoothness) values
        multi: bool — use multiprocessing
        seed: array of integer seeds
        noise: surrogate noise scale
        disable_tqdm: suppress progress bars
    """
    model_name = "Lossless_Efficient"

    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"gan_bif/{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)

    base_path = os.path.join(os.path.dirname(__file__), workspace)
    full_path = base_path + folder_of_the_day

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

    list_models = []

    # Load surrogate once to get ground truth for plotting
    surrogate_dir = os.path.dirname(__file__)
    surrogate_path = os.path.join(surrogate_dir, "gan_surrogate.pt")
    if os.path.exists(surrogate_path):
        _, y_sub1_gt, _, y_sub2_gt, _, y_hier_gt, _, _ = \
            load_gan_surrogate(surrogate_path)
    else:
        _, y_sub1_gt, _, y_sub2_gt, _, y_hier_gt, _, _ = \
            load_gan_surrogate()

    for kappa in k_vals:
        for gamma in g_vals:
            for nu in nu_vals:
                better_exploration_score = []
                better_exploitation_score = []
                heatmap_data = []
                child_1_r2_data = []
                child_2_r2_data = []
                best_fid_data = []
                processes = []

                if multi:
                    with mp.Pool(processes=min(nbr_repetition, mp.cpu_count() - 1)) as pool:
                        # Submit all repetitions to the pool
                        for i in range(nbr_repetition):
                            p = pool.apply_async(run_repetition, (
                                kappa, gamma, nu, nbr_query, nbr_rand_init,
                                training_iter, acq_func, seed[i], noise, True))
                            processes.append(p)

                        # Collect results
                        for i, proc in enumerate(processes):
                            try:
                                (rep_explor, rep_exploit, h_rep,
                                 c1_r2, c2_r2, fid_rep) = proc.get()
                                better_exploration_score.append(rep_explor)
                                better_exploitation_score.append(rep_exploit)
                                heatmap_data.append(h_rep)
                                child_1_r2_data.append(c1_r2)
                                child_2_r2_data.append(c2_r2)
                                best_fid_data.append(fid_rep)
                            except Exception as e:
                                print(f"Repetition {i} (seed {seed[i]}) failed: {e}")
                                continue

                else:
                    for i in range(nbr_repetition):
                        print(f"\n--- Repetition {i+1}/{nbr_repetition} "
                              f"(seed={seed[i]}) k={kappa} g={gamma} nu={nu} ---")
                        try:
                            (rep_explor, rep_exploit, h_rep,
                             c1_r2, c2_r2, fid_rep) = run_repetition(
                                kappa, gamma, nu, nbr_query, nbr_rand_init,
                                training_iter, acq_func, seed=seed[i], noise=noise,
                                disable_tqdm=disable_tqdm)
                        except Exception as e1:
                            print(f"  Attempt 1 failed ({e1}), retrying seed+1")
                            try:
                                (rep_explor, rep_exploit, h_rep,
                                 c1_r2, c2_r2, fid_rep) = run_repetition(
                                    kappa, gamma, nu, nbr_query, nbr_rand_init,
                                    training_iter, acq_func, seed=seed[i] + 1, noise=noise,
                                    disable_tqdm=disable_tqdm)
                            except Exception as e2:
                                print(f"  Attempt 2 failed ({e2}), retrying seed+2")
                                try:
                                    (rep_explor, rep_exploit, h_rep,
                                     c1_r2, c2_r2, fid_rep) = run_repetition(
                                        kappa, gamma, nu, nbr_query, nbr_rand_init,
                                        training_iter, acq_func,  seed=seed[i] + 2, noise=noise,
                                        disable_tqdm=disable_tqdm)
                                except Exception as e3:
                                    print(f"  Attempt 3 failed ({e3}), skipping rep {i}")
                                    continue

                        better_exploration_score.append(rep_explor)
                        better_exploitation_score.append(rep_exploit)
                        heatmap_data.append(h_rep)
                        child_1_r2_data.append(c1_r2)
                        child_2_r2_data.append(c2_r2)
                        best_fid_data.append(fid_rep)

                # ---------- Post-processing and plotting ----------
                if len(better_exploration_score) == 0:
                    print(f"WARNING: All repetitions failed for k={kappa} g={gamma} nu={nu}")
                    continue

                n_completed = len(better_exploration_score)
                heatmap_data = np.array(heatmap_data)

                # String-safe HP labels for filenames
                k = str(kappa).replace('.', ',')
                g = str(gamma).replace('.', ',')
                n = str(nu).replace('.', '_')
                noi = str(noise).replace('.', ',')
                tag = (f"{n_completed}_rep_init_{nbr_rand_init}_"
                       f"train_iter_{training_iter}_k_{k}_g_{g}_nu_{n}_noise_{noi}")

                # --- Prepend zeros for the initial random queries ---
                n_pad = nbr_rand_init  # 2 children * nbr_rand_init / 2 = nbr_rand_init queries spent on init

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

                # 3) Child R^2 (average of both children)
                all_children = np.concatenate([child_1_r2_data, child_2_r2_data])
                all_children = np.pad(all_children, ((0, 0), (n_pad, 0)),
                                      mode='constant')[:, :nbr_query]
                avg_child = np.mean(all_children, axis=0)
                std_child = np.std(all_children, axis=0) / np.sqrt(len(all_children))

                plt.plot(avg_child, label='Child Avg R2')
                plt.fill_between(range(len(avg_child)),
                                 avg_child - std_child,
                                 avg_child + std_child, alpha=0.4)

                plt.legend()
                plt.ylim(-0.1, 1.1)
                plt.ylabel('Performance')
                plt.xlabel('Query Number')
                plt.title(f'BIF GAN {n_completed} reps  k={kappa} g={gamma} nu={nu}')
                plt.savefig(os.path.join(
                    full_path, 'differentiable_plots', f'{tag}.svg'))
                plt.savefig(os.path.join(
                    full_path, 'png', f'{tag}.png'))
                plt.close()

                # 4) Best FID convergence
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
                plt.title(f'Best FID convergence  k={kappa} g={gamma} nu={nu}')
                plt.savefig(os.path.join(
                    full_path, 'png', f'fid_convergence_{tag}.png'))
                plt.savefig(os.path.join(
                    full_path, 'differentiable_plots', f'fid_convergence_{tag}.svg'))
                plt.close()

                # --- Save CSV summary ---
                auc = np.sum(y_explor + avg_child + r2_avg)
                print(f"\n===== k={kappa} g={gamma} nu={nu} =====")
                print(f"  Instantaneous regret (last): {y_explor[-1]:.4f}")
                print(f"  Parent R2 (last):            {r2_avg[-1]:.4f}")
                print(f"  Child Avg R2 (last):         {avg_child[-1]:.4f}")
                print(f"  AUC:                         {auc:.4f}")
                print(f"  Best FID (last avg):         {fid_avg[-1]:.2f}")

                df = pd.DataFrame(
                    [f'kappa_{k}_gamma_{g}_nu_{n}_{nbr_query}q_init_{nbr_rand_init}',
                     y_explor[-1], r2_avg[-1], avg_child[-1], auc, fid_avg[-1]])
                df.index = ['name', 'instantaneous_regret', 'parent_r2',
                            'avg_child_r2', 'auc', 'best_fid']
                df.to_csv(os.path.join(full_path, 'csv', f'{tag}.csv'))

                # Also save the full time-series for later analysis
                ts = pd.DataFrame({
                    'query': np.arange(nbr_query),
                    'regret_mean': y_explor,
                    'regret_sem': std_explor,
                    'parent_r2_mean': r2_avg,
                    'parent_r2_sem': r2_std,
                    'child_r2_mean': avg_child,
                    'child_r2_sem': std_child,
                    'fid_mean': np.pad(fid_avg, (n_pad, 0),
                                       mode='constant')[:nbr_query],
                    'fid_sem': np.pad(fid_std, (n_pad, 0),
                                      mode='constant')[:nbr_query],
                })
                ts.to_csv(os.path.join(
                    full_path, 'csv', f'timeseries_{tag}.csv'), index=False)

                list_models.append([
                    f'kappa_{k}_gamma_{g}_nu_{n}',
                    y_explor, r2_avg, avg_child, fid_avg])

    return list_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    nbr_query = 100
    training_iter = 15
    nbr_repetition = 10
    k_vals = [8]
    g_vals = [4]
    nu_vals = [0.5]
    nbr_rand_init = 3
    noise = 0.1
    multi = False
    acq_func = 'pi'
    seed = np.array([9049607, 2402697, 6510749, 758529, 3523986, 3224638, 9729091,
       5830471, 5343420, 2417321, 9891788, 9314146, 9488226, 2697408,
       5135059, 6813578, 430826, 6192331, 8026546, 6735254, 1112898,
       5609958, 4736968, 617977, 8500888, 4205117, 756214, 4283694,
       7449696, 9848369])

    training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter,
                       k_vals, g_vals, nu_vals, multi, seed, noise=noise, acq_func=acq_func)
