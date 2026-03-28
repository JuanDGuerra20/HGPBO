"""
Laferriere (unidirectional) baseline for GAN hyperparameter optimization.

Same hierarchical structure as BIF but with NO downward information flow:
  - Children provide initial prior to parent (upward)
  - Parent queries the surrogate
  - Children are NEVER updated (no contribution decomposition)
  - Child R^2 stays flat throughout the run

This directly matches Laferriere et al. (2020) as implemented in
model_testing/laferriere_script_synthetic.py.
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

import gan_models as models
import gan_hmodel as hmodel
from gan_surrogate import load_gan_surrogate, select_random_queries_2d, lookup_surrogate
from bif_gan import heatmap_r_score, compute_child_r2


# ---------------------------------------------------------------------------
# Single repetition
# ---------------------------------------------------------------------------

def run_repetition(kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter,
                   seed=True, noise=0.1, disable_tqdm=False, children=[],
                   surrogate_path=None):
    """
    Run one Laferriere (unidirectional) repetition on the GAN surrogate.

    Children are initialized once and never updated. They provide a static
    prior to the parent, which is the only model that learns from queries.

    children: [] = fresh init, [sub1] = pretrained gen (frozen) + fresh disc
    """
    warnings.filterwarnings('ignore')

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

    better_exploration_score = []
    better_exploitation_score = []
    child_1_r2 = []
    child_2_r2 = []
    heatmap_rep = []
    best_fid_so_far = []

    for q in tqdm(range(nbr_query), disable=disable_tqdm):

        if q == 0:
            if len(children) >= 1:
                # ----- Pretrained child 1 (gen, FROZEN) -----
                sub1 = children[0]
                sub1_like = sub1.likelihood
                sub1_qc = sub1.query_counter if sub1.query_counter is not None else sub1_qc
            else:
                # ----- Fresh child 1 -----
                train_x_sub1, train_y_sub1 = select_random_queries_2d(
                    nbr_rand_init, x_sub1, y_sub1, seed=seed, noise=noise)
                sub1_like = gpytorch.likelihoods.GaussianLikelihood()
                sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1, sub1_like,
                                           query_counter=sub1_qc, nu=nu)

            if len(children) >= 2:
                sub2 = children[1]
                sub2_like = sub2.likelihood
                sub2_qc = sub2.query_counter if sub2.query_counter is not None else sub2_qc
            else:
                train_x_sub2, train_y_sub2 = select_random_queries_2d(
                    nbr_rand_init, x_sub2, y_sub2,
                    seed=seed + 1 if isinstance(seed, int) else seed,
                    noise=noise)
                sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like,
                                           query_counter=sub2_qc, nu=nu)

            # Get child predictions (computed ONCE, never updated)
            sub1.eval(); sub1_like.eval()
            sub2.eval(); sub2_like.eval()

            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
                observed_pred2 = models.make_prediction(sub2, x_sub2, sub2_like)

            y_mu1, y_conf1 = observed_pred1.mean, observed_pred1.stddev
            y_mu2, y_conf2 = observed_pred2.mean, observed_pred2.stddev

            # Build static prior map
            p1 = y_mu1 + gamma * y_conf1 / torch.sqrt(sub1_qc)
            p2 = y_mu2 + gamma * y_conf2 / torch.sqrt(sub2_qc)

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
                prior_map / prior_map_max,
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

        # --- Child R^2 (static — children never change) ---
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

        next_query_value_random, max_seen_resp_2D = models.update_max_seen_response_no_norm(
            next_query_value_random, max_seen_resp_2D)

        response = torch.tensor(next_query_value_random).double()

        # --- NO contribution decomposition, NO child updates ---
        # (This is the key difference from BIF)

        # Update parent query counter only
        hier_qc = master.increment_q_n(hier_qc, next_query_pins, test_x_hier)

        # Prior map stays static (children unchanged)
        # master.mean_module.map stays the same

        # Update parent kernel from static children
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

        best_neg_fid = torch.max(train_y_hier).item()
        best_fid_so_far.append(-best_neg_fid)

    return (better_exploration_score, better_exploitation_score, heatmap_rep,
            child_1_r2, child_2_r2, best_fid_so_far)


# ---------------------------------------------------------------------------
# Training procedure (multi-seed loop)
# ---------------------------------------------------------------------------

def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter,
                       k_vals, g_vals, nu_vals, multi, seed, noise=0.1,
                       disable_tqdm=False):
    model_name = "Laferriere"

    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"gan_laferriere/{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)

    base_path = os.path.join(os.path.dirname(__file__), workspace)
    full_path = base_path + folder_of_the_day

    os.makedirs(full_path + '/csv', exist_ok=True)
    os.makedirs(full_path + '/differentiable_plots', exist_ok=True)
    os.makedirs(full_path + '/png', exist_ok=True)

    surrogate_dir = os.path.dirname(__file__)
    surrogate_path = os.path.join(surrogate_dir, "gan_surrogate.pt")
    if os.path.exists(surrogate_path):
        _, y_sub1_gt, _, y_sub2_gt, _, y_hier_gt, _, _ = \
            load_gan_surrogate(surrogate_path)
    else:
        _, y_sub1_gt, _, y_sub2_gt, _, y_hier_gt, _, _ = \
            load_gan_surrogate()

    list_models = []

    for kappa in k_vals:
        for gamma in g_vals:
            for nu in nu_vals:
                better_exploration_score = []
                better_exploitation_score = []
                heatmap_data = []
                child_1_r2_data = []
                child_2_r2_data = []
                best_fid_data = []

                for i in range(nbr_repetition):
                    print(f"\n--- Laferriere Rep {i+1}/{nbr_repetition} "
                          f"(seed={seed[i]}) k={kappa} g={gamma} nu={nu} ---")
                    try:
                        result = run_repetition(
                            kappa, gamma, nu, nbr_query, nbr_rand_init,
                            training_iter, seed=seed[i], noise=noise,
                            disable_tqdm=disable_tqdm)
                    except Exception as e1:
                        print(f"  Attempt 1 failed ({e1}), retrying seed+1")
                        try:
                            result = run_repetition(
                                kappa, gamma, nu, nbr_query, nbr_rand_init,
                                training_iter, seed=seed[i] + 1, noise=noise,
                                disable_tqdm=disable_tqdm)
                        except Exception as e2:
                            print(f"  Attempt 2 failed ({e2}), skipping")
                            continue

                    rep_explor, rep_exploit, h_rep, c1_r2, c2_r2, fid_rep = result
                    better_exploration_score.append(rep_explor)
                    better_exploitation_score.append(rep_exploit)
                    heatmap_data.append(h_rep)
                    child_1_r2_data.append(c1_r2)
                    child_2_r2_data.append(c2_r2)
                    best_fid_data.append(fid_rep)

                if len(better_exploration_score) == 0:
                    print(f"WARNING: All reps failed for k={kappa} g={gamma} nu={nu}")
                    continue

                n_completed = len(better_exploration_score)
                heatmap_data = np.array(heatmap_data)

                k = str(kappa).replace('.', ',')
                g = str(gamma).replace('.', ',')
                n = str(nu).replace('.', '_')
                noi = str(noise).replace('.', ',')
                tag = (f"{n_completed}_rep_init_{nbr_rand_init}_"
                       f"train_iter_{training_iter}_k_{k}_g_{g}_nu_{n}_noise_{noi}")

                n_pad = nbr_rand_init

                # Regret
                y_explor = np.mean(better_exploration_score, axis=0)
                y_explor = np.insert(y_explor, 0, np.zeros(n_pad))[:nbr_query]
                std_explor = np.std(better_exploration_score, axis=0) / np.sqrt(n_completed)
                std_explor = np.insert(std_explor, 0, np.zeros(n_pad))[:nbr_query]

                plt.figure()
                plt.plot(y_explor, label='Instantaneous Regret')
                plt.fill_between(range(len(y_explor)),
                                 y_explor - std_explor, y_explor + std_explor, alpha=0.4)

                # Parent R2
                r2 = heatmap_r_score(heatmap_data, y_hier_gt)
                r2 = np.pad(np.array(r2), ((0, 0), (n_pad, 0)), mode='constant')[:, :nbr_query]
                r2_avg = np.mean(r2, axis=0)
                r2_std = np.std(r2, axis=0) / np.sqrt(n_completed)

                plt.plot(r2_avg, label='Parent R2')
                plt.fill_between(range(len(r2_avg)),
                                 r2_avg - r2_std, r2_avg + r2_std, alpha=0.4)

                # Child R2 (flat — children static)
                all_children = np.concatenate([child_1_r2_data, child_2_r2_data])
                all_children = np.pad(all_children, ((0, 0), (n_pad, 0)),
                                      mode='constant')[:, :nbr_query]
                avg_child = np.mean(all_children, axis=0)
                std_child = np.std(all_children, axis=0) / np.sqrt(len(all_children))

                plt.plot(avg_child, label='Child Avg R2 (static)')
                plt.fill_between(range(len(avg_child)),
                                 avg_child - std_child, avg_child + std_child, alpha=0.4)

                plt.legend()
                plt.ylim(-0.1, 1.1)
                plt.ylabel('Performance')
                plt.xlabel('Query Number')
                plt.title(f'Laferriere GAN {n_completed} reps  k={kappa} g={gamma} nu={nu}')
                plt.savefig(os.path.join(full_path, 'differentiable_plots', f'{tag}.svg'))
                plt.savefig(os.path.join(full_path, 'png', f'{tag}.png'))
                plt.close()

                # FID convergence
                fid_arr = np.array(best_fid_data)
                fid_avg = np.mean(fid_arr, axis=0)
                fid_std = np.std(fid_arr, axis=0) / np.sqrt(n_completed)

                plt.figure()
                plt.plot(fid_avg, label='Best FID')
                plt.fill_between(range(len(fid_avg)),
                                 fid_avg - fid_std, fid_avg + fid_std, alpha=0.4)
                plt.legend()
                plt.ylabel('FID (lower is better)')
                plt.xlabel('Query Number')
                plt.title(f'Laferriere Best FID  k={kappa} g={gamma} nu={nu}')
                plt.savefig(os.path.join(full_path, 'png', f'fid_{tag}.png'))
                plt.savefig(os.path.join(full_path, 'differentiable_plots', f'fid_{tag}.svg'))
                plt.close()

                # CSV
                auc = np.sum(y_explor + avg_child + r2_avg)
                print(f"\n===== Laferriere k={kappa} g={gamma} nu={nu} =====")
                print(f"  Regret (last):       {y_explor[-1]:.4f}")
                print(f"  Parent R2 (last):    {r2_avg[-1]:.4f}")
                print(f"  Child R2 (last):     {avg_child[-1]:.4f}")
                print(f"  AUC:                 {auc:.4f}")
                print(f"  Best FID (last avg): {fid_avg[-1]:.2f}")

                df = pd.DataFrame(
                    [f'kappa_{k}_gamma_{g}_nu_{n}_{nbr_query}q_init_{nbr_rand_init}',
                     y_explor[-1], r2_avg[-1], avg_child[-1], auc, fid_avg[-1]])
                df.index = ['name', 'instantaneous_regret', 'parent_r2',
                            'avg_child_r2', 'auc', 'best_fid']
                df.to_csv(os.path.join(full_path, 'csv', f'{tag}.csv'))

                ts = pd.DataFrame({
                    'query': np.arange(nbr_query),
                    'regret_mean': y_explor,
                    'regret_sem': std_explor,
                    'parent_r2_mean': r2_avg,
                    'parent_r2_sem': r2_std,
                    'child_r2_mean': avg_child,
                    'child_r2_sem': std_child,
                    'fid_mean': np.pad(fid_avg, (n_pad, 0), mode='constant')[:nbr_query],
                    'fid_sem': np.pad(fid_std, (n_pad, 0), mode='constant')[:nbr_query],
                })
                ts.to_csv(os.path.join(full_path, 'csv', f'timeseries_{tag}.csv'), index=False)

                list_models.append([f'kappa_{k}_gamma_{g}_nu_{n}',
                                    y_explor, r2_avg, avg_child, fid_avg])

    return list_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    nbr_query = 100
    training_iter = 15
    nbr_repetition = 30
    k_vals = [8]
    g_vals = [4]
    nu_vals = [0.5]
    nbr_rand_init = 3
    noise = 0.1
    multi = False

    seed = np.array([9049607, 2402697, 6510749, 758529, 3523986, 3224638, 9729091,
       5830471, 5343420, 2417321, 9891788, 9314146, 9488226, 2697408,
       5135059, 6813578, 430826, 6192331, 8026546, 6735254, 1112898,
       5609958, 4736968, 617977, 8500888, 4205117, 756214, 4283694,
       7449696, 9848369])

    training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter,
                       k_vals, g_vals, nu_vals, multi, seed, noise=noise)
