import gpytorch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np

import models_3d as models
import hmodel_3d as hmodel
from dataset_actions_3d import *
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
import os
import warnings


def run_repetition(kappa, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                   data_creation_func, eps, model_name, folder_of_the_day, data_name,
                   final=False, visualize=True, seed=False, noise=0.1, disable_tqdm=False):
    warnings.filterwarnings('ignore')
    x_sub1, y_sub1, x_sub2, y_sub2, x_sub3, y_sub3, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    ground_truth_max_hier = torch.max(y_hier)

    sub1_qc = torch.ones(x_sub1.shape)
    sub2_qc = torch.ones(x_sub2.shape)
    sub3_qc = torch.ones(x_sub3.shape)

    better_exploitation_score = []
    better_exploration_score = []
    child_1_r2 = []
    child_2_r2 = []
    child_3_r2 = []
    heatmap_rep = []

    for q in tqdm(range(nbr_query), disable=disable_tqdm):

        if q == 0:
            train_x_sub1, train_y_sub1 = select_random_queries(nbr_rand_init, x_sub1, y_sub1, seed=seed, noise=noise)
            train_x_sub2, train_y_sub2 = select_random_queries(nbr_rand_init, x_sub2, y_sub2, seed=seed, noise=noise)
            train_x_sub3, train_y_sub3 = select_random_queries(nbr_rand_init, x_sub3, y_sub3, seed=seed, noise=noise)

            sub1_like = gpytorch.likelihoods.GaussianLikelihood()
            sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1, sub1_like, sub1_qc, nu=nu)

            sub2_like = gpytorch.likelihoods.GaussianLikelihood()
            sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like, sub2_qc, nu=nu)

            sub3_like = gpytorch.likelihoods.GaussianLikelihood()
            sub3 = models.ExactGPModel(train_x_sub3, train_y_sub3, sub3_like, sub3_qc, nu=nu)

            for i in range(len(train_x_sub1)):
                sub1_qc = sub1.increment_q_n(sub1_qc, train_x_sub1[i], x_sub1)
                sub2_qc = sub2.increment_q_n(sub2_qc, train_x_sub2[i], x_sub2)
                sub3_qc = sub3.increment_q_n(sub3_qc, train_x_sub3[i], x_sub3)

            sub1.eval(); sub1_like.eval()
            sub2.eval(); sub2_like.eval()
            sub3.eval(); sub3_like.eval()

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
            observed_pred2 = models.make_prediction(sub2, x_sub2, sub2_like)
            observed_pred3 = models.make_prediction(sub3, x_sub3, sub3_like)

        y_mu1, y_conf1 = observed_pred1.mean, observed_pred1.stddev
        y_mu2, y_conf2 = observed_pred2.mean, observed_pred2.stddev
        y_mu3, y_conf3 = observed_pred3.mean, observed_pred3.stddev

        # Additive UCB: UCB(x1,x2,x3) = UCB1(x1) + UCB2(x2) + UCB3(x3)
        ucb1 = y_mu1 + kappa * torch.nan_to_num(y_conf1 / torch.sqrt(sub1_qc))
        ucb2 = y_mu2 + kappa * torch.nan_to_num(y_conf2 / torch.sqrt(sub2_qc))
        ucb3 = y_mu3 + kappa * torch.nan_to_num(y_conf3 / torch.sqrt(sub3_qc))

        # Outer sum over 3D grid, row-major layout matching test_x_hier
        acq_3d_flat = (
            ucb1.unsqueeze(1).unsqueeze(2) +
            ucb2.unsqueeze(0).unsqueeze(2) +
            ucb3.unsqueeze(0).unsqueeze(0)
        ).reshape(-1)

        # Argmax with tie-breaking
        tied = torch.where(acq_3d_flat == acq_3d_flat.max())[0]
        flat_idx = tied[np.random.randint(len(tied))].item()
        i_star, j_star, k_star = np.unravel_index(flat_idx, (dimension, dimension, dimension))
        next_query_pins = torch.as_tensor(test_x_hier[flat_idx], dtype=torch.float64)

        next_query_value_random, next_query_value_mean = models.get_next_query_value(
            next_query_pins, test_x_hier, y_hier, noise=noise)

        # Update query counters for all three sub-models
        sub1_qc = sub1.increment_q_n(sub1_qc, x_sub1[i_star], x_sub1)
        sub2_qc = sub2.increment_q_n(sub2_qc, x_sub2[j_star], x_sub2)
        sub3_qc = sub3.increment_q_n(sub3_qc, x_sub3[k_star], x_sub3)

        # Update each 1D GP with the observed value
        sub1, sub1_like, train_x_sub1, train_y_sub1 = hmodel.update_model1_1D_max_seen(
            sub1, sub1_like, train_x_sub1, train_y_sub1,
            x_sub1[i_star], torch.tensor(next_query_value_random), True,
            training_iter=training_iter)

        sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model1_1D_max_seen(
            sub2, sub2_like, train_x_sub2, train_y_sub2,
            x_sub2[j_star], torch.tensor(next_query_value_random), True,
            training_iter=training_iter)

        sub3, sub3_like, train_x_sub3, train_y_sub3 = hmodel.update_model1_1D_max_seen(
            sub3, sub3_like, train_x_sub3, train_y_sub3,
            x_sub3[k_star], torch.tensor(next_query_value_random), True,
            training_iter=training_iter)

        sub1.eval(); sub1_like.eval()
        sub2.eval(); sub2_like.eval()
        sub3.eval(); sub3_like.eval()

        # Re-predict for metrics
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
            observed_pred2 = models.make_prediction(sub2, x_sub2, sub2_like)
            observed_pred3 = models.make_prediction(sub3, x_sub3, sub3_like)

        y_mu1 = observed_pred1.mean
        y_mu2 = observed_pred2.mean
        y_mu3 = observed_pred3.mean

        # Additive mean surface flattened in row-major order
        mean_3d_flat = (
            y_mu1.unsqueeze(1).unsqueeze(2) +
            y_mu2.unsqueeze(0).unsqueeze(2) +
            y_mu3.unsqueeze(0).unsqueeze(0)
        ).reshape(-1)

        instantaneous_regret, _ = models.get_instantaneous_regret(
            mean_3d_flat, ground_truth_max_hier, test_x_hier, x_hier, y_hier)
        exploitation_score_3D = models.get_exploitation_score(next_query_value_mean, ground_truth_max_hier)

        better_exploration_score.append(instantaneous_regret)
        better_exploitation_score.append(exploitation_score_3D)
        heatmap_rep.append(mean_3d_flat.detach().cpu().numpy())

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            children_r2 = vi.child_contour_r2(
                [sub1, sub2, sub3], [x_sub1, x_sub2, x_sub3], [y_sub1, y_sub2, y_sub3])
        child_1_r2.append(children_r2[0])
        child_2_r2.append(children_r2[1])
        child_3_r2.append(children_r2[2])

    if visualize:
        k = str(kappa).replace('.', ',')
        n = str(nu).replace('.', ',')
        noi = str(noise).replace('.', ',')
        vi.contour_plot_1D(
            [sub1, sub2, sub3], [x_sub1, x_sub2, x_sub3], [y_sub1, y_sub2, y_sub3],
            [train_y_sub1, train_y_sub2, train_y_sub3],
            f'/contour/Contour_init_{nbr_rand_init}_train_iter_{training_iter}_k_{k}_nu_{n}_noise_{noi}',
            model_name.lower(), folder_of_the_day, data_name, parent=None, query=q,
            visualize=visualize)

    if final:
        return sub1, sub2, sub3, better_exploration_score, better_exploitation_score, heatmap_rep, child_1_r2, child_2_r2, child_3_r2
    else:
        return better_exploration_score, better_exploitation_score, heatmap_rep, child_1_r2, child_2_r2, child_3_r2


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
                       k_vals, nu_vals,
                       data_name, data_creation_func, eps,
                       multi, seed,
                       visualize=True, noise=0.1, disable_tqdm=False):
    model_name = "add_gp_ucb"

    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name}"
    folder_of_the_day = '/data-' + str(current_dateday)

    os.makedirs(workspace, exist_ok=True)
    if not os.path.exists(workspace + folder_of_the_day):
        os.mkdir(workspace + folder_of_the_day)
        for subdir in ['/contour', '/differentiable_plots', '/csv', '/hp_analysis', '/models', '/png']:
            os.mkdir(workspace + folder_of_the_day + subdir)

    list_models = []

    for kappa in k_vals:
        for nu in nu_vals:
            better_exploration_score = []
            better_exploitation_score = []
            heatmap_data = []
            c1_r2_data = []
            c2_r2_data = []
            c3_r2_data = []
            processes = []

            if multi:
                with mp.Pool(processes=nbr_repetition - 1) as pool:
                    for i in range(nbr_repetition - 1):
                        p = pool.apply_async(run_repetition, (
                            kappa, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                            data_creation_func, eps, model_name, folder_of_the_day, data_name,
                            False, visualize, seed[i], noise, disable_tqdm))
                        processes.append(p)

                    try:
                        sub1, sub2, sub3, rep_exploration_score, rep_exploitation_score, heatmap_rep, c1_r2, c2_r2, c3_r2 = run_repetition(
                            kappa, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                            data_creation_func, eps, model_name, folder_of_the_day, data_name,
                            True, visualize, seed[-1], noise, disable_tqdm)
                    except Exception:
                        try:
                            sub1, sub2, sub3, rep_exploration_score, rep_exploitation_score, heatmap_rep, c1_r2, c2_r2, c3_r2 = run_repetition(
                                kappa, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                                data_creation_func, eps, model_name, folder_of_the_day, data_name,
                                True, visualize, seed[-1], noise, disable_tqdm)
                        except Exception:
                            try:
                                sub1, sub2, sub3, rep_exploration_score, rep_exploitation_score, heatmap_rep, c1_r2, c2_r2, c3_r2 = run_repetition(
                                    kappa, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                                    data_creation_func, eps, model_name, folder_of_the_day, data_name,
                                    True, visualize, seed[-1], noise, disable_tqdm)
                            except Exception:
                                continue

                    better_exploration_score.append(rep_exploration_score)
                    better_exploitation_score.append(rep_exploitation_score)
                    heatmap_data.append(heatmap_rep)
                    c1_r2_data.append(c1_r2)
                    c2_r2_data.append(c2_r2)
                    c3_r2_data.append(c3_r2)

                    for proc in processes:
                        try:
                            rep_exploration_score, rep_exploitation_score, heatmap_rep, c1_r2, c2_r2, c3_r2 = proc.get()
                            better_exploration_score.append(rep_exploration_score)
                            better_exploitation_score.append(rep_exploitation_score)
                            heatmap_data.append(heatmap_rep)
                            c1_r2_data.append(c1_r2)
                            c2_r2_data.append(c2_r2)
                            c3_r2_data.append(c3_r2)
                        except Exception:
                            continue

            else:
                for i in range(nbr_repetition):
                    sub1, sub2, sub3, rep_exploration_score, rep_exploitation_score, heatmap_rep, c1_r2, c2_r2, c3_r2 = run_repetition(
                        kappa, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                        data_creation_func, eps, model_name, folder_of_the_day, data_name,
                        True, visualize, seed[i], noise, disable_tqdm)
                    better_exploration_score.append(rep_exploration_score)
                    better_exploitation_score.append(rep_exploitation_score)
                    heatmap_data.append(heatmap_rep)
                    c1_r2_data.append(c1_r2)
                    c2_r2_data.append(c2_r2)
                    c3_r2_data.append(c3_r2)

            heatmap_data = np.array(heatmap_data)

            # Reload ground truth for metrics and CSV output
            x_sub1, y_sub1, x_sub2, y_sub2, x_sub3, y_sub3, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

            k = str(kappa).replace('.', ',')
            n = str(nu).replace('.', '_')
            noi = str(noise).replace('.', ',')

            torch.save(sub1.state_dict(),
                       f'{workspace}{folder_of_the_day}/models/sub1_{nbr_repetition}_rep_init_{nbr_rand_init}_k_{k}_nu_{n}_noise_{noi}.pth')
            torch.save(sub2.state_dict(),
                       f'{workspace}{folder_of_the_day}/models/sub2_{nbr_repetition}_rep_init_{nbr_rand_init}_k_{k}_nu_{n}_noise_{noi}.pth')
            torch.save(sub3.state_dict(),
                       f'{workspace}{folder_of_the_day}/models/sub3_{nbr_repetition}_rep_init_{nbr_rand_init}_k_{k}_nu_{n}_noise_{noi}.pth')

            y = np.mean(better_exploration_score, axis=0)
            y = np.insert(y, 0, np.zeros(3 * nbr_rand_init))[:nbr_query]
            std = np.std(better_exploration_score, axis=0) / np.sqrt(len(better_exploration_score))
            std = np.insert(std, 0, np.zeros(3 * nbr_rand_init))[:nbr_query]
            plt.plot(y, label='Instantaneous Regret')
            plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)

            r2 = vi.heatmap_r_score(heatmap_data, y_hier)
            r2 = np.insert(r2, 0, np.zeros((3 * nbr_rand_init, 1)), axis=1)[:, :nbr_query]
            r2_avg = np.mean(r2, axis=0)
            r2_std = np.std(r2, axis=0) / np.sqrt(len(r2))
            plt.plot(r2_avg, label="Parent R2")
            plt.fill_between(range(len(r2_avg)), r2_avg - r2_std, r2_avg + r2_std, alpha=0.4)

            all_children = np.concatenate([c1_r2_data, c2_r2_data, c3_r2_data])
            all_children = np.insert(all_children, 0, np.zeros((3 * nbr_rand_init, 1)), 1)[:, :nbr_query]
            avg_child = np.mean(all_children, axis=0)
            std_child = np.std(all_children, axis=0) / np.sqrt(len(all_children))
            plt.plot(avg_child, label='Child Avg R2')
            plt.fill_between(range(len(avg_child)), avg_child - std_child, avg_child + std_child, alpha=0.4)

            plt.legend()
            plt.ylim(-0.1, 1.1)
            plt.ylabel('Performance')
            plt.xlabel('Query Number')
            plt.title(f'{model_name} 3D {nbr_repetition} repetitions kappa {k} Nu {n} Init {nbr_rand_init}')
            plt.savefig(
                f'{workspace}{folder_of_the_day}/differentiable_plots/{nbr_repetition}_rep_init_{nbr_rand_init}_k_{k}_nu_{n}_noise_{noi}.svg')
            plt.savefig(
                f'{workspace}{folder_of_the_day}/png/{nbr_repetition}_rep_init_{nbr_rand_init}_k_{k}_nu_{n}_noise_{noi}.png')
            plt.close()

            x_sub1, y_sub1, x_sub2, y_sub2, x_sub3, y_sub3, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

            data = np.mean(heatmap_data[:, -1, :], axis=0)
            df = pd.DataFrame(data)
            df.to_csv(
                f'{workspace}{folder_of_the_day}/csv/heatmap_{nbr_repetition}_rep_init_{nbr_rand_init}_k_{k}_nu_{n}_noise_{noi}.csv')
            df = pd.DataFrame(y_hier.reshape(-1).numpy())
            df.to_csv(f'{workspace}{folder_of_the_day}/csv/True_State_Space_Values.csv')

            auc = np.sum(y + avg_child + r2_avg)
            if not disable_tqdm:
                print(f"explor {y[-1] * 100:.2f} +/- {std[-1] * 100:.2f}")
                print(f"r2 avg {r2_avg[-1] * 100:.2f} +/- {r2_std[-1] * 100:.2f}")
                print(f"child r2 {avg_child[-1] * 100:.2f} +/- {std_child[-1] * 100:.2f}")
                print(f"AUC {auc * 100:.2f}")
                print(f'\n{data_name} {model_name} Kappa {k} Nu {n} complete!\n')

            df = pd.DataFrame(
                [f'kappa_{k}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                 y[-1], r2_avg[-1], avg_child[-1], auc])
            df.index = ['name', 'instantaneous_regret', 'parent_r2', 'avg_child_r2', 'auc']
            df.to_csv(
                f'{workspace}{folder_of_the_day}/csv/final_scores_kappa_{k}_nu_{n}_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}')

            list_models.append(
                [f'kappa_{k}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                 sub1, sub2, sub3, y, r2_avg, avg_child])

    return list_models


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    dimension = 10
    nbr_query = 100
    training_iter = 10
    nbr_repetition = 10
    k_vals = [7.5]
    nu_vals = [0.5]
    multi = True
    nbr_rand_init = 1
    noise = 0.1
    seed = [False] * nbr_repetition

    for dataset_num in [6]:
        data_name, data_creation_func, eps = get_dataset_info(dataset_num)
        training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
                           k_vals, nu_vals, data_name, data_creation_func, eps,
                           multi, seed, visualize=True, noise=noise, disable_tqdm=False)
