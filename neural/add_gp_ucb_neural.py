import gpytorch
import matplotlib
matplotlib.use('Agg')

import models
import update_hmodel as hmodel
from dataset_actions import *
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
import pandas as pd
import os
import warnings
import multiprocessing as mp

xy2ch = [[2, 6, 10, 14, 9],
         [13, 17, 21, 18, 22]]
ch2xy = {}
for ch in CHS:
    x, y = np.where(np.array(xy2ch) == ch)
    ch2xy[ch] = [x[0], y[0]]

DT = 60
EMG = 4


def run_repetition(kappa, nu, nbr_query, nbr_rand_init, training_iter,
                   model_name, folder_of_the_day,
                   final=False, seed=False, disable_tqdm=False):
    warnings.filterwarnings('ignore')
    if type(seed) != bool:
        np.random.seed(seed)

    trainsC = Trains(clean_thresh=0.06)
    X_1D, Y_1D, Xmean_1D, Ymean_1D = make_dataset_1d(trainsC)
    test_x_1D = torch.tensor(Xmean_1D)
    test_y_1D = torch.tensor(Ymean_1D)

    x_sub1 = torch.from_numpy(X_1D.copy())
    x_sub2 = torch.from_numpy(X_1D.copy())
    y_sub1 = torch.from_numpy(Y_1D[:, 0].copy())
    y_sub2 = torch.from_numpy(Y_1D[:, 0].copy())

    X_2D, Y_2D, Xmean_2D, Ymean_2D = make_dataset_2d(trainsC)
    ground_truth_max_2D = np.max(Ymean_2D)

    x_hier = torch.from_numpy(X_2D.copy())
    y_hier = torch.from_numpy(Y_2D[:, 0].copy())
    test_x_hier = torch.tensor(Xmean_2D)   # shape (100, 4)
    test_y_hier = torch.tensor(Ymean_2D)   # shape (100,)

    n_test = len(test_x_1D)  # typically 10

    sub1_qc = torch.ones(n_test)
    sub2_qc = torch.ones(n_test)

    better_exploitation_score = []
    better_exploration_score = []
    child_1_r2 = []
    child_2_r2 = []
    heatmap_rep = []

    for q in tqdm(range(nbr_query), disable=disable_tqdm):

        if q == 0:
            train_x_sub1, train_y_sub1 = random_initialization_1D(nbr_rand_init, trainsC, 0, emg=EMG)
            train_x_sub2, train_y_sub2 = random_initialization_1D(nbr_rand_init, trainsC, 0, emg=EMG)

            sub1_like = gpytorch.likelihoods.GaussianLikelihood()
            sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1, sub1_like, nu=nu, query_counter=sub1_qc)

            sub2_like = gpytorch.likelihoods.GaussianLikelihood()
            sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like, nu=nu, query_counter=sub2_qc)

            for i in range(len(train_x_sub1)):
                sub1_qc = sub1.increment_q_n(sub1_qc, train_x_sub1[i], test_x_1D)
                sub2_qc = sub2.increment_q_n(sub2_qc, train_x_sub2[i], test_x_1D)

            sub1.eval(); sub1_like.eval()
            sub2.eval(); sub2_like.eval()

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            observed_pred1 = models.make_prediction(sub1, test_x_1D, sub1_like)
            observed_pred2 = models.make_prediction(sub2, test_x_1D, sub2_like)

        y_mu1, y_conf1 = observed_pred1.mean, observed_pred1.stddev
        y_mu2, y_conf2 = observed_pred2.mean, observed_pred2.stddev

        # Additive UCB over the n_test × n_test acquisition grid
        ucb1 = y_mu1 + kappa * torch.nan_to_num(y_conf1 / torch.sqrt(sub1_qc))
        ucb2 = y_mu2 + kappa * torch.nan_to_num(y_conf2 / torch.sqrt(sub2_qc))
        acq_2d_flat = (ucb1.unsqueeze(1) + ucb2.unsqueeze(0)).reshape(-1)

        # Argmax with tie-breaking
        tied = torch.where(acq_2d_flat == acq_2d_flat.max())[0]
        flat_idx = tied[np.random.randint(len(tied))].item()
        next_query_pins = torch.as_tensor(test_x_hier[flat_idx], dtype=torch.float64)

        next_query_value_random, next_query_value_mean = models.get_next_query_value(
            next_query_pins, x_hier, y_hier)

        # Update query counters using sub-coordinates from the 4D query point
        sub1_qc = sub1.increment_q_n(sub1_qc, next_query_pins[:2], test_x_1D)
        sub2_qc = sub2.increment_q_n(sub2_qc, next_query_pins[2:], test_x_1D)

        # Update each 1D GP with the observed value (env=True: no BIF decomposition)
        sub1, sub1_like, train_x_sub1, train_y_sub1 = hmodel.update_model1_1D_max_seen(
            sub1, sub1_like, train_x_sub1, train_y_sub1,
            next_query_pins[:2], torch.tensor(next_query_value_random), True,
            training_iter=training_iter)

        sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model1_1D_max_seen(
            sub2, sub2_like, train_x_sub2, train_y_sub2,
            next_query_pins[2:], torch.tensor(next_query_value_random), True,
            training_iter=training_iter)

        sub1.eval(); sub1_like.eval()
        sub2.eval(); sub2_like.eval()

        # Re-predict after update for metrics
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            observed_pred1 = models.make_prediction(sub1, test_x_1D, sub1_like)
            observed_pred2 = models.make_prediction(sub2, test_x_1D, sub2_like)

        y_mu1 = observed_pred1.mean
        y_mu2 = observed_pred2.mean

        # Additive mean surface over the test grid
        mean_2d_flat = (y_mu1.unsqueeze(1) + y_mu2.unsqueeze(0)).reshape(-1)

        instantaneous_regret, _ = models.get_instantaneous_regret(
            mean_2d_flat, ground_truth_max_2D, test_x_hier, x_hier, y_hier)
        exploitation_score_2D = models.get_exploitation_score(next_query_value_mean, ground_truth_max_2D)

        better_exploration_score.append(instantaneous_regret)
        better_exploitation_score.append(exploitation_score_2D)
        heatmap_rep.append(mean_2d_flat.detach().cpu().numpy())

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            c1_r2, c2_r2 = vi.child_contour_r2([sub1, sub2], [x_sub1, x_sub2], [y_sub1, y_sub2])
        child_1_r2.append(c1_r2)
        child_2_r2.append(c2_r2)
    vi.contour_plot_1D(
        [sub1, sub2], [test_x_1D, test_x_1D], [test_y_1D, test_y_1D],
        [train_y_sub1, train_y_sub2],
        f'/contour/Contour_init_{nbr_rand_init}_train_iter_{training_iter}',
        model_name.lower(), folder_of_the_day, "", parent=None, query=q,
        visualize=False, neural=True)
    if final:
        return sub1, sub2, better_exploration_score, better_exploitation_score, heatmap_rep, child_1_r2, child_2_r2
    else:
        return better_exploration_score, better_exploitation_score, heatmap_rep, child_1_r2, child_2_r2


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter,
                       k_vals, nu_vals, seed=None, multi=False, disable_tqdm=False):
    model_name = "add_gp_ucb"

    if seed is None:
        seed = [False] * nbr_repetition

    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = model_name
    folder_of_the_day = '/data-' + str(current_dateday)

    os.makedirs(workspace, exist_ok=True)
    if not os.path.exists(workspace + folder_of_the_day):
        os.mkdir(workspace + folder_of_the_day)
        for subdir in ['/contour', '/differentiable_plots', '/csv', '/hp_analysis', '/models', '/png']:
            os.mkdir(workspace + folder_of_the_day + subdir)

    # Load reference dataset for R² computation (identical across repetitions)
    trainsC = Trains(clean_thresh=0.06)
    X_2D, Y_2D, Xmean_2D, Ymean_2D = make_dataset_2d(trainsC)
    test_y_hier = torch.tensor(Ymean_2D)
    test_x_hier = torch.tensor(Xmean_2D)

    list_models = []

    for kappa in k_vals:
        for nu in nu_vals:
            better_exploration_score = []
            better_exploitation_score = []
            heatmap_data = []
            child_1_r2_data = []
            child_2_r2_data = []

            if multi:
                with mp.Pool(processes=nbr_repetition - 1) as pool:
                    processes = []
                    for i in range(nbr_repetition - 1):
                        p = pool.apply_async(run_repetition, (
                            kappa, nu, nbr_query, nbr_rand_init, training_iter,
                            model_name, folder_of_the_day,
                            False, seed[i], disable_tqdm))
                        processes.append(p)

                    try:
                        sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, c1_r2, c2_r2 = run_repetition(
                            kappa, nu, nbr_query, nbr_rand_init, training_iter,
                            model_name, folder_of_the_day,
                            True, seed[-1], disable_tqdm)
                    except Exception:
                        try:
                            sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, c1_r2, c2_r2 = run_repetition(
                                kappa, nu, nbr_query, nbr_rand_init, training_iter,
                                model_name, folder_of_the_day,
                                True, seed[-1], disable_tqdm)
                        except Exception:
                            try:
                                sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, c1_r2, c2_r2 = run_repetition(
                                    kappa, nu, nbr_query, nbr_rand_init, training_iter,
                                    model_name, folder_of_the_day,
                                    True, seed[-1], disable_tqdm)
                            except Exception:
                                sub1 = sub2 = None
                                rep_exploration_score, rep_exploitation_score, heatmap_rep, c1_r2, c2_r2 = [], [], [], [], []

                    better_exploration_score.append(rep_exploration_score)
                    better_exploitation_score.append(rep_exploitation_score)
                    heatmap_data.append(heatmap_rep)
                    child_1_r2_data.append(c1_r2)
                    child_2_r2_data.append(c2_r2)

                    for proc in processes:
                        try:
                            rep_exploration_score, rep_exploitation_score, heatmap_rep, c1_r2, c2_r2 = proc.get()
                            better_exploration_score.append(rep_exploration_score)
                            better_exploitation_score.append(rep_exploitation_score)
                            heatmap_data.append(heatmap_rep)
                            child_1_r2_data.append(c1_r2)
                            child_2_r2_data.append(c2_r2)
                        except Exception:
                            continue
            else:
                for i in range(nbr_repetition):
                    try:
                        sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, c1_r2, c2_r2 = run_repetition(
                            kappa, nu, nbr_query, nbr_rand_init, training_iter,
                            model_name, folder_of_the_day,
                            final=True, seed=seed[i], disable_tqdm=disable_tqdm)
                        better_exploration_score.append(rep_exploration_score)
                        better_exploitation_score.append(rep_exploitation_score)
                        heatmap_data.append(heatmap_rep)
                        child_1_r2_data.append(c1_r2)
                        child_2_r2_data.append(c2_r2)
                    except Exception as e:
                        print(f"Repetition {i} failed: {e}")
                        continue

            if not better_exploration_score:
                print(f"All repetitions failed for kappa={kappa}, nu={nu}. Skipping.")
                continue

            heatmap_data = np.array(heatmap_data)

            k = str(kappa).replace('.', ',')
            n = str(nu).replace('.', '_')

            y = np.mean(better_exploration_score, axis=0)
            y = np.insert(y, 0, np.zeros(2 * nbr_rand_init))[:nbr_query]
            std = np.std(better_exploration_score, axis=0) / np.sqrt(len(better_exploration_score))
            std = np.insert(std, 0, np.zeros(2 * nbr_rand_init))[:nbr_query]
            plt.plot(y, label='Instantaneous Regret')
            plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)

            r2 = vi.heatmap_r_score(heatmap_data, test_y_hier)
            r2 = np.insert(r2, 0, np.zeros((2 * nbr_rand_init, 1)), axis=1)[:, :nbr_query]
            r2_avg = np.mean(r2, axis=0)
            r2_std = np.std(r2, axis=0) / np.sqrt(len(r2))
            plt.plot(r2_avg, label="Parent R2")
            plt.fill_between(range(len(r2_avg)), r2_avg - r2_std, r2_avg + r2_std, alpha=0.4)

            all_children = np.concatenate([child_1_r2_data, child_2_r2_data])
            all_children = np.insert(all_children, 0, np.zeros((2 * nbr_rand_init, 1)), 1)[:, :nbr_query]
            avg_child = np.mean(all_children, axis=0)
            std_child = np.std(all_children, axis=0) / np.sqrt(len(all_children))
            plt.plot(avg_child, label='Child Avg R2')
            plt.fill_between(range(len(avg_child)), avg_child - std_child, avg_child + std_child, alpha=0.4)

            plt.legend()
            plt.ylim(-0.1, 1.1)
            plt.ylabel('Performance')
            plt.xlabel('Query Number')
            plt.title(f'{model_name} Neural {nbr_repetition} repetitions kappa {k} Nu {n} Init {nbr_rand_init}')
            plt.savefig(
                f'{workspace}{folder_of_the_day}/differentiable_plots/'
                f'{model_name}_Neural_{nbr_repetition}_rep_init_{nbr_rand_init}_kappa_{k}_nu_{n}.svg')
            plt.savefig(
                f'{workspace}{folder_of_the_day}/png/'
                f'{model_name}_Neural_{nbr_repetition}_rep_init_{nbr_rand_init}_kappa_{k}_nu_{n}.png')
            plt.close()

            vi.model_heatmap(
                heatmap_data[:, -1, :], test_x_hier, test_y_hier,
                f'/Heatmap_Neural_{model_name}_{nbr_repetition}_reps_k_{k}_nu_{n}_rand_init_{nbr_rand_init}',
                model_name, folder_of_the_day, "Neural", neural=True)

            auc = np.sum(y + avg_child + r2_avg)
            if not disable_tqdm:
                print(f"explor {y[-1] * 100:.2f} +/- {std[-1] * 100:.2f}")
                print(f"r2 avg {r2_avg[-1] * 100:.2f} +/- {r2_std[-1] * 100:.2f}")
                print(f"child r2 {avg_child[-1] * 100:.2f} +/- {std_child[-1] * 100:.2f}")
                print(f"AUC {auc * 100:.2f}")
                print(f'\nNeural {model_name} Kappa {k} Nu {n} complete!\n')

            df = pd.DataFrame(
                [f'kappa_{k}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                 y[-1], r2_avg[-1], avg_child[-1], auc])
            df.index = ['name', 'instantaneous_regret', 'parent_r2', 'avg_child_r2', 'auc']
            df.to_csv(
                f'{workspace}{folder_of_the_day}/csv/'
                f'final_scores_kappa_{k}_nu_{n}_{nbr_query}_queries_init_{nbr_rand_init}'
                f'_train_iter_{training_iter}_repetitions_{nbr_repetition}')

            torch.save(sub1.state_dict(),
                       f'{workspace}{folder_of_the_day}/models/sub1_kappa_{k}_nu_{n}.pth')
            torch.save(sub2.state_dict(),
                       f'{workspace}{folder_of_the_day}/models/sub2_kappa_{k}_nu_{n}.pth')

            list_models.append(
                [f'kappa_{k}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                 sub1, sub2, y, r2_avg, avg_child])

    return list_models


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    nbr_query = 100
    training_iter = 5
    nbr_repetition = 10
    nbr_rand_init = 1
    k_vals = [7.5]
    nu_vals = [0.5]
    seed = [False] * nbr_repetition

    training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter,
                       k_vals, nu_vals, seed, multi=True, disable_tqdm=False)
