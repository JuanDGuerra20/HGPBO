import math
import torch
import numpy as np
import gpytorch
import synthetic_models as models
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import hmodel_synthetic as hmodel
from dataset_actions import *
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
import pandas as pd


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def joint_plots(joint_exploit, joint_explor, k_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition,
                data_name):
    for i, kappa in enumerate(k_vals):
        plt.plot(joint_exploit[i], label=f'kappa {kappa}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploitation Score')
    plt.title(f'Joint vanilla Propagation HGPBO {nbr_repetition} Exploitation eps 0,75')
    plt.savefig(
        f'{data_name}/vanilla{folder_of_the_day}/differentiable_plots/Joint_vanilla_Propagation_HGPBO_{nbr_repetition}_Exploitation_eps_0,75')
    plt.close()

    for i, kappa in enumerate(k_vals):
        plt.plot(joint_explor[i], label=f'kappa {kappa}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploration Score')
    plt.title(f'Joint vanilla Propagation HGPBO {nbr_repetition} Exploration eps 0,75')
    plt.savefig(
        f'{data_name}/vanilla{folder_of_the_day}/differentiable_plots/Joint_vanilla_Propagation_HGPBO_{nbr_repetition}_Exploration_eps_0,75')
    plt.close()


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, data_name, data_creation_func,
                           eps, seed):

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/vanilla"
    folder_of_the_day = '/data-' + str(current_dateday)
    if os.path.exists(workspace + folder_of_the_day):
        print('Data folder is ready')
    else:
        os.mkdir(workspace + folder_of_the_day)
        print('Data folder created')
        os.mkdir(workspace + folder_of_the_day + '/contour')
        print("Contour folder created")
        os.mkdir(workspace + folder_of_the_day + '/differentiable_plots')
        print("Plots folder created")
        os.mkdir(workspace + folder_of_the_day + '/plots')
        print("PNG folder created")
        os.mkdir(workspace + folder_of_the_day + '/png')
        print("CSV folder created")
        os.mkdir(workspace + folder_of_the_day + '/csv')

    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    ground_truth_max_hier = torch.max(y_hier)
    sub1_qc = torch.ones(x_sub1.shape)
    sub2_qc = torch.ones(x_sub2.shape)
    hier_qc = torch.ones(len(x_sub1) * len(x_sub2))

    over_exploit = []
    over_explor = []
    heatmap_data = []

    for kappa in k_vals:
        better_exploration_score = []
        better_exploitation_score = []
        for repetition in range(nbr_repetition):
            max_seen_resp_2D = 0
            heatmap_rep = []
            for q in tqdm(range(nbr_query)):
                if q == 0:
                    train_x_hier, train_y_hier = hierarchical_select_random_queries(nbr_rand_init, x_hier, y_hier, seed=seed[repetition])
                    max_seen_resp_2D = torch.max(train_y_hier)


                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                   # master = ExactGPModel(train_x_hier, train_y_hier/ max_seen_resp_2D, likelihood)
                    master = ExactGPModel(train_x_hier, train_y_hier - train_y_hier.mean(), likelihood)

                    optimizer = torch.optim.Adam(master.parameters(), lr=1e-3)
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, master)

                    master.eval()
                    likelihood.eval()

                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

                acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred, hier_qc)

                next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)

                next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins,
                                                                                             test_x_hier,
                                                                                             y_hier)

                next_query_value_random, max_seen_resp_2D = models.update_max_seen_response_no_norm(next_query_value_random,
                                                                                            max_seen_resp_2D)

                response = torch.tensor(next_query_value_random)

                train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins,
                                                                  response)
                #master.set_train_data(train_x_hier, train_y_hier/max_seen_resp_2D, strict=False)
                master.set_train_data(train_x_hier, (train_y_hier - train_y_hier.mean())/train_y_hier.std() , strict=False)

                """
                train_x_sub1, train_x_sub2 = train_x_hier[:, 0], train_x_hier[:, 1]"""

                master.train()
                likelihood.train()
                for i in range(training_iter):
                    # Find optimal model hyperparameters
                    optimizer.zero_grad()

                    output = master(train_x_hier)

                    loss = -mll(output, train_y_hier)

                    loss.backward()
                    optimizer.step()
                    # Get into evaluation (predictive posterior) mode
                master.eval()
                likelihood.eval()


                # acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred)

                """exploration_score_2D, next_query_pins_exploration_2D = models.get_exploration_score(hierar_y_mu,
                                                                                                    ground_truth_max_hier,
                                                                                                    test_x_hier,
                                                                                                    x_hier, y_hier)"""
                instantaneous_regret, next_query_pins_exploration_2D = models.get_instantaneous_regret(hierar_y_mu,
                                                                                                    ground_truth_max_hier,
                                                                                                    test_x_hier,
                                                                                                    x_hier, y_hier)
                exploitation_score_2D = models.get_exploitation_score(next_query_value_mean, ground_truth_max_hier)
                """print(f'\nQuery Number: {q}')
                print(f'Next Query Pins: {next_query_pins_exploration_2D}')
                print(f'Next Query Value: {next_query_value_mean}')
                print(f'Exploration_Score: {torch.round(exploration_score_2D, decimals=4)}')
                print(f'Exploitation_Score: {torch.round(exploitation_score_2D, decimals=4)}')"""

                better_exploration_score.append(instantaneous_regret)
                better_exploitation_score.append(exploitation_score_2D)

                pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, master.likelihood)
                master_like = master.likelihood(pred)
                heatmap_rep.append(master_like.mean)

            heatmap_data.append(heatmap_rep)
            #print(f'\nRepetition {repetition} complete!\n')


        exploration_scores = []
        exploitation_scores = []
        heatmap_data = np.array(heatmap_data)

        for i in range(nbr_repetition):
            exploitation_scores.append(better_exploitation_score[i * nbr_query:(i + 1) * nbr_query])
            exploration_scores.append(better_exploration_score[i * nbr_query:(i + 1) * nbr_query])

        y = np.mean(exploration_scores, axis=0)
        over_explor.append(y)
        std = np.std(exploration_scores, axis=0) / np.sqrt(len(exploration_scores))
        plt.plot(y, label='Instantaneous Regret')
        plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)
        print(f"y {y[-1]}")
        y = np.mean(exploitation_scores, axis=0)
        over_exploit.append(y)

        std = np.std(exploitation_scores, axis=0)
        #plt.plot(y, label='Exploitation')
        #plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)

        r2 = vi.heatmap_r_score(heatmap_data, y_hier)
        r2_avg = np.mean(r2, axis=0)
        r2_std = np.std(r2, axis=0) / np.sqrt(len(r2))
        plt.plot(r2_avg, label="Parent R2")
        plt.fill_between(range(len(r2_std)), r2_avg - r2_std, r2_avg + r2_std, alpha=0.4)
        print(f"r2 = {r2_avg[-1]}")
        plt.legend()
        plt.ylim(-0.1, 1.1)
        k = str(kappa).replace('.', ',')

        plt.title(f'vanilla HGP-BO {nbr_repetition} repetitions with kappa value {k}')
        plt.savefig(
            f'{data_name}/vanilla{folder_of_the_day}/differentiable_plots/vanilla_Prop_{data_name}_HGP-BO_{nbr_repetition}_repetitions_kappa_{k}.svg')

        plt.title(f'vanilla HGP-BO {nbr_repetition} repetitions with kappa value {k}')
        plt.savefig(
            f'{data_name}/vanilla{folder_of_the_day}/png/vanilla_Prop_{data_name}_HGP-BO_{nbr_repetition}_repetitions_kappa_{k}.png')
        plt.close()

        vi.model_heatmap(heatmap_data[:, -1, :], x_hier, y_hier,
                         f'/Heatmap_{data_name}_vanilla_HGP-BO_{nbr_repetition}_repetitions_dim_{dimension}_kappa_{k}',
                         "vanilla", folder_of_the_day, data_name)
        vi.model_contour_3d(heatmap_data[:, -1, :], x_hier, y_hier,
                         f'/parent_contour_{data_name}_vanilla_HGP-BO_{nbr_repetition}_repetitions_dim_{dimension}_kappa_{k}',
                         "vanilla", folder_of_the_day, data_name)

        df = pd.DataFrame([
                              f'kappa_{k}_model_state_{nbr_query}_queries_eps_init_{nbr_rand_init}_train_iter_{training_iter}',
                              master, better_exploration_score, better_exploitation_score, r2])
        df.index = ['name', 'master', 'instantaneous_regret', 'exploitation_score', 'parent_r2']

        df.to_csv(
            f'{data_name}/vanilla/{folder_of_the_day}/csv/kappa_{k}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}')

    # Joint Section

    #joint_plots(over_exploit, over_explor, k_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition, data_name)


if __name__ == '__main__':

    dimension = 32
    nbr_query = 100
    training_iter = 10
    nbr_repetition = 30
    nbr_rand_init = 1
    k_vals = [2]
    seed = np.array([9049607, 2402697, 6510749,  758529, 3523986, 3224638, 9729091,
       5830471, 5343420, 2417321, 9891788, 9314146, 9488226, 2697408,
       5135059, 6813578,  430826, 6192331, 8026546, 6735254, 1112898,
       5609958, 4736968,  617977, 8500888, 4205117,  756214, 4283694,
       7449696, 9848369])
    # seed = [False] * nbr_repetition
    for dataset_num in [3,2,6,10]:
        data_name, data_creation_func, eps = get_dataset_info(dataset_num)

        training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, data_name, data_creation_func,
                           eps, seed)
