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
import time


def joint_plots(joint_exploit, joint_explor, k_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition,
                data_name):
    for i, kappa in enumerate(k_vals):
        plt.plot(list(range(nbr_query)), np.mean(joint_exploit[i], 0), label=f'Kappa {str(kappa)}')
    plt.xlabel('Number of Queries')
    plt.ylabel('Exploitation score')
    plt.title(f'Exploitation Score of Norm_Base HGPBO model')
    plt.legend()
    plt.ylim((0, 1.1))
    plt.savefig(
        f"{data_name}/base{folder_of_the_day}/differentiable_plots/Joint_Exploit_Synthetic_Norm_Base_HGP-BO_{nbr_repetition}_repetitions_eps0,75_dim_{dimension}.png")
    plt.close()

    for i, kappa in enumerate(k_vals):
        plt.plot(list(range(nbr_query)), np.mean(joint_explor[i], 0), label=f'Kappa {str(kappa)}')
    plt.xlabel('Number of Queries')
    plt.ylabel('Exploration score')
    plt.title(f'Exploration Score of Norm_Base HGPBO model')
    plt.legend()
    plt.ylim((0, 1.1))
    plt.savefig(
        f"{data_name}/base{folder_of_the_day}/differentiable_plots/Joint_Explor_{data_name}_Norm_Base_HGP-BO_{nbr_repetition}_repetitions_eps0,75_dim_{dimension}.png")
    plt.close()


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, data_name, data_creation_func, eps):

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"C:/Users/preda/PycharmProjects/HierarchicalGPBO/model_testing/{data_name}/base"
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

    max_seen_resp_2D = 0
    max_seen_resp_1_1D = 0
    max_seen_resp_2_1D = 0
    list_prior_map = []
    list_objective_mean_map = []

    prior_map = torch.zeros(dimension, dimension)

    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    ground_truth_max_1 = torch.max(y_sub1)
    ground_truth_max_2 = torch.max(y_sub2)
    ground_truth_max_hier = torch.max(y_hier)

    joint_explor = []
    joint_exploit = []
    heatmap_data = []

    for kappa in k_vals:
        better_exploration_score = []
        better_exploitation_score = []
        for repetition in range(nbr_repetition):
            for q in tqdm(range(nbr_query)):
                if q == 0:
                    # Need to initialize the model
                    train_x_sub1, train_y_sub1 = select_random_queries(nbr_rand_init, x_sub1, y_sub1)
                    train_x_sub2, train_y_sub2 = select_random_queries(nbr_rand_init, x_sub2, y_sub2)
                    train_x_hier, train_y_hier = hierarchical_select_random_queries(nbr_rand_init, x_hier, y_hier)

                    max_seen_resp_1_1D = torch.max(train_y_sub1)
                    max_seen_resp_2_1D = torch.max(train_y_sub2)
                    max_seen_resp_2D = torch.max(train_y_hier)

                    train_y_sub1 = train_y_sub1 / max_seen_resp_1_1D
                    train_y_sub2 = train_y_sub2 / max_seen_resp_2_1D
                    train_y_hier = train_y_hier / max_seen_resp_2D

                    sub1_like = gpytorch.likelihoods.GaussianLikelihood()
                    sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1, sub1_like)

                    sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                    sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like)

                    sub1.eval()
                    sub2.eval()

                    sub1_like.eval()
                    sub2_like.eval()
                    with gpytorch.settings.lazily_evaluate_kernels(state=False):
                        observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
                        observed_pred2 = models.make_prediction(sub2, x_sub2, sub2_like)
                    y_mu1 = observed_pred1.mean
                    y_mu2 = observed_pred2.mean

                    for i in range(len(prior_map)):
                        for j in range(len(prior_map)):
                            prior_map[i, j] = y_mu1[i] + y_mu2[j]

                    prior_map_max = torch.max(prior_map)

                    prior_hierarchical_kernel = hmodel.hierarchical_kernel("add_kernel", sub1, sub2)
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    master = hmodel.Hierarchical_GP(train_x_hier, train_y_hier, likelihood, prior_hierarchical_kernel,
                                                    prior_map / prior_map_max, kernel_op='add_kernel', sub_models=[sub1, sub2],
                                                    kappa=kappa)

                    master.eval()
                    likelihood.eval()

                    with gpytorch.settings.lazily_evaluate_kernels(state=False):
                        observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

                acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred)

                next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)

                next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins,
                                                                                             test_x_hier,
                                                                                             y_hier)

                y_mu_point_a = hmodel.get_y_mu_point_value(next_query_pins[0], y_mu1, x_sub1)
                y_mu_point_b = hmodel.get_y_mu_point_value(next_query_pins[1], y_mu2, x_sub2)

                response_1, response_2 = hmodel.compute_responses(y_mu_point_a, y_mu_point_b, next_query_value_random)

                response_1, max_seen_resp_1_1D = models.update_max_seen_response(response_1, max_seen_resp_1_1D)
                response_2, max_seen_resp_2_1D = models.update_max_seen_response(response_2, max_seen_resp_2_1D)
                next_query_value_random, max_seen_resp_2D = models.update_max_seen_response(next_query_value_random,
                                                                                            max_seen_resp_2D)

                response = torch.tensor(next_query_value_random)
                response_1 = response_1.clone().detach()
                response_2 = response_2.clone().detach()

                # next_query_pins = next_query_pins.to(torch.int)

                flag = True
                for x in range(len(x_hier)):
                    for y in range(len(x_hier[x])):
                        if x_hier[x][y][0] == next_query_pins[0] and x_hier[x][y][1] == next_query_pins[1]:
                            next_query_indices = [x, y]
                            flag = False
                        if x == dimension - 1 and y == dimension - 1 and flag:
                            raise Exception("Could not find pins in X hier for indices")

                # update unitary model with response_1 and response_2
                sub1, sub1_like, train_x_sub1, train_y_sub1 = hmodel.update_model1_1D(sub1, sub1_like, train_x_sub1,
                                                                                      train_y_sub1,
                                                                                      x_sub1[next_query_indices[0]],
                                                                                      response_1,
                                                                                      training_iter=training_iter)

                sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model1_1D(sub2, sub2_like, train_x_sub2,
                                                                                      train_y_sub2,
                                                                                      x_sub2[next_query_indices[1]],
                                                                                      response_2,
                                                                                      training_iter=training_iter)

                sub1.eval()
                sub1_like.eval()

                sub2.eval()
                sub2_like.eval()

                # Make a prediction, observed_pred = likelihood
                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
                    observed_pred2 = models.make_prediction(sub2, x_sub2, sub2_like)

                y_mu1 = observed_pred1.mean
                y_mu2 = observed_pred2.mean

                for i in range(len(prior_map)):
                    for j in range(len(prior_map)):
                        prior_map[i, j] = y_mu1[i] + y_mu2[j]

                prior_map_max = torch.max(prior_map)

                master.mean_module.map = torch.nn.Parameter(prior_map / prior_map_max)

                master = hmodel.update_kernel_parameters(master, sub1, sub2)

                train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins, response)
                master.set_train_data(train_x_hier, train_y_hier, strict=False)

                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    # Find optimal model hyperparameters
                    master.train()
                    likelihood.train()

                    master, likelihood = master.Hoptimize(likelihood, training_iter, train_x_hier, train_y_hier,
                                                          verbose=False)
                    # Get into evaluation (predictive posterior) mode
                    master.eval()
                    likelihood.eval()

                    # Make a prediction, observed_pred = likelihood, prediction_mean = mu
                    observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

                # acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred)

                exploration_score_2D, next_query_pins_exploration_2D = models.get_exploration_score(hierar_y_mu,
                                                                                                    ground_truth_max_hier,
                                                                                                    test_x_hier,
                                                                                                    x_hier, y_hier)
                exploitation_score_2D = models.get_exploitation_score(next_query_value_mean, ground_truth_max_hier)
                """print(f'\nQuery Number: {q}')
                print(f'Next Query Pins: {next_query_pins_exploration_2D}')
                print(f'Next Query Value: {next_query_value_mean}')
                print(f'Exploration_Score: {torch.round(exploration_score_2D, decimals=4)}')
                print(f'Exploitation_Score: {torch.round(exploitation_score_2D, decimals=4)}')"""

                better_exploration_score.append(exploration_score_2D)
                better_exploitation_score.append(exploitation_score_2D)

            print(f'\nRepetition {repetition} complete!\n')
            pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, master.likelihood)
            master_like = master.likelihood(pred)
            heatmap_data.append(master_like.mean)

        # Currently only takes the last model of the repetitions, currently too lazy to fix
        vi.contour_plot_1D(master.sub_models, x_sub1, [y_sub1 / torch.max(y_sub1), y_sub2 / torch.max(y_sub2)],
                           f'/contour/Contour_{data_name}_Norm_Base_HGP-BO_{nbr_repetition}_repetitions_eps0,75_dim_{dimension}_kappa_{kappa}',
                           'base', folder_of_the_day, data_name)

        exploration_scores = []
        exploitation_scores = []

        for i in range(nbr_repetition):
            exploitation_scores.append(better_exploitation_score[i * nbr_query:(i + 1) * nbr_query])
            exploration_scores.append(better_exploration_score[i * nbr_query:(i + 1) * nbr_query])

        joint_explor.append(exploration_scores)
        joint_exploit.append(exploitation_scores)

        vi.model_heatmap(heatmap_data, x_hier, y_hier,
                         f'/Heatmap_{data_name}_Norm_Base_HGP-BO_{nbr_repetition}_repetitions_eps0,75_dim_{dimension}_kappa_{kappa}',
                         "base", folder_of_the_day, data_name)

    joint_plots(joint_exploit, joint_explor, k_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition, data_name)


if __name__ == '__main__':

    dimension = 10
    nbr_query = 50
    training_iter = 5
    nbr_repetition = 10
    nbr_rand_init = 5
    k_vals = [1, 2, 3, 4, 5, 6]

    t0 = time.time()
    for dataset_num in [1, 2, 3]:
        data_name, data_creation_func, eps = get_dataset_info(dataset_num)

        training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, data_name, data_creation_func,
                           eps)
    t1 = time.time()

    print(f"No Multiprocessing time: {t1 - t0}")