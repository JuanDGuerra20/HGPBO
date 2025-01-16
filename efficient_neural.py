import math
import torch
import numpy as np
import gpytorch
import models
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import update_hmodel as hmodel
from dataset_actions import *
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
from seaborn import heatmap



xy2ch = [[2,6,10,14,9],
         [13,17,21,18,22]]
ch2xy = {}
for ch in CHS:
    x, y = np.where(np.array(xy2ch) == ch)
    ch2xy[ch] = [x[0],y[0]]

DTS = [0, 10, 20, 40, 60, 80, 100]
EMG = 4
N_EMGS = 7

DT = 60
EMG = 4

training_iter = 5
q = 0
nbr_rdm_points_ini = 5       # Number of random points to initialize the model
nbr_rdm_points_data = 20     # Number of random points from the Ground Truth (GT) EMG responses to create a dataset
max_seen_resp = 0            # To store the maximum response observed, and normalize between 0 and 1 (arbitrary choice)
nbr_repetition = 5          # Number of repetition of the entire process to average the results
nbr_pins = 100               # Number of pins in he input space (2*matrix of 2*5 pins)

name_code = 'HGP_BO-test6-priorMAP-1model1D'
current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
current_dateday = datetime.now().strftime("%Y-%m-%d")

workspace_folder = (r'C:\Users\preda\PycharmProjects\HierarchicalGPBO') #path to folder

os.chdir(workspace_folder)

kern_op = 'add_kernel'
folder_of_the_day = (str(workspace_folder) + f'/efficient_3D/data-' + str(name_code) + str(current_dateday))


def joint_plots(joint_exploit, joint_explor, kappa, gamma, nu_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition,
                data_name, model_name):
    print('Entering Joint Plots')
    for i, nu in enumerate(nu_vals):
        plt.plot(joint_exploit[i], label=f'nu {nu}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploitation Score')
    plt.title(f'Joint {model_name} Propagation HGPBO {nbr_repetition} Exploitation')
    plt.savefig(
        f'{model_name.lower()}{folder_of_the_day}/hp_analysis/Joint_{model_name}_Neural_Propagation_HGPBO_{nbr_repetition}_Exploitation_query_kappa_{kappa}_gamma_{gamma}')
    plt.close()

    for i, kappa in enumerate(k_vals):
        plt.plot(joint_explor[i], label=f'kappa {kappa}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploration Score')
    plt.title(f'Joint {model_name} Propagation HGPBO {nbr_repetition} Exploration')
    plt.savefig(
        f'{model_name.lower()}{folder_of_the_day}/hp_analysis/Joint_{model_name}_Neural_Propagation_HGPBO_{nbr_repetition}_Exploration_query_kappa_{kappa}_gamma_{gamma}')
    plt.close()


def joint_performance(joint_exploit, joint_explor, kappa, gamma, nu_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition,
                data_name, model_name):

    names = []

    for n in nu_vals:
        names.append(f'kappa_{kappa}_gamma_{gamma}_nu_{n}')
    fig, ax = plt.subplots(figsize=(nbr_query/5, len(nu_vals)*3))
    heatmap(joint_exploit, xticklabels=list(range(nbr_query)), yticklabels=names, cmap='coolwarm', ax=ax)

    plt.ylabel(f'Model Type')
    plt.xlabel(f'Training Step')
    plt.title(f'Joint HP Exploitation Performance over {nbr_repetition} repetitions')
    plt.tight_layout()
    plt.savefig(
        f'{model_name.lower()}{folder_of_the_day}/hp_analysis/HP_Exploitation_Performance_{nbr_repetition}_repetitions_kappa_{kappa}_gamma_{gamma}')
    plt.close()

    fig, ax = plt.subplots(figsize=(nbr_query/5, len(nu_vals)*3))
    heatmap(joint_explor, xticklabels=list(range(nbr_query)), yticklabels=names, cmap='coolwarm', ax=ax)

    plt.ylabel(f'Model Type')
    plt.xlabel(f'Training Step')
    plt.title(f'Joint HP Exploration Performance over {nbr_repetition} repetitions')
    plt.tight_layout()
    plt.savefig(
        f'{model_name.lower()}{folder_of_the_day}/hp_analysis/HP_Exploration_Performance_{nbr_repetition}_repetitions_kappa_{kappa}_gamma_{gamma}')
    plt.close()


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals):

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"C:/Users/preda/PycharmProjects/HierarchicalGPBO/efficient_3D"
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

    ground_truth_max_1 = torch.max(y_sub1)
    ground_truth_max_2 = torch.max(y_sub2)
    ground_truth_max_hier = torch.max(y_hier)

    over_exploit = []
    over_explor = []
    heatmap_data = []
    list_prior_map = []
    list_objective_mean_map = []
    prior_map = torch.zeros(10, 10)
    list_queries = []
    list_query_values = []
    list_max_seen_1_1D = []
    list_max_seen_2_1D = []
    list_max_seen_2D = []
    list_acquisition_map = []
    list_next_value_mean = []


    for kappa in k_vals:
        better_exploration_score = []
        better_exploitation_score = []
        for repetition in range(nbr_repetition):
            prior_map_max = 0
            max_seen_resp_2D = 0
            max_seen_resp_1_1D = 0
            max_seen_resp_2_1D = 0
            for q in tqdm(range(nbr_query)):
                if q == 0:
                    # Need to initialize the model - Will be random in this method
                    train_x_sub1, train_y_sub1 = random_initialization_1D(nbr_rand_init, trainsC,
                                                                    max_seen_resp_1_1D, emg=EMG)
                    train_x_sub2, train_y_sub2 = random_initialization_1D(nbr_rand_init, trainsC,
                                                                    max_seen_resp_1_1D, emg=EMG)
                    train_x_hier, train_y_hier = hmodel.random_initialization(nbr_rand_init, EMG, trainsC,
                                                                      max_seen_resp_2D, DT)

                    train_x_hier = torch.tensor(train_x_hier, dtype=torch.float64)
                    train_y_hier = torch.tensor(train_y_hier, dtype=torch.float64)

                    """max_seen_resp_1_1D = torch.max(train_y_sub1)
                    max_seen_resp_2_1D = torch.max(train_y_sub2)
                    max_seen_resp_2D = torch.max(train_y_hier)

                    train_y_sub1 = train_y_sub1 / max_seen_resp_1_1D
                    train_y_sub2 = train_y_sub2 / max_seen_resp_2_1D
                    train_y_hier = train_y_hier / max_seen_resp_2D"""

                    # Need to modify this section such that the model is receiving the partial contribution
                    train_x_sub1, train_y_sub1 = update_training_data(train_x_sub1, train_y_sub1,
                                                                      train_x_hier[:, :2], train_y_hier)
                    train_x_sub2, train_y_sub2 = update_training_data(train_x_sub2, train_y_sub2,
                                                                      train_x_hier[:, 2:], train_y_hier)

                    sub1_like = gpytorch.likelihoods.GaussianLikelihood()
                    sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1, sub1_like)

                    sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                    sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like)

                    sub1.eval()
                    sub2.eval()

                    sub1_like.eval()
                    sub2_like.eval()
                    with gpytorch.settings.lazily_evaluate_kernels(state=False):
                        observed_pred1 = models.make_prediction(sub1, test_x_1D, sub1_like)
                        observed_pred2 = models.make_prediction(sub2, test_x_1D, sub2_like)
                    y_mu1 = observed_pred1.mean
                    y_mu2 = observed_pred2.mean

                    for i in range(len(prior_map)):
                        for j in range(len(prior_map)):
                            prior_map[i, j] = y_mu1[i] + y_mu2[j]

                    prior_map_max = torch.max(prior_map)
                    list_prior_map.append(np.array(prior_map / prior_map_max))

                    prior_hierarchical_kernel = hmodel.hierarchical_kernel("add_kernel", sub1, sub2)
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    master = hmodel.Efficient_UCB_Hierarchical_GP(train_x_hier, train_y_hier, likelihood,
                                                                  prior_hierarchical_kernel,
                                                                  prior_map / prior_map_max, kernel_op='add_kernel',
                                                                  sub_models=[sub1, sub2],
                                                                  kappa=kappa)

                    master.eval()
                    likelihood.eval()

                    with gpytorch.settings.lazily_evaluate_kernels(state=False):
                        observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

                    list_objective_mean_map.append(np.array(np.reshape(observed_pred.mean, (10, 10))))

                vi.comparison(list_prior_map, list_objective_mean_map, q, Xmean_1D)

                acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred)
                list_acquisition_map.append(list_acquisition_map)

                next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)

                next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins,
                                                                                             X_2D,
                                                                                             Y_2D)
                list_next_value_mean.append(next_query_value_mean)

                list_queries.append(next_query_pins)
                list_query_values.append(next_query_value_random)

                y_mu_point_a = hmodel.get_y_mu_point_value(next_query_pins[:2], y_mu1, test_x_1D)
                y_mu_point_b = hmodel.get_y_mu_point_value(next_query_pins[2:], y_mu2, test_x_1D)

                response_1, response_2 = hmodel.compute_responses(y_mu_point_a, y_mu_point_b,
                                                                  next_query_value_random)


                next_query_value_random, max_seen_resp_2D = models.update_max_seen_response(next_query_value_random,
                                                                                            max_seen_resp_2D)

                list_max_seen_1_1D.append(max_seen_resp_1_1D)
                list_max_seen_2_1D.append(max_seen_resp_2_1D)
                list_max_seen_2D.append(max_seen_resp_2D)

                response = torch.tensor(next_query_value_random)
                response_1 = response_1.clone().detach()
                response_2 = response_2.clone().detach()

                contribution1 = response * y_mu_point_a / (y_mu_point_a + y_mu_point_b)
                contribution2 = response * y_mu_point_b / (y_mu_point_a + y_mu_point_b)

                response_1, max_seen_resp_1_1D = models.update_max_seen_response(contribution1, max_seen_resp_1_1D)
                response_2, max_seen_resp_2_1D = models.update_max_seen_response(contribution2, max_seen_resp_2_1D)

                # update unitary model with response_1 and response_2
                sub1, sub1_like, train_x_sub1, train_y_sub1 = hmodel.update_model1_1D_Efficient(sub1, sub1_like, train_x_sub1,
                                                                                      train_y_sub1,
                                                                                      next_query_pins[:2],
                                                                                      contribution1,
                                                                                      training_iter=training_iter)

                sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model1_1D_Efficient(sub2, sub2_like, train_x_sub2,
                                                                                      train_y_sub2,
                                                                                      next_query_pins[2:],
                                                                                      contribution2,
                                                                                      training_iter=training_iter)

                sub1.eval()
                sub1_like.eval()

                sub2.eval()
                sub2_like.eval()

                # Make a prediction, observed_pred = likelihood
                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    observed_pred1 = models.make_prediction(sub1, test_x_1D, sub1_like)
                    observed_pred2 = models.make_prediction(sub2, test_x_1D, sub2_like)

                y_mu1 = observed_pred1.mean
                y_mu2 = observed_pred2.mean

                for i in range(len(prior_map)):
                    for j in range(len(prior_map)):
                        prior_map[i, j] = y_mu1[i] + y_mu2[j]

                prior_map_max = torch.max(prior_map)

                list_prior_map.append(np.array(prior_map / prior_map_max))
                list_objective_mean_map.append(np.array(np.reshape(observed_pred.mean, (10, 10))))

                master.mean_module.map = torch.nn.Parameter(prior_map / prior_map_max)

                master = hmodel.update_kernel_parameters(master, sub1, sub2)

                train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins,
                                                                  response)
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
                    sub1.eval()
                    sub1_like.eval()

                    sub2.eval()
                    sub2_like.eval()

                    # Make a prediction, observed_pred = likelihood, prediction_mean = mu
                    observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

                # acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred)

                exploration_score_2D, next_query_pins_exploration_2D = models.get_exploration_score(hierar_y_mu,
                                                                                                    ground_truth_max_2D,
                                                                                                    test_x_hier,
                                                                                                    X_2D, Y_2D)
                exploitation_score_2D = models.get_exploitation_score(next_query_value_mean, ground_truth_max_2D)
                """print(f'\nQuery Number: {q}')
                print(f'Next Query Pins: {next_query_pins_exploration_2D}')
                print(f'Next Query Value: {next_query_value_mean}')
                print(f'Exploration_Score: {torch.round(exploration_score_2D, decimals=4)}')
                print(f'Exploitation_Score: {torch.round(exploitation_score_2D, decimals=4)}')"""

                better_exploration_score.append(exploration_score_2D)
                better_exploitation_score.append(exploitation_score_2D)

            pred = hmodel.make_Hierarchique_prediction(master, x_hier, master.likelihood)
            master_like = master.likelihood(pred)
            heatmap_data.append(master_like.mean)
            print(f'\nRepetition {repetition} complete!\n')

        exploration_scores = []
        exploitation_scores = []

        for i in range(nbr_repetition):
            exploitation_scores.append(better_exploitation_score[i * nbr_query:(i + 1) * nbr_query])
            exploration_scores.append(better_exploration_score[i * nbr_query:(i + 1) * nbr_query])

        y = np.mean(exploration_scores, axis=0)
        over_explor.append(y)
        std = np.std(exploration_scores, axis=0)
        plt.plot(y, label='Exploration')
        plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)

        y = np.mean(exploitation_scores, axis=0)
        over_exploit.append(y)

        std = np.std(exploitation_scores, axis=0)
        plt.plot(y, label='Exploitation')
        plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)

        k = str(kappa).replace(".", ",")

        plt.legend()
        plt.ylim(0, 1.1)
        plt.title(f'Norm_Efficient HGP-BO {nbr_repetition} repetitions with kappa value {k}')
        plt.savefig(
            f'efficient_3D{folder_of_the_day}/differentiable_plots/Norm_Efficient_Prop_Neural_HGP-BO_{nbr_repetition}_repetitions_kappa_{k}')
        plt.close()

        vi.contour_plot_1D([sub1, sub2], test_x_1D, [test_y_1D / torch.max(test_y_1D), test_y_1D / torch.max(test_y_1D)],
                         f'/Heatmap_Neural_Efficient_HGP-BO_{nbr_repetition}_repetitions_kappa_{k}_query_{nbr_query}',
                         "efficient_3D", folder_of_the_day, 'N.A', neural=True)

    # Joint Section

    joint_plots(over_exploit, over_explor, k_vals, folder_of_the_day, nbr_query, nbr_repetition)


if __name__ == '__main__':

    import warnings

    warnings.filterwarnings('ignore')

    max_seen_resp_2D = 0
    max_seen_resp_1_1D = 0
    max_seen_resp_2_1D = 0
    list_prior_map = []
    list_objective_mean_map = []

    trainsC = Trains(clean_thresh=0.06)
    X_1D, Y_1D, Xmean_1D, Ymean_1D = make_dataset_1d(trainsC)
    test_x_1D = torch.tensor(Xmean_1D)
    test_y_1D = torch.tensor(Ymean_1D)
    ground_truth_max_1D = np.max(Ymean_1D)

    x_sub1 = torch.from_numpy(X_1D.copy())
    x_sub2 = torch.from_numpy(X_1D.copy())

    y_sub1 = torch.from_numpy(Y_1D[:, 0].copy())
    y_sub2 = torch.from_numpy(Y_1D[:, 0].copy())

    X_2D, Y_2D, Xmean_2D, Ymean_2D = make_dataset_2d(trainsC)
    ground_truth_max_2D = np.max(Ymean_2D)

    x_hier = torch.from_numpy(X_2D.copy())

    y_hier = torch.from_numpy(Y_2D[:, 0].copy())

    trainsC.plot_response_matrix()
    test_x_hier = torch.tensor(Xmean_2D)

    nbr_query = 100
    training_iter = 5
    nbr_repetition = 30
    nbr_rand_init = 5
    k_vals = np.array(range(12, 37, 2))/4

    training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals)
