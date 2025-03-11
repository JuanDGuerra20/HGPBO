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

xy2ch = [[2, 6, 10, 14, 9],
         [13, 17, 21, 18, 22]]
ch2xy = {}
for ch in CHS:
    x, y = np.where(np.array(xy2ch) == ch)
    ch2xy[ch] = [x[0], y[0]]

DTS = [0, 10, 20, 40, 60, 80, 100]
EMG = 4
N_EMGS = 7

DT = 60
EMG = 4

training_iter = 5
q = 0
nbr_rdm_points_ini = 5  # Number of random points to initialize the model
nbr_rdm_points_data = 20  # Number of random points from the Ground Truth (GT) EMG responses to create a dataset
max_seen_resp = 0  # To store the maximum response observed, and normalize between 0 and 1 (arbitrary choice)
nbr_repetition = 5  # Number of repetition of the entire process to average the results
nbr_pins = 100  # Number of pins in he input space (2*matrix of 2*5 pins)

name_code = 'HGP_BO-test6-priorMAP-1model1D'
current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
current_dateday = datetime.now().strftime("%Y-%m-%d")

workspace_folder = (r'C:\Users\preda\PycharmProjects\HGPBO')  # path to folder

os.chdir(workspace_folder)

kern_op = 'add_kernel'
folder_of_the_day = (str(workspace_folder) + f'/vanilla/data-' + str(name_code) + str(current_dateday))


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def joint_plots(joint_exploit, joint_explor, k_vals, folder_of_the_day, nbr_query, nbr_repetition):
    for i, kappa in enumerate(k_vals):
        plt.plot(joint_exploit[i], label=f'K {kappa}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploitation Score')
    plt.title(f'Joint vanilla Propagation HGPBO {nbr_repetition} Exploitation')
    plt.savefig(
        f'vanilla{folder_of_the_day}/differentiable_plots/Joint_vanilla_Neural_Propagation_HGPBO_{nbr_repetition}_Exploitation_query_{nbr_query}')
    plt.close()

    for i, kappa in enumerate(k_vals):
        plt.plot(joint_explor[i], label=f'K {kappa}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploration Score')
    plt.title(f'Joint vanilla Propagation HGPBO {nbr_repetition} Exploration')
    plt.savefig(
        f'vanilla{folder_of_the_day}/differentiable_plots/Joint_vanilla_Neural_Propagation_HGPBO_{nbr_repetition}_Exploration_query_{nbr_query}')
    plt.close()


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals):
    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"C:/Users/preda/PycharmProjects/HGPBO/vanilla"
    folder_of_the_day = '/data-' + str(current_dateday)
    if os.path.exists(workspace + folder_of_the_day):
        print('Data folder is ready')
    else:
        os.mkdir(workspace + folder_of_the_day)
        print('Data folder created')
        os.mkdir(workspace + folder_of_the_day + '/contour')
        print("Contour folder created")
        os.mkdir(workspace + folder_of_the_day + '/differentiable_plots')
        print("CSV folder created")
        os.mkdir(workspace + folder_of_the_day + '/csv')
        print("HP folder created")
        os.mkdir(workspace + folder_of_the_day + '/hp_analysis')
        print("Model folder created")
        os.mkdir(workspace + folder_of_the_day + '/models')

    ground_truth_max_hier = torch.max(y_hier)

    heatmap_data = []
    list_max_seen_2D = []

    over_exploit = []
    over_explor = []

    hier_qc = torch.ones(len(test_x_hier))
    for kappa in k_vals:

        better_exploration_score = []
        better_exploitation_score = []
        for repetition in range(nbr_repetition):
            max_seen_resp_2D = 0
            for q in tqdm(range(nbr_query)):
                if q == 0:

                    train_x_hier, train_y_hier = hmodel.random_initialization(nbr_rand_init, EMG, trainsC,
                                                                              max_seen_resp_2D, DT, max_update=True)

                    train_x_hier = torch.tensor(train_x_hier, dtype=torch.float64)
                    train_y_hier = torch.tensor(train_y_hier, dtype=torch.float64)

                    max_seen_resp_2D = torch.max(train_y_hier)

                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    master = ExactGPModel(train_x_hier, train_y_hier, likelihood)

                    optimizer = torch.optim.Adam(master.parameters(), lr=1e-3)
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, master)

                    master.eval()
                    likelihood.eval()

                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)


                acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred, hier_qc)

                next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)

                next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins,
                                                                                             x_hier,
                                                                                             y_hier)


                next_query_value_random, max_seen_resp_2D = models.update_max_seen_response_no_norm(
                    next_query_value_random,
                    max_seen_resp_2D)

                list_max_seen_2D.append(max_seen_resp_2D)

                response = torch.tensor(next_query_value_random)

                train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins,
                                                                  response)
                master.set_train_data(train_x_hier, train_y_hier / max_seen_resp_2D, strict=False)


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

                # Make a prediction, observed_pred = likelihood, prediction_mean = mu
                observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

                # acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred)

                exploration_score_2D, next_query_pins_exploration_2D = models.get_exploration_score(hierar_y_mu,
                                                                                                    ground_truth_max_2D,
                                                                                                    test_x_hier,
                                                                                                    x_hier, y_hier)
                exploitation_score_2D = models.get_exploitation_score(next_query_value_mean, ground_truth_max_2D)
                """print(f'\nQuery Number: {q}')
                print(f'Next Query Pins: {next_query_pins_exploration_2D}')
                print(f'Next Query Value: {next_query_value_mean}')
                print(f'Exploration_Score: {torch.round(exploration_score_2D, decimals=4)}')
                print(f'Exploitation_Score: {torch.round(exploitation_score_2D, decimals=4)}')"""

                better_exploration_score.append(exploration_score_2D)
                better_exploitation_score.append(exploitation_score_2D)

                heatmap_data.append(observed_pred.mean.detach().cpu().numpy())

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
        heatmap_data = np.array(heatmap_data)
        heatmap_data = np.reshape(heatmap_data, (-1, nbr_query, len(test_y_hier)))
        r2 = vi.heatmap_r_score(heatmap_data, test_y_hier)
        plt.plot(r2, label="Parent R2")

        k = str(kappa).replace(".", ",")

        plt.legend()
        plt.ylim(0, 1.1)
        plt.title(f'vanilla HGP-BO {nbr_repetition} repetitions with kappa value {k}')
        plt.savefig(
            f'vanilla{folder_of_the_day}/differentiable_plots/vanilla_Neural_HGP-BO_{nbr_repetition}_repetitions_kappa_{k}_query_{nbr_query}')
        plt.close()

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

    #trainsC.plot_response_matrix()
    test_x_hier = torch.tensor(Xmean_2D)
    test_y_hier = torch.tensor(Ymean_2D)

    nbr_query = 100
    training_iter = 5
    nbr_repetition = 30
    nbr_rand_init = 1
    k_vals = [2]
    training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals)
