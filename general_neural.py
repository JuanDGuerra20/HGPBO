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
import multiprocessing as mp
import pandas as pd



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

workspace_folder = (r'C:\Users\preda\PycharmProjects\HGPBO') #path to folder

os.chdir(workspace_folder)

kern_op = 'add_kernel'

def joint_plots(joint_exploit, joint_explor, kappa, gamma, nu_vals, folder_of_the_day, nbr_query, nbr_repetition, model_name):
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


def joint_performance(joint_exploit, joint_explor, kappa, gamma, nu_vals, folder_of_the_day, nbr_query, nbr_repetition, model_name):

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

def run_repetition(kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model, model_name, folder_of_the_day, final=False):
    prior_map_max = 0
    max_seen_resp_2D = 0
    max_seen_resp_1_1D = 0
    max_seen_resp_2_1D = 0
    ground_truth_max_1 = torch.max(y_sub1)
    ground_truth_max_2 = torch.max(y_sub2)
    ground_truth_max_hier = torch.max(y_hier)
    sub1_qc = torch.ones(len(test_x_1D))
    sub2_qc = torch.ones(len(test_x_1D))
    hier_qc = torch.ones(len(test_x_hier))

    heatmap_rep = []
    
    prior_map = torch.zeros(10, 10)
    list_queries = []
    list_query_values = []
    list_max_seen_1_1D = []
    list_max_seen_2_1D = []
    list_max_seen_2D = []
    list_acquisition_map = []
    list_next_value_mean = []
    better_exploitation_score = []
    better_exploration_score = []

    for q in tqdm(range(nbr_query)):

        if q == 0:
            # Need to initialize the model - Will be random in this method
            train_x_sub1, train_y_sub1 = random_initialization_1D(nbr_rand_init, trainsC,
                                                            max_seen_resp_1_1D, emg=EMG)
            train_x_sub2, train_y_sub2 = random_initialization_1D(nbr_rand_init, trainsC,
                                                            max_seen_resp_1_1D, emg=EMG)
            
            max_seen_resp_1_1D = torch.max(train_y_sub1)
            max_seen_resp_2_1D = torch.max(train_y_sub2)
            
            train_x_hier, train_y_hier = hmodel.random_initialization(nbr_rand_init, EMG, trainsC,
                                                                max_seen_resp_2D, DT)
            train_x_hier = torch.tensor(train_x_hier)
            train_y_hier = torch.tensor(train_y_hier)
            max_seen_resp_2D = torch.max(train_y_hier)
            
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

            for i in range(len(train_x_sub1)):
                sub1_qc = sub1.increment_q_n(sub1_qc, train_x_sub1[i], test_x_1D)
                sub2_qc = sub2.increment_q_n(sub2_qc, train_x_sub2[i], test_x_1D)

            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred1 = models.make_prediction(sub1, test_x_1D, sub1_like)
                observed_pred2 = models.make_prediction(sub2, test_x_1D, sub2_like)
            y_mu1 = observed_pred1.mean
            y_mu2 = observed_pred2.mean

            y_conf1 = observed_pred1.stddev
            y_conf2 = observed_pred2.stddev

            p1 = y_mu1 + gamma * y_conf1 / (torch.sqrt(sub1_qc))
            p2 = y_mu2 + gamma * y_conf2 / (torch.sqrt(sub2_qc))

            for i in range(len(prior_map)):
                for j in range(len(prior_map)):
                    prior_map[i, j] = (p1[i] + p2[j]) / 2

            prior_map_max = torch.max(prior_map)
            list_prior_map.append(np.array(prior_map / prior_map_max))

            prior_hierarchical_kernel = hmodel.hierarchical_kernel("add_kernel", sub1, sub2)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            master = hierarchical_model(train_x_hier, train_y_hier / max_seen_resp_2D, likelihood,
                                                            prior_hierarchical_kernel,
                                                            prior_map / prior_map_max, kernel_op='add_kernel',
                                                            sub_models=[sub1, sub2],
                                                            kappa=kappa, query_counter=hier_qc)
            for i in range(nbr_rand_init):
                hier_qc = master.increment_q_n(hier_qc, train_x_hier[i], test_x_hier)
            
            master.eval()
            likelihood.eval()
            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

            list_objective_mean_map.append(np.array(np.reshape(observed_pred.mean, (10, 10))))

        vi.comparison(list_prior_map, list_objective_mean_map, q, Xmean_1D)

        acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred, hier_qc)
        list_acquisition_map.append(list_acquisition_map)

        next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)

        next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins,
                                                                                        x_hier,
                                                                                        y_hier)
        list_next_value_mean.append(next_query_value_mean)

        list_queries.append(next_query_pins)
        list_query_values.append(next_query_value_random)

        y_mu_point_a = hmodel.get_y_mu_point_value(next_query_pins[:2], y_mu1, test_x_1D)
        y_mu_point_b = hmodel.get_y_mu_point_value(next_query_pins[2:], y_mu2, test_x_1D)
        
        y_conf_point_a = hmodel.get_y_mu_point_value(next_query_pins[:2], y_conf1, test_x_1D)
        y_conf_point_b = hmodel.get_y_mu_point_value(next_query_pins[2:], y_conf2, test_x_1D)
        
        y_qc_a = hmodel.get_y_mu_point_value(next_query_pins[:2], sub1_qc, test_x_1D)
        y_qc_b = hmodel.get_y_mu_point_value(next_query_pins[2:], sub2_qc, test_x_1D)

        next_query_value_random, max_seen_resp_2D = models.update_max_seen_response_no_norm(next_query_value_random,
                                                                                    max_seen_resp_2D)

        list_max_seen_1_1D.append(max_seen_resp_1_1D)
        list_max_seen_2_1D.append(max_seen_resp_2_1D)
        list_max_seen_2D.append(max_seen_resp_2D)

        response = torch.tensor(next_query_value_random)

        
        cont1 = y_mu_point_a - gamma * torch.nan_to_num(y_conf_point_a / torch.sqrt(y_qc_a))
        cont2 = y_mu_point_b - gamma * torch.nan_to_num(y_conf_point_b / torch.sqrt(y_qc_b))

        div = torch.exp(cont1) + torch.exp(cont2)

        contribution1 = torch.nan_to_num(response * torch.exp(cont1) / div)
        contribution2 = torch.nan_to_num(response * torch.exp(cont2) / div)

        response_1 = sub1.update_max_seen_response_no_norm(contribution1, max_seen_resp_1_1D)
        response_2 = sub2.update_max_seen_response_no_norm(contribution2, max_seen_resp_2_1D)

        # Potentially could make this more efficient by incorporating it into the next finder
        sub1_qc = sub1.increment_q_n(sub1_qc, next_query_pins[:2], test_x_1D)
        sub2_qc = sub2.increment_q_n(sub2_qc, next_query_pins[2:], test_x_1D)
        hier_qc = master.increment_q_n(hier_qc, next_query_pins, test_x_hier)

        # update unitary model with response_1 and response_2
        sub1, sub1_like, train_x_sub1, train_y_sub1 = hmodel.update_model1_1D_max_seen(sub1, sub1_like, train_x_sub1,
                                                                                train_y_sub1,
                                                                                next_query_pins[:2],
                                                                                contribution1, max_seen_resp_1_1D,
                                                                                training_iter=training_iter)

        sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model1_1D_max_seen(sub2, sub2_like, train_x_sub2,
                                                                                train_y_sub2,
                                                                                next_query_pins[2:],
                                                                                contribution2, max_seen_resp_2_1D,
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

        y_conf1 = observed_pred1.stddev
        y_conf2 = observed_pred2.stddev

        p1 = y_mu1 + gamma * y_conf1 / (torch.sqrt(sub1_qc))
        p2 = y_mu2 + gamma * y_conf2 / (torch.sqrt(sub2_qc))

        for i in range(len(prior_map)):
            for j in range(len(prior_map)):
                prior_map[i, j] = (p1[i] + p2[j]) / 2

        prior_map_max = torch.max(prior_map)

        list_prior_map.append(np.array(prior_map / prior_map_max))
        list_objective_mean_map.append(np.array(np.reshape(observed_pred.mean, (10, 10))))

        master.mean_module.map = torch.nn.Parameter(prior_map / prior_map_max)

        master = hmodel.update_kernel_parameters(master, sub1, sub2)

        train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins,
                                                            response)
        master.set_train_data(train_x_hier, train_y_hier / max_seen_resp_2D, strict=False)

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            # Find optimal model hyperparameters
            master.train()
            likelihood.train()

            master, likelihood = master.Hoptimize(likelihood, training_iter, train_x_hier, train_y_hier/max_seen_resp_2D,
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
                                                                                            x_hier, y_hier)
        exploitation_score_2D = models.get_exploitation_score(next_query_value_mean, ground_truth_max_2D)
        """print(f'\nQuery Number: {q}')
        print(f'Next Query Pins: {next_query_pins_exploration_2D}')
        print(f'Next Query Value: {next_query_value_mean}')
        print(f'Exploration_Score: {torch.round(exploration_score_2D, decimals=4)}')
        print(f'Exploitation_Score: {torch.round(exploitation_score_2D, decimals=4)}')"""

        better_exploration_score.append(exploration_score_2D)
        better_exploitation_score.append(exploitation_score_2D)
        master_like = master.likelihood(observed_pred)
        heatmap_rep.append(master_like.mean.detach().cpu().numpy())

    k = str(kappa).replace('.', ',')
    g = str(gamma).replace('.', ',')
    n = str(nu).replace('.', ',')

    vi.contour_plot_1D(master.sub_models, test_x_1D,
                        [test_y_1D / torch.max(test_y_1D), test_y_1D / torch.max(test_y_1D)],
                        f'/contour/Contour_Neural_{model_name}_HGP-BO_nbr_query_{nbr_query}_kappa_{k}_gamma_{g}_nu_{n}_pid_{os.getpid()}',
                        model_name.lower(), folder_of_the_day, "Neural", neural=True)
    if final:
        return master, sub1, sub2, better_exploration_score, better_exploitation_score, heatmap_rep
    else:
        return better_exploration_score, better_exploitation_score, heatmap_rep

def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals, hierarchical_model, multi):
    if hierarchical_model == hmodel.Efficient_UCB_Hierarchical_GP:
        model_name = "Efficient"

    elif hierarchical_model == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
        model_name = "Lossless_Efficient"
    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"C:/Users/preda/PycharmProjects/HGPBO/{model_name.lower()}"
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

    list_prior_map = []
    list_objective_mean_map = []
    final_exploitation_metric = []
    final_exploration_metric = []

    for kappa in k_vals:
        over_exploit = []
        over_explor = []
        for gamma in g_vals:
            for nu in nu_vals:
                better_exploration_score = []
                better_exploitation_score = []
                heatmap_data = []
                processes = []

                if multi:
                    with mp.Pool(processes=nbr_repetition - 1) as pool:

                        # running the repetitions in parallel except for the last one
                        for i in range(nbr_repetition - 1):
                            p = pool.apply_async(run_repetition, (kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model, model_name, folder_of_the_day,))
                            processes.append(p)

                        # must run the final block manually to allow return of the models
                        master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep = run_repetition(kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model, model_name, folder_of_the_day, final=True)

                        better_exploration_score.append(rep_exploration_score)
                        better_exploitation_score.append(rep_exploitation_score)
                        heatmap_data.append(heatmap_rep)
                        for i, proc in enumerate(processes):
                            rep_exploration_score, rep_exploitation_score, heatmap_rep = proc.get()

                            better_exploration_score.append(rep_exploration_score)
                            better_exploitation_score.append(rep_exploitation_score)
                            heatmap_data.append(heatmap_rep)


                else:

                    for i in range(nbr_repetition ):

                        master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep = run_repetition(
                            kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model, model_name, folder_of_the_day, final=True)
                        better_exploration_score.append(rep_exploration_score)
                        better_exploitation_score.append(rep_exploitation_score)
                        heatmap_data.append(heatmap_rep)

                heatmap_data = np.array(heatmap_data)

                k = str(kappa).replace('.', ',')
                g = str(gamma).replace('.', ',')
                n = str(nu).replace('.', ',')

                y = np.mean(better_exploration_score, axis=0)
                over_explor.append(y)
                std = np.std(better_exploration_score, axis=0)
                plt.plot(y, label='Exploration')
                plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)

                y = np.mean(better_exploitation_score, axis=0)
                over_exploit.append(y)

                std = np.std(better_exploitation_score, axis=0)
                plt.plot(y, label='Exploitation')
                plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)

                r2 = vi.heatmap_r_score(heatmap_data, test_y_hier)
                plt.plot(r2, label="Heatmap R2")

                rand = np.random.rand(*heatmap_data.shape)
                random_r2 = vi.heatmap_r_score(rand, test_y_hier)
                plt.plot(random_r2, label="Random Heatmap R2")

                plt.legend()
                plt.ylim(0, 1.1)
                plt.title(f'{model_name} HGP-BO {nbr_repetition} repetitions with Kappa {k} Gamma {g} Nu {n}')
                plt.savefig(
                    f'{model_name.lower()}{folder_of_the_day}/differentiable_plots/{model_name}_Neural_HGP-BO_{nbr_repetition}_repetitions_kappa_{k}_gamma_{g}_nu_{n}')
                plt.close()

                """vi.model_heatmap(heatmap_data[:, -1, :], test_x_hier, test_y_hier,
                                 f'/Heatmap_Neural_{model_name}_HGPBO_{nbr_repetition}_repetitions_kappa_{k}_gamma_{g}_nu_{n}',
                                 model_name.lower(), folder_of_the_day, "Neural", neural=True)"""
                data = np.mean(heatmap_data[:, -1, :], axis=0)

                re_output = np.reshape(data, test_y_hier.shape)
                df = pd.DataFrame(re_output)
                df.to_csv(
                    f'{model_name.lower()}{folder_of_the_day}/csv/{model_name}_Prop_HGPBO_{nbr_repetition}_repetitions_kappa_{k}_gamma_{g}.csv')
                df = pd.DataFrame(y_hier)
                df.to_csv(
                    f'{model_name.lower()}{folder_of_the_day}/csv/True_State_Space_Values.csv')

                print(f'\n{model_name} Kappa {k} Gamma {g} Nu {n} complete!\n')

            # Joint Section

            joint_plots(over_exploit, over_explor, kappa, gamma, nu_vals, folder_of_the_day, nbr_query,
                              nbr_repetition, model_name)

            joint_performance(over_exploit, over_explor, kappa, gamma, nu_vals, folder_of_the_day, nbr_query,
                              nbr_repetition, model_name)

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
    test_y_hier = torch.tensor(Ymean_2D)


    nbr_query = 40
    training_iter = 5
    nbr_repetition = 3
    nbr_rand_init = 5
    k_vals = [2]
    g_vals = [6]
    nu_vals = [0.5, 1.5, 2.5]
    multi = False
    h_model = [hmodel.Lossless_Efficient_UCB_Hierarchical_GP]
    process = []
    for h in h_model:
        if multi:
            p = mp.Process(target=training_procedure, args=(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals,
                               h, multi,))
            process.append(p)
            p.start()
            print(f"ID of process: {p.pid}")
        else:
            training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals,
                               h, multi)

