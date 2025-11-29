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
import warnings

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

    for n in nu_vals:
        names.append(f'kappa_{kappa}_gamma_{gamma}_nu_{n}')
    fig, ax = plt.subplots(figsize=(nbr_query/5, len(nu_vals)*3))
    heatmap(joint_explor, xticklabels=list(range(nbr_query)), yticklabels=names, cmap='coolwarm', ax=ax)

    plt.ylabel(f'Model Type')
    plt.xlabel(f'Training Step')
    plt.title(f'Joint HP Exploration Performance over {nbr_repetition} repetitions')
    plt.tight_layout()
    plt.savefig(
        f'{model_name.lower()}{folder_of_the_day}/hp_analysis/HP_Exploration_Performance_{nbr_repetition}_repetitions_kappa_{kappa}_gamma_{gamma}')
    plt.close()

def run_repetition(kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model, model_name, folder_of_the_day, final=False, children=[], visualize=True, seed=True):
    if type(seed) != bool:
        np.random.seed(seed)

    warnings.filterwarnings('ignore')    # Setting up the data
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

    test_x_hier = torch.tensor(Xmean_2D)
    test_y_hier = torch.tensor(Ymean_2D)


    max_seen_resp_2D = 0
    max_seen_resp_1_1D = 0
    max_seen_resp_2_1D = 0
    sub1_qc = torch.ones(len(test_x_1D))
    sub2_qc = torch.ones(len(test_x_1D))
    hier_qc = torch.ones(len(test_x_hier))

    heatmap_rep = []
    
    prior_map = torch.zeros(10, 10)
    list_queries = []
    list_prior_map = []
    list_objective_mean_map = []

    list_query_values = []
    list_max_seen_1_1D = []
    list_max_seen_2_1D = []
    list_max_seen_2D = []
    list_acquisition_map = []
    list_next_value_mean = []
    better_exploitation_score = []
    better_exploration_score = []
    child_1_r2 = []
    child_2_r2 = []

    for q in tqdm(range(nbr_query)):

        if q == 0:
            # Need to initialize the model - Will be random in this method
            train_x_sub1, train_y_sub1 = random_initialization_1D(nbr_rand_init, trainsC,
                                                            max_seen_resp_1_1D, emg=EMG)
            train_x_sub2, train_y_sub2 = random_initialization_1D(nbr_rand_init, trainsC,
                                                            max_seen_resp_1_1D, emg=EMG)
            
            max_seen_resp_1_1D = torch.max(train_y_sub1)
            max_seen_resp_2_1D = torch.max(train_y_sub2)

            """max_seen_resp_1_1D = torch.max(train_y_sub1)
            max_seen_resp_2_1D = torch.max(train_y_sub2)
            max_seen_resp_2D = torch.max(train_y_hier)

            train_y_sub1 = train_y_sub1 / max_seen_resp_1_1D
            train_y_sub2 = train_y_sub2 / max_seen_resp_2_1D
            train_y_hier = train_y_hier / max_seen_resp_2D"""

            # Need to modify this section such that the model is receiving the partial contribution
            """train_x_sub1, train_y_sub1 = update_training_data(train_x_sub1, train_y_sub1,
                                                                train_x_hier[:, :2], train_y_hier)
            train_x_sub2, train_y_sub2 = update_training_data(train_x_sub2, train_y_sub2,
                                                                train_x_hier[:, 2:], train_y_hier)"""

            sub1_like = gpytorch.likelihoods.GaussianLikelihood()
            sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1, sub1_like, query_counter=sub1_qc, nu=nu)

            sub2_like = gpytorch.likelihoods.GaussianLikelihood()
            sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like, query_counter=sub2_qc, nu=nu)

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
            next_query_pins = models.get_next_query_pins(torch.flatten(prior_map), test_x_hier)

            next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins,
                                                                                         x_hier,
                                                                                         y_hier)
            response = torch.tensor(next_query_value_random)
            train_x_hier, train_y_hier = torch.reshape(next_query_pins, (1, 4)), torch.reshape(response, (1,))

            max_seen_resp_2D = torch.max(train_y_hier)
            prior_hierarchical_kernel = hmodel.hierarchical_kernel("add_kernel", sub1, sub2)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            """master = hierarchical_model(train_x_hier, train_y_hier/max_seen_resp_2D, test_x_hier, likelihood,
                                        prior_hierarchical_kernel,
                                        prior_map / prior_map_max, kernel_op='add_kernel',
                                        sub_models=[sub1, sub2],
                                        kappa=kappa, query_counter=hier_qc)"""
            master = hierarchical_model(train_x_hier, train_y_hier - train_y_hier.mean(), test_x_hier, likelihood,
                                        prior_hierarchical_kernel,
                                        prior_map / prior_map_max, kernel_op='add_kernel',
                                        sub_models=[sub1, sub2],
                                        kappa=kappa, query_counter=hier_qc)
            #for i in range(nbr_rand_init):
            for i in range(len(train_x_hier)):
                hier_qc = master.increment_q_n(hier_qc, train_x_hier[i], test_x_hier)
            
            master.eval()
            likelihood.eval()
            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

            list_objective_mean_map.append(np.array(np.reshape(observed_pred.mean, (10, 10))))

        #vi.comparison(list_prior_map, list_objective_mean_map, q, Xmean_1D)

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

        cont1 = y_mu_point_a + gamma * torch.nan_to_num(y_conf_point_a / torch.sqrt(y_qc_a))
        cont1_scaled = torch.nan_to_num(cont1/torch.max(y_mu1 + gamma * torch.nan_to_num(y_conf1/ torch.sqrt(sub1_qc))))

        cont2 = y_mu_point_b + gamma * torch.nan_to_num(y_conf_point_b / torch.sqrt(y_qc_b))
        cont2_scaled = torch.nan_to_num(cont2/torch.max(y_mu2 + gamma * torch.nan_to_num(y_conf2/ torch.sqrt(sub2_qc))))

        div = torch.exp(cont1_scaled) + torch.exp(cont2_scaled)

        contribution1 = torch.nan_to_num(response * torch.exp(cont1_scaled) / div)
        contribution2 = torch.nan_to_num(response * torch.exp(cont2_scaled) / div)

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
                                                                                contribution1, False,
                                                                                training_iter=training_iter)

        sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model1_1D_max_seen(sub2, sub2_like, train_x_sub2,
                                                                                train_y_sub2,
                                                                                next_query_pins[2:],
                                                                                contribution2, False,
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

        master.mean_module.map = torch.nn.Parameter(prior_map/prior_map_max)

        master = hmodel.update_kernel_parameters(master, sub1, sub2)

        train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins,
                                                            response)
        #master.set_train_data(train_x_hier, train_y_hier/max_seen_resp_2D, strict=False)
        master.set_train_data(train_x_hier, (train_y_hier - train_y_hier.mean()) / train_y_hier.std(), strict=False)
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            # Find optimal model hyperparameters
            master.train()
            likelihood.train()

            """master, likelihood = master.Hoptimize(likelihood, training_iter, train_x_hier, train_y_hier/max_seen_resp_2D,
                                                    verbose=False)"""
            master, likelihood = master.Hoptimize(likelihood, training_iter, train_x_hier,
                                                  (train_y_hier - train_y_hier.mean()) / train_y_hier.std(),
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
            c1_r2, c2_r2 = vi.child_contour_r2(master.sub_models, [x_sub1, x_sub2],
                                               [y_sub1, y_sub2])

            child_1_r2.append(c1_r2)
            child_2_r2.append(c2_r2)

        # acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred)

        """exploration_score_2D, next_query_pins_exploration_2D = models.get_exploration_score(hierar_y_mu,
                                                                                            ground_truth_max_2D,
                                                                                            test_x_hier,
                                                                                            x_hier, y_hier)"""
        instantaneous_regret, next_query_pins_exploration_2D = models.get_instantaneous_regret(hierar_y_mu,
                                                                                               ground_truth_max_2D,
                                                                                               test_x_hier,
                                                                                               x_hier, y_hier)
        exploitation_score_2D = models.get_exploitation_score(next_query_value_mean, ground_truth_max_2D)
        """print(f'\nQuery Number: {q}')
        print(f'Next Query Pins: {next_query_pins_exploration_2D}')
        print(f'Next Query Value: {next_query_value_mean}')
        print(f'Exploration_Score: {torch.round(exploration_score_2D, decimals=4)}')
        print(f'Exploitation_Score: {torch.round(exploitation_score_2D, decimals=4)}')"""

        better_exploration_score.append(instantaneous_regret)
        better_exploitation_score.append(exploitation_score_2D)
        master_like = master.likelihood(observed_pred)
        heatmap_rep.append(master_like.mean.detach().cpu().numpy())

    k = str(kappa).replace('.', ',')
    g = str(gamma).replace('.', ',')
    n = str(nu).replace('.', ',')

    vi.contour_plot_1D(master.sub_models, [test_x_1D, test_x_1D], [test_y_1D, test_y_1D], [train_y_sub1, train_y_sub2],
                       f'/contour/Contour_init_{nbr_rand_init}_train_iter_{training_iter}_k_{k}_g_{g}_nu_{n}',
                       model_name.lower(), folder_of_the_day, "",parent=master, query=q, visualize=visualize, neural=True)
    '''if multi:
        vi.contour_plot_1D(master.sub_models, test_x_1D,
                            [test_y_1D / torch.max(test_y_1D), test_y_1D / torch.max(test_y_1D)],
                            f'/contour/Contour_Neural_{model_name}_HGP-BO_nbr_query_{nbr_query}_kappa_{k}_gamma_{g}_nu_{n}_nbr_rand_{nbr_rand_init}_pid_{os.getpid()}',
                            model_name.lower(), folder_of_the_day, "Neural", neural=True)
    else:
        rand_id = np.random.randint(99999)
        vi.contour_plot_1D(master.sub_models, test_x_1D,
                           [test_y_1D / torch.max(test_y_1D), test_y_1D / torch.max(test_y_1D)],
                           f'/contour/Contour_Neural_{model_name}_HGP-BO_nbr_query_{nbr_query}_kappa_{k}_gamma_{g}_nu_{n}_nbr_rand_{nbr_rand_init}_pid_{rand_id}',
                           model_name.lower(), folder_of_the_day, "Neural", neural=True)'''
    if final:
        return master, sub1, sub2, better_exploration_score, better_exploitation_score, heatmap_rep, child_1_r2, child_2_r2
    else:
        return better_exploration_score, better_exploitation_score, heatmap_rep, child_1_r2, child_2_r2

def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals, hierarchical_model, multi, seed, children=[], visualize=True):
    if hierarchical_model == hmodel.Efficient_UCB_Hierarchical_GP:
        model_name = "Efficient"

    elif hierarchical_model == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
        model_name = "Lossless_Efficient"
    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)
    trainsC = Trains(clean_thresh=0.06)
    X_2D, Y_2D, Xmean_2D, Ymean_2D = make_dataset_2d(trainsC)

    y_hier = torch.from_numpy(Y_2D[:, 0].copy())

    test_x_hier = torch.tensor(Xmean_2D)
    test_y_hier = torch.tensor(Ymean_2D)

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
        print("PNG folder created")
        os.mkdir(workspace + folder_of_the_day + '/png')


    child_1_r2_data = []
    child_2_r2_data = []
    list_models = []
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
                    print(f'Multi Processing: {multi} with reps {nbr_repetition}')
                    with mp.Pool(processes=nbr_repetition - 1) as pool:

                        # running the repetitions in parallel except for the last one
                        for i in range(nbr_repetition - 1):
                            p = pool.apply_async(run_repetition, (kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model, model_name, folder_of_the_day, False, children, visualize, seed[i],))
                            processes.append(p)

                        # must run the final block manually to allow return of the models
                        try:
                            master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model, model_name, folder_of_the_day, True, children, visualize, seed=seed[i])
                        except:
                            try:
                                master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                    kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model,
                                    model_name, folder_of_the_day, True, children, visualize, seed=seed[i])
                            except:
                                try:
                                    master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                        kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model,
                                        model_name, folder_of_the_day, True, children, visualize, seed=seed[i])
                                except:
                                    raise Exception('Multi Processing Final did not run properly')
                        better_exploration_score.append(rep_exploration_score)
                        better_exploitation_score.append(rep_exploitation_score)
                        heatmap_data.append(heatmap_rep)
                        child_1_r2_data.append(child_1_r2)
                        child_2_r2_data.append(child_2_r2)
                        for i, proc in enumerate(processes):
                            try:
                                rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = proc.get()

                                better_exploration_score.append(rep_exploration_score)
                                better_exploitation_score.append(rep_exploitation_score)
                                heatmap_data.append(heatmap_rep)
                                child_1_r2_data.append(child_1_r2)
                                child_2_r2_data.append(child_2_r2)
                            except:
                                continue
                else:

                    for i in range(nbr_repetition ):
                        try:
                            master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model,
                                model_name, folder_of_the_day, True, children, visualize, seed=seed[i])
                            better_exploration_score.append(rep_exploration_score)
                            better_exploitation_score.append(rep_exploitation_score)
                            heatmap_data.append(heatmap_rep)
                            child_1_r2_data.append(child_1_r2)
                            child_2_r2_data.append(child_2_r2)
                        except:
                            try:
                                master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                    kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model,
                                    model_name, folder_of_the_day, True, children, visualize, seed=seed[i])
                                better_exploration_score.append(rep_exploration_score)
                                better_exploitation_score.append(rep_exploitation_score)
                                heatmap_data.append(heatmap_rep)
                                child_1_r2_data.append(child_1_r2)
                                child_2_r2_data.append(child_2_r2)
                            except:
                                try:
                                    master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                        kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter, hierarchical_model,
                                        model_name, folder_of_the_day, True, children, visualize, seed=seed[i])
                                    better_exploration_score.append(rep_exploration_score)
                                    better_exploitation_score.append(rep_exploitation_score)
                                    heatmap_data.append(heatmap_rep)
                                    child_1_r2_data.append(child_1_r2)
                                    child_2_r2_data.append(child_2_r2)
                                except:
                                    try:
                                        master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                            kappa, gamma, nu, nbr_query, nbr_rand_init, training_iter,
                                            hierarchical_model,
                                            model_name, folder_of_the_day, True, children, visualize, seed=seed[i])
                                        better_exploration_score.append(rep_exploration_score)
                                        better_exploitation_score.append(rep_exploitation_score)
                                        heatmap_data.append(heatmap_rep)
                                        child_1_r2_data.append(child_1_r2)
                                        child_2_r2_data.append(child_2_r2)
                                    except:
                                        continue


                k = str(kappa).replace('.', ',')
                g = str(gamma).replace('.', ',')
                n = str(nu).replace('.', ',')
                heatmap_data = np.array(heatmap_data)
                torch.save(master.state_dict(),
                           f'{model_name.lower()}{folder_of_the_day}/models/kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries.pth')

                y = np.mean(better_exploration_score, axis=0)
                y = np.insert(y, 0, np.zeros(2*nbr_rand_init))[:nbr_query]
                over_explor.append(y)
                std = np.std(better_exploration_score, axis=0) / np.sqrt(len(better_exploration_score))
                std = np.insert(std, 0, np.zeros(2*nbr_rand_init))

                plt.plot(y[:nbr_query], label='Exploration')
                plt.fill_between(range(len(y[:nbr_query])), y[:nbr_query] - std[:nbr_query], y[:nbr_query] + std[:nbr_query], alpha=0.4)
                print(f"explor {y[-1]}")
                """y = np.mean(better_exploitation_score, axis=0)
                y = np.insert(y, 0, np.zeros(2*nbr_rand_init))[:nbr_query]

                over_exploit.append(y)
                """
                r2 = vi.heatmap_r_score(heatmap_data, test_y_hier)
                r2 = np.insert(r2, 0, np.zeros((nbr_rand_init * 2, 1)), axis=1)[:, :nbr_query]
                r2_avg = np.mean(r2, axis=0)
                r2_std = np.std(r2, axis=0) / np.sqrt(len(r2))
                #r2_std = np.insert(r2_std, 0, np.zeros((2 - len(children)) *nbr_rand_init))[:nbr_query]
                print(f"r2 avg : {r2_avg[-1]}")

                plt.plot(r2_avg, label="Parent R2")
                plt.fill_between(range(len(r2_std)), r2_avg-r2_std, r2_avg + r2_std, alpha=0.4)

                all_children = np.concatenate([child_1_r2_data, child_2_r2_data])
                all_children = np.insert(all_children, 0, np.zeros((nbr_rand_init * 2, 1)), 1)[:, :nbr_query]

                avg_child = np.mean(all_children, axis=0)
                std_child = np.std(all_children, axis=0) / np.sqrt(len(all_children))
                print(f"child avg : {avg_child[-1]}")
                plt.plot(avg_child, label='Child Avg R2')
                plt.fill_between(range(len(avg_child)), avg_child - std_child, avg_child + std_child, alpha=0.4)
                auc = np.sum(y + avg_child + r2_avg)
                print(f"auc {auc}")

                plt.legend()
                plt.ylim(-0.1, 1.1)
                plt.title(
                    f'{model_name} HGP-BO {nbr_repetition} repetitions with kappa {k} Gamma {g} Nu {n} Init {nbr_rand_init}')
                plt.savefig(
                    f'{model_name.lower()}{folder_of_the_day}/differentiable_plots/{model_name}_Prop_HGPBO_{nbr_repetition}_repetitions_init_{nbr_rand_init}_training_iter_{training_iter}_kappa_{k}_gamma_{g}_nu_{n}.svg')
                plt.savefig(
                    f'{model_name.lower()}{folder_of_the_day}/png/{model_name}_Prop_HGPBO_{nbr_repetition}_repetitions_init_{nbr_rand_init}_training_iter_{training_iter}_kappa_{k}_gamma_{g}_nu_{n}.png')

                plt.close()

                vi.model_heatmap(heatmap_data[:, -1, :], test_x_hier, test_y_hier,
                                 f'/Heatmap_Neural_{model_name}_Neural_{nbr_repetition}_reps_k_{k}_g_{g}_nu_{n}_rand_init_{nbr_rand_init}',
                                 model_name.lower(), folder_of_the_day, "Neural", neural=True)
                data = np.mean(heatmap_data[:, -1, :], axis=0)

                re_output = np.reshape(data, test_y_hier.shape)
                """df = pd.DataFrame(re_output)
                df.to_csv(
                    f'{model_name.lower()}{folder_of_the_day}/csv/{model_name}_Prop_HGPBO_{nbr_repetition}_repetitions_init_{nbr_rand_init}_kappa_{k}_gamma_{g}_nu_{n}.csv')
                df = pd.DataFrame(y_hier)
                df.to_csv(
                    f'{model_name.lower()}{folder_of_the_day}/csv/True_State_Space_Values.csv')"""

                print(f'\n{model_name} Kappa {k} Gamma {g} Nu {n} complete!\n')

                #list_models.append([f"kappa_{k}_gamma_{g}_nu_{n}_init_{nbr_rand_init}", master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2])
                """df = pd.DataFrame([
                                      f'kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                                      master, better_exploration_score, r2, child_1_r2,
                                      child_2_r2])
                df.index = ['name', 'master', 'exploration_score', 'parent_r2', 'avg_child_r2', 'auc']

                df.to_csv(
                    f'{model_name.lower()}{folder_of_the_day}/csv/kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}')"""
                np.save(f'{model_name.lower()}{folder_of_the_day}/csv/parent_r2', r2)
                df = pd.DataFrame([
                    f'kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                    y[-1], r2_avg[-1], avg_child[-1], auc])
                df.index = ['name', 'exploration_score', 'parent_r2', 'avg_child_r2',
                            'auc']
                df.to_csv(f"{model_name.lower()}{folder_of_the_day}/csv/final_scores_kappa_{k}_gamma_{g}_nu_{n}_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}")

            # Joint Section

            """joint_plots(over_exploit, over_explor, kappa, gamma, nu_vals, folder_of_the_day, nbr_query,
                              nbr_repetition, model_name)

            joint_performance(over_exploit, over_explor, kappa, gamma, nu_vals, folder_of_the_day, nbr_query,
                              nbr_repetition, model_name)"""
    return list_models

def hp_plotting(scores, hp_name, hp_list, model_name, folder_of_the_day):
    for j in range(len(scores)):
        eval_name, evaluation = scores[j]
        for i in range(len(hp_list)):
            plt.plot(range(nbr_query), evaluation[i][:nbr_query], label=f"Init {hp_list[i]}")
        plt.title(f"{eval_name} with varying {hp_name}")
        plt.xlabel("Query Number")
        plt.ylabel(f"{eval_name}")
        plt.legend()
        plt.savefig(
            f"{model_name.lower()}{folder_of_the_day}/hp_analysis/{eval_name}_varying_{hp_name}.svg")
        plt.close()

    for eval_name, evaluation in scores:
        plt.plot(hp_list, evaluation[:, nbr_query - 1], label=eval_name)

    plt.title(f"End Model Scores for different evals at {nbr_query} Queries")
    plt.xlabel(f"{hp_name}")
    plt.ylabel(f"Performance")
    plt.legend()
    plt.savefig(f"{model_name.lower()}{folder_of_the_day}/hp_analysis/final_scores_varying_{hp_name}.svg")
    plt.close()

if __name__ == '__main__':


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
    training_iter = 15
    nbr_repetition = 10
    nbr_rand_init = 1
    k_vals = [8]  # Found through HP Testing
    g_vals = [4]  # Found through HP Testing
    nu_vals = [0.5]
    multi = False
    h_model = [hmodel.Lossless_Efficient_UCB_Hierarchical_GP]
    process = []
    seed = np.array([901112484, 798576827, 862109006, 256960071,  67686131, 960919614,
       542146925, 225453837, 328655096, 167690914, 578139702, 126081086,
       445226178, 339718381, 278636500, 570547118, 459828174, 673392709,
        56896553, 749380297, 635521450,  19699771, 351850900, 520687372,
       833438344, 355138099, 382604277,  40529313, 441069895, 797772191])
    #seed = [False]*nbr_repetition
    for h in h_model:
        if h == hmodel.Efficient_UCB_Hierarchical_GP:
            model_name = "Efficient"

        elif h == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
            model_name = "Lossless_Efficient"

        current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
        current_dateday = datetime.now().strftime("%Y-%m-%d")
        workspace = f"{model_name.lower()}"
        folder_of_the_day = '/data-' + str(current_dateday)

        training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals, h, multi, seed, children=[], visualize=True)

        """if h == hmodel.Efficient_UCB_Hierarchical_GP:
            model_name = "Efficient"

        elif h == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
            model_name = "Lossless_Efficient"

        current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
        current_dateday = datetime.now().strftime("%Y-%m-%d")
        workspace = f"{model_name.lower()}"
        folder_of_the_day = '/data-' + str(current_dateday)

        parent_r2 = []
        child_1_r2_over = []
        child_2_r2_over = []
        explor = []
        exploit = []
        names = []

        nbr_rand_init_list = np.arange(2, 22, 2)

        for nbr_rand_init in nbr_rand_init_list:
            name, master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2 = \
                training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals,
                                   h, multi)[0]
            parent_r2.append(r2[:nbr_query])
            child_1_r2_over.append(child_1_r2[:nbr_query])
            child_2_r2_over.append(child_2_r2[:nbr_query])
            explor.append(np.mean(better_exploration_score, axis=0)[:nbr_query])
            exploit.append(np.mean(better_exploitation_score, axis=0)[:nbr_query])

        scores = [["Parent_R2", np.array(parent_r2)], ["Child_1_R2", np.array(child_1_r2_over)],
                  ["Child_2_R2", np.array(child_2_r2_over)], ["Exploration", np.array(explor)],
                  ["Exploitation", np.array(exploit)]]
        hp_plotting(scores, "nbr_rand_init", nbr_rand_init_list)
        nbr_rand_init = 6

        print("============================================================")
        print("Done Rand Init")

        parent_r2 = []
        child_1_r2_over = []
        child_2_r2_over = []
        explor = []
        exploit = []
        names = []

        # Doing Training Iteration Hyper Parameter
        training_iter_list = np.arange(2, 22, 2)

        for training_iter in training_iter_list:
            name, master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2 = \
                training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals,
                                   h, multi)[0]
            parent_r2.append(r2[:nbr_query])
            child_1_r2_over.append(child_1_r2[:nbr_query])
            child_2_r2_over.append(child_2_r2[:nbr_query])
            explor.append(np.mean(better_exploration_score, axis=0)[:nbr_query])
            exploit.append(np.mean(better_exploitation_score, axis=0)[:nbr_query])

        scores = [["Parent_R2", np.array(parent_r2)], ["Child_1_R2", np.array(child_1_r2_over)],
                  ["Child_2_R2", np.array(child_2_r2_over)], ["Exploration", np.array(explor)],
                  ["Exploitation", np.array(exploit)]]
        hp_plotting(scores, "training_iter", training_iter_list)
        training_iter = 10

        print("============================================================")
        print("Done Training Iter")

        # HP search for kappa values
        parent_r2 = []
        child_1_r2_over = []
        child_2_r2_over = []
        explor = []
        exploit = []
        names = []

        k_vals_list = np.linspace(0.5, 10, 20)
        for k_vals in k_vals_list:
            k_vals = [k_vals]
            name, master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2 = \
                training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals,
                                   h, multi)[0]
            parent_r2.append(r2[:nbr_query])
            child_1_r2_over.append(child_1_r2[:nbr_query])
            child_2_r2_over.append(child_2_r2[:nbr_query])
            explor.append(np.mean(better_exploration_score, axis=0)[:nbr_query])
            exploit.append(np.mean(better_exploitation_score, axis=0)[:nbr_query])

        scores = [["Parent_R2", np.array(parent_r2)], ["Child_1_R2", np.array(child_1_r2_over)],
                  ["Child_2_R2", np.array(child_2_r2_over)], ["Exploration", np.array(explor)],
                  ["Exploitation", np.array(exploit)]]
        hp_plotting(scores, "k_vals", k_vals_list)
        k_vals = [2]
        # HP search for Gamma values
        parent_r2 = []
        child_1_r2_over = []
        child_2_r2_over = []
        explor = []
        exploit = []
        names = []

        print("============================================================")
        print("Done Kappa")

        g_vals_list = np.linspace(0.5, 10, 20)
        for g_vals in g_vals_list:
            g_vals = [g_vals]
            name, master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2 = \
                training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals,
                                   h, multi)[0]
            parent_r2.append(r2[:nbr_query])
            child_1_r2_over.append(child_1_r2[:nbr_query])
            child_2_r2_over.append(child_2_r2[:nbr_query])
            explor.append(np.mean(better_exploration_score, axis=0)[:nbr_query])
            exploit.append(np.mean(better_exploitation_score, axis=0)[:nbr_query])

        scores = [["Parent_R2", np.array(parent_r2)], ["Child_1_R2", np.array(child_1_r2_over)],
                  ["Child_2_R2", np.array(child_2_r2_over)], ["Exploration", np.array(explor)],
                  ["Exploitation", np.array(exploit)]]
        hp_plotting(scores, "g_vals", g_vals_list)
        g_vals = [6]

        print("============================================================")
        print("Done Gamma")"""