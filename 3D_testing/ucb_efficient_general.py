import gpytorch
import numpy as np

import models_3d as models
import hmodel_3d as hmodel
from dataset_actions_3d import *
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
import multiprocessing as mp
from seaborn import heatmap
import time
import pandas as pd
import cProfile
import pstats

import warnings

def joint_plots(joint_exploit, joint_explor, kappa, gamma, nu_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition,
                data_name, model_name):
    for i, nu in enumerate(nu_vals):
        plt.plot(joint_exploit[i], label=f'nu {nu}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploitation Score')
    plt.title(f'Joint Efficient Propagation HGPBO {nbr_repetition} Exploitation')
    plt.savefig(
        f'{data_name}/{model_name}{folder_of_the_day}/hp_analysis/Joint_{model_name}_Propagation_HGPBO_{nbr_repetition}_Exploitation_kappa_{kappa}_gamma_{gamma}.svg')
    plt.close()

    for i, nu in enumerate(nu_vals):
        plt.plot(joint_explor[i], label=f'nu {nu}')


    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploration Score')
    plt.title(f'Joint Efficient Propagation HGPBO {nbr_repetition} Exploration')
    plt.savefig(
        f'{data_name}/{model_name}{folder_of_the_day}/hp_analysis/Joint_{model_name}_Propagation_HGPBO_{nbr_repetition}_Exploration_kappa_{kappa}_gamma_{gamma}.svg')
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
    plt.title(f'Joint HP Exploitation 3D Performance over {nbr_repetition} repetitions')
    plt.tight_layout()
    plt.savefig(
        f'{data_name}/{model_name}{folder_of_the_day}/hp_analysis/HP_3D_Exploitation_Performance_{nbr_repetition}_repetitions_kappa_{kappa}_gamma_{gamma}.svg')
    plt.close()

    fig, ax = plt.subplots(figsize=(nbr_query/5, len(nu_vals)*3))
    heatmap(joint_explor, xticklabels=list(range(nbr_query)), yticklabels=names, cmap='coolwarm', ax=ax)

    plt.ylabel(f'Model Type')
    plt.xlabel(f'Training Step')
    plt.title(f'Joint HP Exploration 3D Performance over {nbr_repetition} repetitions')
    plt.tight_layout()
    plt.savefig(
        f'{data_name}/{model_name}{folder_of_the_day}/hp_analysis/HP_3D_Exploration_Performance_{nbr_repetition}_repetitions_kappa_{kappa}_gamma_{gamma}.svg')
    plt.close()


def run_repetition(kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model, data_creation_func, eps, model_name, folder_of_the_day, data_name, final=False, children=[], seed=True, noise=0.1):
    warnings.filterwarnings('ignore')

    x_sub1, y_sub1, x_sub2, y_sub2, x_sub3, y_sub3, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)
    prior_map = torch.zeros(dimension, dimension, dimension)

    ground_truth_max_hier = torch.max(y_hier)

    max_seen_resp_2D = 0
    max_seen_resp_1_1D = 0
    max_seen_resp_2_1D = 0
    max_seen_resp_3_1D = 0
    sub1_qc = torch.ones(x_sub1.shape)
    sub2_qc = torch.ones(x_sub2.shape)
    sub3_qc = torch.ones(x_sub3.shape)
    hier_qc = torch.ones(len(x_sub1) * len(x_sub2) * len(x_sub3))

    better_exploration_score = []
    better_exploitation_score = []
    heatmap_rep = []
    h_opt_time = []
    h_pred_time = []
    children_r2 = []

    k = str(kappa).replace('.', ',')
    g = str(gamma).replace('.', ',')
    n = str(nu).replace('.', ',')

    for q in tqdm(range(nbr_query)):
        if q == 0:
            # Need to initialize the model - Will be random in this method
            train_x_sub1, train_y_sub1 = select_random_queries(nbr_rand_init, x_sub1, y_sub1, seed=seed, noise=noise)
            train_x_sub2, train_y_sub2 = select_random_queries(nbr_rand_init, x_sub2, y_sub2, seed=seed, noise=noise)
            train_x_sub3, train_y_sub3 = select_random_queries(nbr_rand_init, x_sub3, y_sub3, seed=seed, noise=noise)
            train_x_hier, train_y_hier = hierarchical_select_random_queries(1, x_hier, y_hier, seed=seed)
            max_seen_resp_1_1D = torch.max(train_y_sub1)
            max_seen_resp_2_1D = torch.max(train_y_sub2)
            max_seen_resp_3_1D = torch.max(train_y_sub3)

            max_seen_resp_2D = torch.max(train_y_hier)

            """
            train_y_sub1 = train_y_sub1 / max_seen_resp_1_1D
            train_y_sub2 = train_y_sub2 / max_seen_resp_2_1D
            train_y_hier = train_y_hier / max_seen_resp_2D"""

            # Need to modify this section such that the model is receiving the partial contribution
            """train_x_sub1, train_y_sub1 = update_training_data(train_x_sub1, train_y_sub1,
                                                                train_x_hier[:, 0], train_y_hier)
            train_x_sub2, train_y_sub2 = update_training_data(train_x_sub2, train_y_sub2,
                                                                train_x_hier[:, 1], train_y_hier)
            train_x_sub3, train_y_sub3 = update_training_data(train_x_sub3, train_y_sub3,
                                                                train_x_hier[:, 2], train_y_hier)"""

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

            sub1.eval()
            sub2.eval()
            sub3.eval()

            sub1_like.eval()
            sub2_like.eval()
            sub3_like.eval()

            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
                observed_pred2 = models.make_prediction(sub2, x_sub2, sub2_like)
                observed_pred3 = models.make_prediction(sub3, x_sub3, sub3_like)
            y_mu1 = observed_pred1.mean
            y_mu2 = observed_pred2.mean
            y_mu3 = observed_pred3.mean

            y_conf1 = observed_pred1.stddev
            y_conf2 = observed_pred2.stddev
            y_conf3 = observed_pred3.stddev

            p1 = y_mu1 + gamma * y_conf1 / (torch.sqrt(sub1_qc))
            p2 = y_mu2 + gamma * y_conf2 / (torch.sqrt(sub2_qc))
            p3 = y_mu3 + gamma * y_conf3 / (torch.sqrt(sub3_qc))


            for i in range(len(prior_map)):
                for j in range(len(prior_map)):
                    for k in range(len(prior_map)):
                        prior_map[i, j, k] = (p1[i] + p2[j] + p3[k]) / 3

            prior_map_max = torch.max(prior_map)

            next_query_pins = models.get_next_query_pins(torch.flatten(prior_map), test_x_hier)

            next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins,
                                                                                         test_x_hier,
                                                                                         y_hier, noise=noise)
            response = torch.tensor(next_query_value_random)
            train_x_hier, train_y_hier = torch.reshape(next_query_pins, (1, 3)), torch.reshape(response, (1,))

            # train_x_hier, train_y_hier = hierarchical_select_random_queries(1, x_hier, y_hier, seed=seed, noise=noise)

            max_seen_resp_2D = torch.max(train_y_hier)

            prior_hierarchical_kernel = hmodel.hierarchical_kernel("add_kernel", [sub1, sub2, sub3])
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            master = hierarchical_model(train_x_hier, train_y_hier - torch.mean(train_y_hier), x_hier, likelihood,
                                        prior_hierarchical_kernel,
                                        prior_map / prior_map_max, kernel_op='add_kernel',
                                        sub_models=[sub1, sub2, sub3],
                                        kappa=kappa, query_counter=hier_qc)
            """master = hierarchical_model(train_x_hier, train_y_hier / max_seen_resp_2D, x_hier, likelihood,
                                        prior_hierarchical_kernel,
                                        prior_map / prior_map_max, kernel_op='add_kernel',
                                        sub_models=[sub1, sub2, sub3],
                                        kappa=kappa, query_counter=hier_qc)"""

            for i in range(len(train_x_hier)):
                hier_qc = master.increment_q_n(hier_qc, train_x_hier[i], x_hier)

            master.eval()
            likelihood.eval()

            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)
        with gpytorch.settings.lazily_evaluate_kernels(state=False):

            child_r2 = vi.child_contour_r2(master.sub_models, [x_sub1, x_sub2, x_sub3],[y_sub1, y_sub2, y_sub3])

        children_r2.append(child_r2)
       
        acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred, hier_qc)

        next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)

        next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins,
                                                                                        test_x_hier,
                                                                                        y_hier, noise=noise)

        y_mu_point_a = hmodel.get_y_mu_point_value(next_query_pins[0], y_mu1, x_sub1)
        y_mu_point_b = hmodel.get_y_mu_point_value(next_query_pins[1], y_mu2, x_sub2)
        y_mu_point_c = hmodel.get_y_mu_point_value(next_query_pins[2], y_mu3, x_sub3)

        y_conf_point_a = hmodel.get_y_mu_point_value(next_query_pins[0], y_conf1, x_sub1)
        y_conf_point_b = hmodel.get_y_mu_point_value(next_query_pins[1], y_conf2, x_sub2)
        y_conf_point_c = hmodel.get_y_mu_point_value(next_query_pins[2], y_conf3, x_sub3)

        y_qc_a = hmodel.get_y_mu_point_value(next_query_pins[0], sub1_qc, x_sub1)
        y_qc_b = hmodel.get_y_mu_point_value(next_query_pins[1], sub2_qc, x_sub2)
        y_qc_c = hmodel.get_y_mu_point_value(next_query_pins[2], sub3_qc, x_sub3)

        next_query_value_random, max_seen_resp_2D = models.update_max_seen_response_no_norm(next_query_value_random,   
                                                                                    max_seen_resp_2D)

        response = torch.tensor(next_query_value_random)

        cont1 = y_mu_point_a + gamma * torch.nan_to_num(y_conf_point_a / torch.sqrt(y_qc_a))

        cont1_scaled = torch.nan_to_num(
            cont1 / torch.max(y_mu1 + gamma * torch.nan_to_num(y_conf1 / torch.sqrt(sub1_qc))))

        cont2 = y_mu_point_b + gamma * torch.nan_to_num(y_conf_point_b / torch.sqrt(y_qc_b))
        cont2_scaled = torch.nan_to_num(
            cont2 / torch.max(y_mu2 + gamma * torch.nan_to_num(y_conf2 / torch.sqrt(sub2_qc))))

        cont3 = y_mu_point_c + gamma * torch.nan_to_num(y_conf_point_c / torch.sqrt(y_qc_c))
        cont3_scaled = torch.nan_to_num(
            cont3 / torch.max(y_mu3 + gamma * torch.nan_to_num(y_conf3 / torch.sqrt(sub3_qc))))

        div = torch.exp(cont1_scaled) + torch.exp(cont2_scaled) + torch.exp(cont3_scaled)

        contribution1 = torch.nan_to_num(response * torch.exp(cont1_scaled) / div)
        contribution2 = torch.nan_to_num(response * torch.exp(cont2_scaled) / div)
        contribution3 = torch.nan_to_num(response * torch.exp(cont3_scaled) / div)


        response_1 = sub1.update_max_seen_response_no_norm(contribution1, max_seen_resp_1_1D)
        response_2 = sub2.update_max_seen_response_no_norm(contribution2, max_seen_resp_2_1D)
        response_3 = sub3.update_max_seen_response_no_norm(contribution3, max_seen_resp_3_1D)

        """train_x_sub1, train_y_sub1 = update_training_data(train_x_sub1, train_y_sub1, next_query_pins[0],
                                                            response_1)
        train_x_sub2, train_y_sub2 = update_training_data(train_x_sub2, train_y_sub2, next_query_pins[1],
                                                            response_2)

        sub1.set_train_data(train_x_sub1, train_y_sub1 / max_seen_resp_1_1D, strict=False)
        sub2.set_train_data(train_x_sub2, train_y_sub2 / max_seen_resp_2_1D, strict=False)"""

        sub1_qc = sub1.increment_q_n(sub1_qc, next_query_pins[0], x_sub1)
        sub2_qc = sub2.increment_q_n(sub2_qc, next_query_pins[1], x_sub2)
        sub3_qc = sub3.increment_q_n(sub3_qc, next_query_pins[2], x_sub3)
        hier_qc = master.increment_q_n(hier_qc, next_query_pins, x_hier)

        # next_query_pins = next_query_pins.to(torch.int)

        flag = True
        x = 0
        while x < len(x_hier) and flag:
            y = 0
            while y < len(x_hier) and flag:
                z = 0
                while z < len(x_hier) and flag:
                    if x_hier[x][y][z][0] == next_query_pins[0] and x_hier[x][y][z][1] == next_query_pins[1] and x_hier[x][y][z][2] == next_query_pins[2]:
                        next_query_indices = [x, y, z]
                        flag = False
                    if x == dimension - 1 and y == dimension - 1 and z == dimension - 1 and flag:
                        raise Exception("Could not find pins in X hier for indices")
                    z += 1
                y += 1
            x += 1


        sub1, sub1_like, train_x_sub1, train_y_sub1 = hmodel.update_model1_1D_max_seen(sub1, sub1_like, train_x_sub1,
                                                                                       train_y_sub1,
                                                                                       x_sub1[next_query_indices[0]],
                                                                                       contribution1,
                                                                                       False,
                                                                                       training_iter=training_iter)

        sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model1_1D_max_seen(sub2, sub2_like, train_x_sub2,
                                                                                       train_y_sub2,
                                                                                       x_sub2[next_query_indices[1]],
                                                                                       contribution2,
                                                                                       False,
                                                                                       training_iter=training_iter)

        sub3, sub3_like, train_x_sub3, train_y_sub3 = hmodel.update_model1_1D_max_seen(sub3, sub3_like, train_x_sub3,
                                                                                       train_y_sub3,
                                                                                       x_sub3[next_query_indices[2]],
                                                                                       contribution3,
                                                                                       False,
                                                                                       training_iter=training_iter)

        sub1.eval()
        sub1_like.eval()

        sub2.eval()
        sub2_like.eval()

        sub3.eval()
        sub3_like.eval()

        # Make a prediction, observed_pred = likelihood
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
            observed_pred2 = models.make_prediction(sub2, x_sub2, sub2_like)
            observed_pred3 = models.make_prediction(sub3, x_sub3, sub3_like)

        y_mu1 = observed_pred1.mean
        y_mu2 = observed_pred2.mean
        y_mu3 = observed_pred3.mean

        y_conf1 = observed_pred1.stddev
        y_conf2 = observed_pred2.stddev
        y_conf3 = observed_pred3.stddev

        p1 = y_mu1 + gamma * y_conf1 / (torch.sqrt(sub1_qc))
        p2 = y_mu2 + gamma * y_conf2 / (torch.sqrt(sub2_qc))
        p3 = y_mu3 + gamma * y_conf3 / (torch.sqrt(sub3_qc))

        for i in range(len(prior_map)):
            for j in range(len(prior_map)):
                for k in range(len(prior_map)):
                    prior_map[i, j, k] = (p1[i] + p2[j] + p3[k]) / 3

        prior_map_max = torch.max(prior_map)

        master.mean_module.map = torch.nn.Parameter(prior_map / prior_map_max)

        master = hmodel.update_kernel_parameters(master, [sub1, sub2, sub3])

        train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins,
                                                            response)
        master.set_train_data(train_x_hier, (train_y_hier - torch.mean(train_y_hier))/torch.std(train_y_hier), strict=False)
        #master.set_train_data(train_x_hier, train_y_hier / max_seen_resp_2D, strict=False)
        """
        train_x_sub1, train_x_sub2 = train_x_hier[:, 0], train_x_hier[:, 1]"""


        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            # Find optimal model hyperparameters
            master.train()
            likelihood.train()

            master, likelihood = master.Hoptimize(likelihood, training_iter, train_x_hier,
                                                  (train_y_hier - torch.mean(train_y_hier))/torch.std(train_y_hier),
                                                    verbose=False)

            """master, likelihood = master.Hoptimize(likelihood, training_iter, train_x_hier,
                                                  train_y_hier / max_seen_resp_2D,
                                                  verbose=False)"""

            # Get into evaluation (predictive posterior) mode
            master.eval()
            likelihood.eval()
            sub1.eval()
            sub1_like.eval()

            sub2.eval()
            sub2_like.eval()

            sub3.eval()
            sub3_like.eval()

            # Make a prediction, observed_pred = likelihood, prediction_mean = mu
            observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

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
        heatmap_rep.append(observed_pred.mean.detach().cpu().numpy())
    noi = str(noise).replace('.', ',')
    vi.contour_plot_1D(master.sub_models, [x_sub1, x_sub2, x_sub3], [y_sub1, y_sub2, y_sub3], [train_y_sub1, train_y_sub2, train_y_sub3],
                       f'/contour/Contour_init_{nbr_rand_init}_train_iter_{training_iter}_k_{k}_g_{g}_nu_{n}_noise_{noi}',
                       model_name.lower(), folder_of_the_day, data_name, parent=master, query=q)

    
    if final:
        return master, sub1, sub2, sub3, better_exploration_score, better_exploitation_score, heatmap_rep, children_r2
    else:
        return better_exploration_score, better_exploitation_score, heatmap_rep, children_r2


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals, nu_vals, data_name, data_creation_func,
                           eps, hierarchical_model, multi, seed, children=[], visualize=True, noise=0.1):
    
    if hierarchical_model == hmodel.Efficient_UCB_Hierarchical_GP:
        model_name = "Efficient_3D"

    elif hierarchical_model == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
        model_name = "Lossless_Efficient"

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name}"
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
        print("PNG folder created")
        os.mkdir(workspace + folder_of_the_day + '/png')

    x_sub1, y_sub1, x_sub2, y_sub2, x_sub3, y_sub3, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)


    list_prior_map = []
    list_objective_mean_map = []
    final_exploitation_metric = []
    final_exploration_metric = []

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
                c1_r2_data = []
                c2_r2_data = []
                c3_r2_data = []

                if multi:
                    with mp.Pool(processes=nbr_repetition - 1) as pool:
                        for i in range(nbr_repetition - 1):
                            p = pool.apply_async(run_repetition, (
                                kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                                hierarchical_model,
                                data_creation_func, eps, model_name, folder_of_the_day, data_name, False, children,
                                seed[i], noise))
                            processes.append(p)

                        master, sub1, sub2, sub3, rep_exploration_score, rep_exploitation_score, heatmap_rep, children_r2 = run_repetition(
                            kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model,
                            data_creation_func, eps, model_name, folder_of_the_day, data_name, True, children, seed[-1],
                            noise)

                        better_exploration_score.append(rep_exploration_score)
                        better_exploitation_score.append(rep_exploitation_score)
                        heatmap_data.append(heatmap_rep)
                        children_r2 = np.array(children_r2)
                        c1_r2_data.append(children_r2[:, 0])
                        c2_r2_data.append(children_r2[:, 1])
                        c3_r2_data.append(children_r2[:, 2])

                        for proc in processes:
                            try:
                                rep_exploration_score, rep_exploitation_score, heatmap_rep, children_r2 = proc.get()

                                better_exploration_score.append(rep_exploration_score)
                                better_exploitation_score.append(rep_exploitation_score)
                                heatmap_data.append(heatmap_rep)
                                children_r2 = np.array(children_r2)
                                c1_r2_data.append(children_r2[:, 0])
                                c2_r2_data.append(children_r2[:, 1])
                                c3_r2_data.append(children_r2[:, 2])
                            except:
                                continue

                else:
                    for i in range(nbr_repetition):
                        master, sub1, sub2, sub3, rep_exploration_score, rep_exploitation_score, heatmap_rep, children_r2 = run_repetition(
                            kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model,
                            data_creation_func, eps, model_name, folder_of_the_day, data_name, True, children, seed[i],
                            noise)

                        better_exploration_score.append(rep_exploration_score)
                        better_exploitation_score.append(rep_exploitation_score)
                        heatmap_data.append(heatmap_rep)
                        children_r2 = np.array(children_r2)
                        c1_r2_data.append(children_r2[:, 0])
                        c2_r2_data.append(children_r2[:, 1])
                        c3_r2_data.append(children_r2[:, 2])
                        print(f"Complete trial {i}/{nbr_repetition}")
                x_sub1, y_sub1, x_sub2, y_sub2, x_sub3, y_sub3, x_hier, y_hier, test_x, test_x_hier = data_creation_func(
                    dimension, eps)
                heatmap_data = np.array(heatmap_data)
                k = str(kappa).replace('.', ',')
                g = str(gamma).replace('.', ',')
                n = str(nu).replace('.', '_')
                e = str(eps).replace('[', '')
                e = str(e).replace(']', '')
                e = str(e).replace(' ', '')
                e = str(e).replace('.', '')
                e = str(e).replace(',', '_')
                noi = str(noise).replace('.', ',')

                torch.save(master.state_dict(),
                           f'{data_name}/{model_name.lower()}{folder_of_the_day}/models/{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}_eps_{e}_k_{k}_g_{g}_nu_{n}_noise_{noi}.pth')
                y = np.mean(better_exploration_score, axis=0)
                y = np.insert(y, 0, np.zeros(3*nbr_rand_init))[:nbr_query]
                over_explor.append(y)
                std = np.std(better_exploration_score, axis=0)/ np.sqrt(len(better_exploration_score))
                std = np.insert(std, 0, np.zeros(3*nbr_rand_init))[:nbr_query]
                print(f"explor {y[-1]}")
                plt.plot(y, label='Exploration')
                plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)

                """y = np.mean(better_exploitation_score, axis=0)
                y = np.insert(y, 0, np.zeros(3*nbr_rand_init))[:nbr_query]
                over_exploit.append(y)

                std = np.std(better_exploitation_score, axis=0)
                std = np.insert(std, 0, np.zeros(3*nbr_rand_init))[:nbr_query]"""

                #plt.plot(y, label='Exploitation')
                #plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)

                r2 = vi.heatmap_r_score(heatmap_data, y_hier)
                r2 = np.insert(r2, 0, np.zeros((3*nbr_rand_init, 1)), axis=1)[:, :nbr_query]
                r2_avg = np.mean(r2, axis=0)
                r2_std = np.std(r2, axis=0) / np.sqrt(len(r2))
                # r2_std = np.insert(r2_std, 0, np.zeros((2 - len(children)) *nbr_rand_init))[:nbr_query]
                print(f"r2 {r2_avg[-1]}")
                plt.plot(r2_avg, label="Parent R2")
                plt.fill_between(range(len(r2_std)), r2_avg - r2_std, r2_avg + r2_std, alpha=0.4)


                """
                plt.plot(child_1_r2, label="Child 1 R2")
                plt.plot(child_2_r2, label="Child 2 R2")
                plt.plot(child_3_r2, label="Child 3 R2")"""

                all_children = np.concatenate([c1_r2_data, c2_r2_data, c3_r2_data])
                all_children = np.insert(all_children, 0, np.zeros((3*nbr_rand_init, 1)), axis=1)[:, :nbr_query]

                avg_child = np.mean(all_children, axis=0)
                std_child = np.std(all_children, axis=0) / np.sqrt(len(all_children))

                plt.plot(avg_child, label='Child Avg R2')
                plt.fill_between(range(len(avg_child)), avg_child - std_child, avg_child + std_child, alpha=0.4)
                plt.legend()
                plt.ylim(-0.1, 1.1)

                plt.title(
                    f'{model_name} HGP-BO 3D {nbr_repetition} repetitions with kappa {k} Gamma {g} Nu {n} Init {nbr_rand_init}')
                plt.savefig(
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}_eps_{e}_k_{k}_g_{g}_nu_{n}_noise_{noi}.svg')
                plt.savefig(
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/png/{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}_eps_{e}_k_{k}_g_{g}_nu_{n}_noise_{noi}.png')

                plt.close()

                print(f"child r2 {avg_child[-1]}")
                auc = np.sum(avg_child + y + r2_avg)
                print(f"AUC {auc}")
                df = pd.DataFrame([f'kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                                      master, better_exploration_score, better_exploitation_score, r2, c1_r2_data,
                                      c2_r2_data, c3_r2_data])
                df.index = ['name', 'master', 'exploration_score', 'exploitation_score', 'parent_r2', 'child1_r2',
                            'child2_r2', 'child3_r2']

                df.to_csv(
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}')

                np.save(f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/parent_r2', r2)
                list_models.append([f'kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                                       master, better_exploration_score, better_exploitation_score, r2, c1_r2_data,
                                       c2_r2_data, c3_r2_data])

            # Joint Section
            '''joint_performance(over_exploit, over_explor, kappa, gamma, nu_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition, data_name, model_name)

            joint_plots(over_exploit, over_explor, kappa, gamma, nu_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition, data_name, model_name)'''
    return list_models

def hp_plotting(scores, hp_name, hp_list):
    for j in range(len(scores)):
        eval_name, evaluation = scores[j]
        for i in range(len(hp_list)):
            plt.plot(range(nbr_query), evaluation[i][:nbr_query], label=f"Init {hp_list[i]}")
        plt.title(f"{eval_name} with varying {hp_name}")
        plt.xlabel("Query Number")
        plt.ylabel(f"{eval_name}")
        plt.legend()
        plt.savefig(
            f"{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/{eval_name}_varying_{hp_name}.svg")
        plt.close()

    for eval_name, evaluation in scores:
        plt.plot(hp_list, evaluation[:, nbr_query - 1], label=eval_name)

    plt.title(f"End Model Scores for different evals at {nbr_query} Queries")
    plt.xlabel(f"{hp_name}")
    plt.ylabel(f"Performance")
    plt.legend()
    plt.savefig(f"{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/final_scores_varying_{hp_name}.svg")
    plt.close()


if __name__ == '__main__':


    dimension = 10
    nbr_query = 100
    training_iter = 10
    nbr_repetition = 30
    nbr_rand_init = 6
    k_vals = [9.5] # Found through HP Testing
    g_vals = [3]  # Found through HP Testing
    nu_vals = [0.5]

    h_model = [hmodel.Lossless_Efficient_UCB_Hierarchical_GP]
    process = []

    multi = False
    seed = np.array([901112484, 798576827, 862109006, 256960071,  67686131, 960919614,
       542146925, 225453837, 328655096, 167690914, 578139702, 126081086,
       445226178, 339718381, 278636500, 570547118, 459828174, 673392709,
        56896553, 749380297, 635521450,  19699771, 351850900, 520687372,
       833438344, 355138099, 382604277,  40529313, 441069895, 797772191])
    #seed = [False] * nbr_repetition

    for h in h_model:
        for dataset_num in [6, 5, 4]:
            data_name, data_creation_func, eps = get_dataset_info(dataset_num)


            name, master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2, child_3_r2 = \
                training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                   g_vals, nu_vals, data_name, data_creation_func, eps, h, multi, seed, noise=0.1)[0]
