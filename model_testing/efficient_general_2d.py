import os
import gpytorch
import matplotlib.pyplot as plt
import numpy as np

import synthetic_models as models
import hmodel_synthetic as hmodel
from dataset_actions import *
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
import multiprocessing as mp
from threading import Thread
import pandas as pd
from seaborn import heatmap
import time

from model_testing.efficient_synthetic_script import joint_performance

def joint_plots(joint_exploit, joint_explor, kappa, gamma, nu_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition,
                data_name, model_name):
    for i, nu in enumerate(nu_vals):
        plt.plot(joint_exploit[i], label=f'nu {nu}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploitation Score')
    plt.title(f'Joint {model_name} Propagation HGPBO {nbr_repetition} Exploitation')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/Joint_{model_name}_Propagation_HGPBO_{nbr_repetition}_Exploitation_kappa_{kappa}_gamma_{gamma}')
    plt.close()

    for i, nu in enumerate(nu_vals):
        plt.plot(joint_explor[i], label=f'nu {nu}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploration Score')
    plt.title(f'Joint {model_name} Propagation HGPBO {nbr_repetition} Exploration')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/Joint_{model_name}_Propagation_HGPBO_{nbr_repetition}_Exploration_kappa_{kappa}_gamma_{gamma}')
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
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/HP_Exploitation_Performance_{nbr_repetition}_repetitions_kappa_{kappa}_gamma_{gamma}')
    plt.close()

    fig, ax = plt.subplots(figsize=(nbr_query/5, len(nu_vals)*3))
    heatmap(joint_explor, xticklabels=list(range(nbr_query)), yticklabels=names, cmap='coolwarm', ax=ax)

    plt.ylabel(f'Model Type')
    plt.xlabel(f'Training Step')
    plt.title(f'Joint HP Exploration Performance over {nbr_repetition} repetitions')
    #plt.tight_layout()
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/HP_Exploration_Performance_{nbr_repetition}_repetitions_kappa_{kappa}_gamma_{gamma}')
    plt.close()



def run_repetition(kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model,
                   data_creation_func, eps, model_name, folder_of_the_day, data_name, final=False, children=[], visualize=True):
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    prior_map = torch.zeros(dimension, dimension)

    ground_truth_max_1 = torch.max(y_sub1)
    ground_truth_max_2 = torch.max(y_sub2)
    ground_truth_max_hier = torch.max(y_hier)

    max_seen_resp_2D = 0
    max_seen_resp_1_1D = 0
    max_seen_resp_2_1D = 0
    sub1_qc = torch.ones(x_sub1.shape)
    sub2_qc = torch.ones(x_sub2.shape)
    hier_qc = torch.ones(len(x_sub1) * len(x_sub2))
    better_exploitation_score = []
    better_exploration_score = []
    child_1_r2 = []
    child_2_r2 = []

    heatmap_rep = []
    h_opt_time = []
    h_pred_time = []
    for q in tqdm(range(nbr_query)):

        if q == 0:
            if children == []:
                # Need to initialize the model - Will be random in this method
                train_x_sub1, train_y_sub1 = select_random_queries(nbr_rand_init, x_sub1, y_sub1)
                train_x_sub2, train_y_sub2 = select_random_queries(nbr_rand_init, x_sub2, y_sub2)
                max_seen_resp_1_1D = torch.max(train_y_sub1)
                max_seen_resp_2_1D = torch.max(train_y_sub2)
                sub1_like = gpytorch.likelihoods.GaussianLikelihood()
                sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1 / max_seen_resp_1_1D, sub1_like,
                                           query_counter=sub1_qc, nu=nu)

                sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2 / max_seen_resp_2_1D, sub2_like,
                                           query_counter=sub2_qc, nu=nu)

            elif len(children) == 1:
                sub1 = children[0]

                sub1_like = sub1.likelihood

                train_x_sub1 = sub1.train_inputs[0][:,0]
                train_y_sub1 = sub1.train_targets
                max_seen_resp_1_1D = torch.max(train_y_sub1)

                train_x_sub2, train_y_sub2 = select_random_queries(nbr_rand_init, x_sub2, y_sub2)
                max_seen_resp_2_1D = torch.max(train_y_sub2)
                sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2 / max_seen_resp_2_1D, sub2_like,
                                           query_counter=sub2_qc, nu=nu)

            elif len(children) == 2:
                sub1 = children[0]
                sub2 = children[1]

                sub1_like = sub1.likelihood
                sub2_like = sub2.likelihood

                train_x_sub1 = sub1.train_inputs[0][:,0]
                train_y_sub1 = sub1.train_targets

                train_x_sub2 = sub2.train_inputs[0][:,0]
                train_y_sub2 = sub2.train_targets

                max_seen_resp_1_1D = torch.max(train_y_sub1)
                max_seen_resp_2_1D = torch.max(train_y_sub2)


            train_x_hier, train_y_hier = hierarchical_select_random_queries(nbr_rand_init, x_hier, y_hier)
            max_seen_resp_2D = torch.max(train_y_hier)


            sub1.eval()
            sub2.eval()

            sub1_like.eval()
            sub2_like.eval()


            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
                observed_pred2 = models.make_prediction(sub2, x_sub2, sub2_like)
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

            prior_hierarchical_kernel = hmodel.hierarchical_kernel("add_kernel", sub1, sub2)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            master = hierarchical_model(train_x_hier, train_y_hier / max_seen_resp_2D, x_hier, likelihood,
                                        prior_hierarchical_kernel,
                                        prior_map / prior_map_max, kernel_op='add_kernel',
                                        sub_models=[sub1, sub2],
                                        kappa=kappa, query_counter=hier_qc)

            for i in range(nbr_rand_init):
                hier_qc = master.increment_q_n(hier_qc, train_x_hier[i], x_hier)

            master.eval()
            likelihood.eval()

            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

        acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred, hier_qc)

        next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)

        next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins,
                                                                                     test_x_hier,
                                                                                     y_hier)

        y_mu_point_a = hmodel.get_y_mu_point_value(next_query_pins[0], y_mu1, x_sub1)
        y_mu_point_b = hmodel.get_y_mu_point_value(next_query_pins[1], y_mu2, x_sub2)

        y_conf_point_a = hmodel.get_y_mu_point_value(next_query_pins[0], y_conf1, x_sub1)
        y_conf_point_b = hmodel.get_y_mu_point_value(next_query_pins[1], y_conf2, x_sub2)

        y_qc_a = hmodel.get_y_mu_point_value(next_query_pins[0], sub1_qc, x_sub1)
        y_qc_b = hmodel.get_y_mu_point_value(next_query_pins[1], sub2_qc, x_sub2)

        next_query_value_random, max_seen_resp_2D = models.update_max_seen_response_no_norm(next_query_value_random,
                                                                                            max_seen_resp_2D)

        response = torch.tensor(next_query_value_random)

        cont1 = y_mu_point_a - gamma * torch.nan_to_num(y_conf_point_a / torch.sqrt(y_qc_a))
        cont2 = y_mu_point_b - gamma * torch.nan_to_num(y_conf_point_b / torch.sqrt(y_qc_b))

        div = torch.exp(cont1) + torch.exp(cont2)

        contribution1 = torch.nan_to_num(response * torch.exp(cont1) / div)
        contribution2 = torch.nan_to_num(response * torch.exp(cont2) / div)

        response_1 = sub1.update_max_seen_response_no_norm(contribution1, max_seen_resp_1_1D)
        response_2 = sub2.update_max_seen_response_no_norm(contribution2, max_seen_resp_2_1D)

        # Potentially could make this more efficient by incorporating it into the next finder
        sub1_qc = sub1.increment_q_n(sub1_qc, next_query_pins[0], x_sub1)
        sub2_qc = sub2.increment_q_n(sub2_qc, next_query_pins[1], x_sub2)
        hier_qc = master.increment_q_n(hier_qc, next_query_pins, x_hier)

        # next_query_pins = next_query_pins.to(torch.int)
        flag = True
        for x in range(len(x_hier)):
            for y in range(len(x_hier[x])):
                if x_hier[x][y][0] == next_query_pins[0] and x_hier[x][y][1] == next_query_pins[1]:
                    next_query_indices = [x, y]
                    flag = False
                if x == dimension - 1 and y == dimension - 1 and flag:
                    raise Exception("Could not find pins in X hier for indices")

        sub1, sub1_like, train_x_sub1, train_y_sub1 = hmodel.update_model1_1D_max_seen(sub1, sub1_like, train_x_sub1,
                                                                                       train_y_sub1,
                                                                                       x_sub1[next_query_indices[0]],
                                                                                       contribution1,
                                                                                       max_seen_resp_1_1D,
                                                                                       training_iter=training_iter)

        sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model1_1D_max_seen(sub2, sub2_like, train_x_sub2,
                                                                                       train_y_sub2,
                                                                                       x_sub2[next_query_indices[1]],
                                                                                       contribution2,
                                                                                       max_seen_resp_2_1D,
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

        y_conf1 = observed_pred1.stddev
        y_conf2 = observed_pred2.stddev

        p1 = y_mu1 + gamma * y_conf1 / (torch.sqrt(sub1_qc))
        p2 = y_mu2 + gamma * y_conf2 / (torch.sqrt(sub2_qc))

        for i in range(len(prior_map)):
            for j in range(len(prior_map)):
                prior_map[i, j] = (p1[i] + p2[j]) / 2

        prior_map_max = torch.max(prior_map)

        master.mean_module.map = torch.nn.Parameter(prior_map / prior_map_max)

        master = hmodel.update_kernel_parameters(master, sub1, sub2)

        train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins,
                                                          response)
        master.set_train_data(train_x_hier, train_y_hier / max_seen_resp_2D, strict=False)

        """
        train_x_sub1, train_x_sub2 = train_x_hier[:, 0], train_x_hier[:, 1]"""

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            # Find optimal model hyperparameters
            master.train()
            likelihood.train()

            # start = time.time()

            master, likelihood = master.Hoptimize(likelihood, training_iter, train_x_hier,
                                                  train_y_hier / max_seen_resp_2D,
                                                  verbose=False)

            """t = time.time() - start
            h_opt_time.append(t)
            print(f"Hoptimize time: {t}")"""

            master.eval()
            likelihood.eval()
            sub1.eval()
            sub1_like.eval()

            sub2.eval()
            sub2_like.eval()

            # Make a prediction, observed_pred = likelihood, prediction_mean = mu
            # start = time.time()
            observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)
            """t = time.time() - start
            h_pred_time.append(t)
            print(f"Hierarchical pred time: {t}")"""

            c1_r2, c2_r2 = vi.child_contour_r2(master.sub_models, x_sub1,
                        [y_sub1 / torch.max(y_sub1), y_sub2 / torch.max(y_sub2)])

        child_1_r2.append(c1_r2)
        child_2_r2.append(c2_r2)

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

        heatmap_rep.append(observed_pred.mean.detach().cpu().numpy())



    if visualize:
        k = str(kappa).replace('.', ',')
        g = str(gamma).replace('.', ',')
        n = str(nu).replace('.', ',')
        e = str(eps).replace('.', '_')
        e = str(e).replace(' ', '_')
        e = str(e).replace('[', '')
        e = str(e).replace(']', '')
        e = str(e).replace(',_', '_')
        e = str(e).replace('_,', '_')


        vi.contour_plot_1D(master.sub_models, x_sub1,
                        [y_sub1 / torch.max(y_sub1), y_sub2 / torch.max(y_sub2)],
                        f'/contour/Contour_{data_name}_{model_name}_HGP-BO_nbr_query_{nbr_query}_init_{nbr_rand_init}_dim_{dimension}_kappa_{k}_gamma_{g}_nu_{n}_pid_{os.getpid()}_eps_{e}',
                        model_name.lower(), folder_of_the_day, data_name)



    if final:
        return master, sub1, sub2, better_exploration_score, better_exploitation_score, heatmap_rep, child_1_r2, child_2_r2
    else:
        return better_exploration_score, better_exploitation_score, heatmap_rep, child_1_r2, child_2_r2


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals, nu_vals,
                       data_name, data_creation_func, eps, hierarchical_model, multi, children=[], visualize=True):
    if hierarchical_model == hmodel.Efficient_UCB_Hierarchical_GP:
        model_name = "Efficient"

    elif hierarchical_model == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
        model_name = "Lossless_Efficient"

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name.lower()}"
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
                child_1_r2_data = []
                child_2_r2_data = []

                if multi:
                    with mp.Pool(processes=nbr_repetition - 1) as pool:

                        # running the repetitions in parallel except for the last one
                        for i in range(nbr_repetition - 1):
                            p = pool.apply_async(run_repetition, (
                            kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model,
                            data_creation_func, eps, model_name, folder_of_the_day, data_name, False, children, visualize, ))
                            processes.append(p)

                        # must run the final block manually to allow return of the models
                        try:
                            master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model,
                                data_creation_func, eps, model_name, folder_of_the_day, data_name, final=True, children=children, visualize=visualize)
                        except:
                            master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                                hierarchical_model,
                                data_creation_func, eps, model_name, folder_of_the_day, data_name, final=True,
                                children=children, visualize=visualize)

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

                    for i in range(nbr_repetition):
                        try:
                            master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model,
                                data_creation_func, eps, model_name, folder_of_the_day, data_name, final=True, children=children, visualize=visualize)
                        except:
                            try:
                                master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                    kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                                    hierarchical_model,
                                    data_creation_func, eps, model_name, folder_of_the_day, data_name, final=True,
                                    children=children, visualize=visualize)
                            except:
                                master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                    kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                                    hierarchical_model,
                                    data_creation_func, eps, model_name, folder_of_the_day, data_name, final=True,
                                    children=children, visualize=visualize)

                        better_exploration_score.append(rep_exploration_score)
                        better_exploitation_score.append(rep_exploitation_score)
                        heatmap_data.append(heatmap_rep)
                        child_1_r2_data.append(child_1_r2)
                        child_2_r2_data.append(child_2_r2)

                # Save the model
                x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

                heatmap_data = np.array(heatmap_data)

                k = str(kappa).replace('.', ',')
                g = str(gamma).replace('.', ',')
                n = str(nu).replace('.', '_')
                e = str(eps).replace('[', '')
                e = str(e).replace(']', '')
                e = str(e).replace(' ', '')
                e = str(e).replace('.', '')
                e = str(e).replace(',', '_')

                torch.save(master.state_dict(),
                           f'{data_name}/{model_name.lower()}{folder_of_the_day}/models/kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_eps_{e}_init_{nbr_rand_init}.pth')

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

                r2 = vi.heatmap_r_score(heatmap_data, y_hier)
                plt.plot(r2, label="Parent R2")

                child_1_r2 = np.mean(child_1_r2_data, axis=0)
                child_2_r2 = np.mean(child_2_r2_data, axis=0)

                plt.plot(child_1_r2, label="Child 1 R2")
                plt.plot(child_2_r2, label="Child 2 R2")


                """rand = np.random.rand(*heatmap_data.shape)
                random_r2 = vi.heatmap_r_score(rand, y_hier)
                plt.plot(random_r2, label="Random Heatmap R2")"""

                plt.legend()
                plt.ylim(-0.1, 1.1)
                plt.ylabel('Performance')
                plt.xlabel('Query Number')

                plt.title(f'{model_name} HGP-BO {nbr_repetition} repetitions with kappa {k} Gamma {g} Nu {n} Init {nbr_rand_init}')
                plt.savefig(
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/{model_name}_Prop_{data_name}_HGPBO_{nbr_repetition}_repetitions_init_{nbr_rand_init}_eps_{e}_kappa_{k}_gamma_{g}_nu_{n}')
                plt.close()

                vi.model_heatmap(heatmap_data[:, -1, :], x_hier, y_hier,
                                 f'Heatmap_{data_name}_{model_name}_HGPBO_{nbr_repetition}_repetitions_eps_{e}_kappa_{k}_gamma_{g}_nu_{n}',
                                 model_name.lower(), folder_of_the_day, data_name)

                data = np.mean(heatmap_data[:, -1, :], axis=0)

                re_output = np.reshape(data, y_hier.shape)
                df = pd.DataFrame(re_output)
                df.to_csv(
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/{model_name}_Prop_{data_name}_HGPBO_{nbr_repetition}_repetitions_init_{nbr_rand_init}_eps_{e}_kappa_{k}_gamma_{g}_nu_{n}.csv')
                df = pd.DataFrame(y_hier)
                df.to_csv(
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/True_State_Space_Values.csv')

                print(f'\n{model_name} Kappa {k} Gamma {g} Nu {n} eps_{e}_ complete!\n')

                list_models.append([f"eps_{e}_kappa_{k}_gamma_{g}_nu_{n}_init_{nbr_rand_init}", master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2])

            # Joint Section
            joint_performance(over_exploit, over_explor, kappa, gamma, nu_vals, folder_of_the_day, dimension, nbr_query,
                              nbr_repetition, data_name, model_name)

            joint_plots(over_exploit, over_explor, kappa, gamma, nu_vals, folder_of_the_day, dimension, nbr_query,
                        nbr_repetition, data_name, model_name)

    return list_models
if __name__ == '__main__':

    dimension = 15
    nbr_query = 80
    training_iter = 5
    nbr_repetition = 15
    k_vals = [2]
    g_vals = [6]
    nu_vals = [0.5]
    multi = True
    h_model = [hmodel.Lossless_Efficient_UCB_Hierarchical_GP]
    process = []

    for h in h_model:
        for dataset_num in [2, 3]:

            data_name, data_creation_func, eps = get_dataset_info(dataset_num)

            if h == hmodel.Efficient_UCB_Hierarchical_GP:
                model_name = "Efficient"

            elif h == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
                model_name = "Lossless_Efficient"

            current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
            current_dateday = datetime.now().strftime("%Y-%m-%d")
            workspace = f"{data_name}/{model_name.lower()}"
            folder_of_the_day = '/data-' + str(current_dateday)

            parent_r2 = []
            child_1_r2_over = []
            child_2_r2_over = []
            explor = []
            exploit = []
            names = []
            nbr_rand_init = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

            for rand_init in nbr_rand_init:

                """if multi:
                    p = mp.Process(target=training_procedure, args=(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals, nu_vals, data_name, data_creation_func,
                                   eps, h, multi,))
                    process.append(p)
                    p.start()
                    print(f"ID of process: {p.pid}")
                else:"""
                name, master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2 = training_procedure(nbr_query, nbr_repetition, rand_init, dimension, training_iter, k_vals, g_vals, nu_vals, data_name, data_creation_func,
                                   eps, h, multi)[0]
                parent_r2.append(r2)
                child_1_r2_over.append(child_1_r2)
                child_2_r2_over.append(child_2_r2)
                explor.append(np.mean(better_exploration_score, axis=0))
                exploit.append(np.mean(better_exploitation_score, axis=0))

            scores = [["Parent_R2", np.array(parent_r2)],["Child_1_R2", np.array(child_1_r2_over)],["Child_2_R2", np.array(child_2_r2_over)], ["Exploration", np.array(explor)], ["Exploitation", np.array(exploit)]]
            for j in range(len(scores)):
                eval_name, evaluation = scores[j]
                for i in range(len(nbr_rand_init)):
                    plt.plot(range(nbr_query), evaluation[i], label=f"Init {nbr_rand_init[i]}")
                plt.title(f"{eval_name} with varying random init")
                plt.xlabel("Query Number")
                plt.ylabel(f"{eval_name}")
                plt.legend()
                plt.savefig(f"{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/{eval_name}_varying_random_init.png")
                plt.close()

            for eval_name, evaluation in scores:
                plt.plot(nbr_rand_init, evaluation[:,-1], label=eval_name)

            plt.title(f"End Model Scores for different evals")
            plt.xlabel("Query Number")
            plt.ylabel(f"Performance")
            plt.legend()
            plt.savefig(f"{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/final_scores_varying_random_init")
            plt.close()