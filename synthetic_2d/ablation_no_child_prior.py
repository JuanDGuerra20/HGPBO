import gpytorch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import synthetic_models as models
import hmodel_synthetic as hmodel
from dataset_actions import *
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from seaborn import heatmap
import warnings


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
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/Joint_{nbr_repetition}_Exploitation_kappa_{kappa}_gamma_{gamma}.svg')
    plt.close()

    for i, nu in enumerate(nu_vals):
        plt.plot(joint_explor[i], label=f'nu {nu}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploration Score')
    plt.title(f'Joint {model_name} Propagation HGPBO {nbr_repetition} Exploration')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/Joint_{nbr_repetition}_Exploration_kappa_{kappa}_gamma_{gamma}.svg')
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
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/HP_Exploitation_{nbr_repetition}_repetitions_k_{kappa}_g_{gamma}.svg')
    plt.close()

    fig, ax = plt.subplots(figsize=(nbr_query/5, len(nu_vals)*3))
    heatmap(joint_explor, xticklabels=list(range(nbr_query)), yticklabels=names, cmap='coolwarm', ax=ax)

    plt.ylabel(f'Model Type')
    plt.xlabel(f'Training Step')
    plt.title(f'Joint HP Exploration Performance over {nbr_repetition} repetitions')
    #plt.tight_layout()
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/HP_Exploration_{nbr_repetition}_reps_k_{kappa}_g_{gamma}.svg')
    plt.close()



def run_repetition(kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model,
                   data_creation_func, eps, model_name, folder_of_the_day, data_name, final=False, children=[], visualize=True, seed=True, noise=0.1, alpha=0, disable_tqdm=False, acq_func='ucb', num_dim=2):
    warnings.filterwarnings('ignore')
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    ground_truth_max_1 = torch.max(y_sub1)
    ground_truth_max_2 = torch.max(y_sub2)
    ground_truth_max_hier = torch.max(y_hier)

    """y_sub1 = y_sub1 + np.random.normal(0, noise*(ground_truth_max_1-torch.min(y_sub1)))
    y_sub2 = y_sub2 + np.random.normal(0, noise*(ground_truth_max_2-torch.min(y_sub2)))
    y_hier = y_hier + np.random.normal(0, noise*(ground_truth_max_hier-torch.min(y_hier)))"""

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
    for q in tqdm(range(nbr_query), disable=disable_tqdm):

        if q == 0:
            if children == []:

                # Need to initialize the model - Will be random in this method
                train_x_sub1, train_y_sub1 = select_random_queries(nbr_rand_init, x_sub1, y_sub1, seed=seed,
                                                                   noise=noise)
                train_x_sub2, train_y_sub2 = select_random_queries(nbr_rand_init, x_sub2, y_sub2, seed=seed,
                                                                   noise=noise)
                max_seen_resp_1_1D = torch.max(train_y_sub1)
                max_seen_resp_2_1D = torch.max(train_y_sub2)
                sub1_like = gpytorch.likelihoods.GaussianLikelihood()
                sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1, sub1_like,
                                           query_counter=sub1_qc, nu=nu)

                sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like,
                                           query_counter=sub2_qc, nu=nu)

                for i in range(len(train_x_sub1)):
                    sub1_qc = sub1.increment_q_n(sub1_qc, train_x_sub1[i], x_sub1)
                    sub2_qc = sub2.increment_q_n(sub2_qc, train_x_sub2[i], x_sub2)

                """sub1, sub1_like, train_x_sub1, train_y_sub1, sub1_qc = hmodel.train_submodels(sub1, sub1_like,
                                                                                              train_x_sub1,
                                                                                              train_y_sub1, x_sub1,
                                                                                              y_sub1, sub1_qc,
                                                                                              nbr_rand_init - 1,
                                                                                              2, noise, kappa, nu)
                sub2, sub2_like, train_x_sub2, train_y_sub2, sub2_qc = hmodel.train_submodels(sub2, sub2_like,
                                                                                              train_x_sub2,
                                                                                              train_y_sub2, x_sub2,
                                                                                              y_sub2, sub2_qc,
                                                                                              nbr_rand_init - 1,
                                                                                              2, noise,
                                                                                              kappa, nu)"""



            elif len(children) == 1:
                sub1 = children[0]

                # Collecting the acquisition function for the child
                sub1_like = sub1.likelihood

                sub1.eval()

                sub1_like.eval()

                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
                y_mu1 = observed_pred1.mean

                y_conf1 = observed_pred1.stddev

                p1 = y_mu1 + gamma * y_conf1 / (torch.sqrt(sub1_qc))

                sub1.train()
                sub1_like.train()

                p1_max = torch.max(p1)

                x_ind_p1 = torch.argwhere(p1 == p1_max)[:, 0]

                """sub1.train_inputs = (sub1.train_inputs[0][-1:],)
                sub1.train_inputs[0][0][0] = x_sub1[x_ind_p1][0]"""
                sub1.train_inputs = (torch.reshape(x_sub1[x_ind_p1], (1, 1)),)
                train_x_sub1 = sub1.train_inputs[0][:,0]

                sub1.train_targets = y_sub1[x_ind_p1] + np.random.normal(0, noise, size=y_sub1[x_ind_p1].shape)
                train_y_sub1 = sub1.train_targets

                sub1.env_ind = [0]
                sub1.bif_ind = []
                sub1_qc = sub1.increment_q_n(sub1_qc, train_x_sub1[0], x_sub1)

                train_x_sub2, train_y_sub2 = select_random_queries(nbr_rand_init, x_sub2, y_sub2, seed=seed,
                                                                   noise=noise)
                max_seen_resp_1_1D = torch.max(train_y_sub1)
                max_seen_resp_2_1D = torch.max(train_y_sub2)
                sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2 , sub2_like,
                                           query_counter=sub2_qc, nu=nu)

                for i in range(len(train_x_sub2)):
                    sub2_qc = sub2.increment_q_n(sub2_qc, train_x_sub2[i], x_sub2)

            elif len(children) == 2:
                sub1 = children[0]

                # Collecting the acquisition function for the child
                sub1_like = sub1.likelihood

                sub1.eval()

                sub1_like.eval()

                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    observed_pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
                y_mu1 = observed_pred1.mean

                y_conf1 = observed_pred1.stddev

                p1 = y_mu1 + gamma * y_conf1 / (torch.sqrt(sub1_qc))

                sub1.train()
                sub1_like.train()

                p1_max = torch.max(p1)

                x_ind_p1 = torch.argwhere(p1 == p1_max)[:, 0]

                """sub1.train_inputs = (sub1.train_inputs[0][-1:],)
                sub1.train_inputs[0][0][0] = x_sub1[x_ind_p1][0]"""
                sub1.train_inputs = (torch.reshape(x_sub1[x_ind_p1], (1, 1)),)
                train_x_sub1 = sub1.train_inputs[0][:, 0]

                sub1.train_targets = y_sub1[x_ind_p1] + np.random.normal(0, noise, size=y_sub1[x_ind_p1].shape)
                train_y_sub1 = sub1.train_targets

                sub1.env_ind = [0]
                sub1.bif_ind = []
                sub1_qc = sub1.increment_q_n(sub1_qc, train_x_sub1[0], x_sub1)
                # ========================================================================================

                # doing the same for the second child
                sub2 = children[1]

                # Collecting the acquisition function for the child
                sub2_like = sub2.likelihood

                sub2.eval()

                sub2_like.eval()

                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    observed_pred2 = models.make_prediction(sub2, x_sub2, sub2_like)
                y_mu2 = observed_pred2.mean

                y_conf2 = observed_pred2.stddev

                p2 = y_mu2 + gamma * y_conf2 / (torch.sqrt(sub2_qc))

                sub2.train()
                sub2_like.train()

                p2_max = torch.max(p2)

                x_ind_p2 = torch.argwhere(p2 == p2_max)[:, 0]

                """sub1.train_inputs = (sub1.train_inputs[0][-1:],)
                sub1.train_inputs[0][0][0] = x_sub1[x_ind_p1][0]"""
                sub2.train_inputs = (torch.reshape(x_sub2[x_ind_p2], (1, 1)),)
                train_x_sub2 = sub2.train_inputs[0][:, 0]

                sub2.train_targets = y_sub2[x_ind_p2] + np.random.normal(0, noise, size=y_sub2[x_ind_p2].shape)
                train_y_sub2 = sub2.train_targets

                sub2.env_ind = [0]
                sub2.bif_ind = []
                sub2_qc = sub2.increment_q_n(sub2_qc, train_x_sub2[0], x_sub2)

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

            # No bottom-up: first parent query is selected randomly (no child prior to guide it)
            train_x_hier, train_y_hier = hierarchical_select_random_queries(1, x_hier, y_hier, seed=seed, noise=noise)

            max_seen_resp_2D = torch.max(train_y_hier)

            prior_hierarchical_kernel = hmodel.hierarchical_kernel("add_kernel", sub1, sub2)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # Pass zeros as dummy prior_map — ConstantMean overrides it immediately in the constructor
            master = hierarchical_model(train_x_hier, train_y_hier - torch.mean(train_y_hier), x_hier, likelihood,
                                        prior_hierarchical_kernel,
                                        torch.zeros(dimension, dimension), kernel_op='add_kernel',
                                        sub_models=[sub1, sub2],
                                        kappa=kappa, query_counter=hier_qc)

            for i in range(len(train_x_hier)):
                hier_qc = master.increment_q_n(hier_qc, train_x_hier[i], x_hier)

            master.eval()
            likelihood.eval()

            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

        with gpytorch.settings.lazily_evaluate_kernels(state=False):

            c1_r2, c2_r2 = vi.child_contour_r2(master.sub_models, [x_sub1, x_sub2],
                                               [y_sub1, y_sub2])

        child_1_r2.append(c1_r2)
        child_2_r2.append(c2_r2)
        best_f_hier = torch.max(master.train_targets)
        acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred, hier_qc, acq_func=acq_func, best_f=best_f_hier)

        next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)

        next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins,
                                                                                     test_x_hier,
                                                                                     y_hier, noise=noise)

        y_mu_point_a = hmodel.get_y_mu_point_value(next_query_pins[0], y_mu1, x_sub1)
        y_mu_point_b = hmodel.get_y_mu_point_value(next_query_pins[1], y_mu2, x_sub2)

        y_conf_point_a = hmodel.get_y_mu_point_value(next_query_pins[0], y_conf1, x_sub1)
        y_conf_point_b = hmodel.get_y_mu_point_value(next_query_pins[1], y_conf2, x_sub2)

        y_qc_a = hmodel.get_y_mu_point_value(next_query_pins[0], sub1_qc, x_sub1)
        y_qc_b = hmodel.get_y_mu_point_value(next_query_pins[1], sub2_qc, x_sub2)

        next_query_value_random, max_seen_resp_2D = models.update_max_seen_response_no_norm(next_query_value_random,
                                                                                            max_seen_resp_2D)

        response = torch.tensor(next_query_value_random)

        cont1 = models.compute_acq_value(y_mu_point_a, y_conf_point_a, y_qc_a, gamma, acq_func, max_seen_resp_1_1D)
        norm1 = torch.max(models.compute_acq_value(y_mu1, y_conf1, sub1_qc, gamma, acq_func, max_seen_resp_1_1D))
        cont1_scaled = torch.nan_to_num(cont1 / norm1)

        cont2 = models.compute_acq_value(y_mu_point_b, y_conf_point_b, y_qc_b, gamma, acq_func, max_seen_resp_2_1D)
        norm2 = torch.max(models.compute_acq_value(y_mu2, y_conf2, sub2_qc, gamma, acq_func, max_seen_resp_2_1D))
        cont2_scaled = torch.nan_to_num(cont2 / norm2)

        div = torch.exp(cont1_scaled) + torch.exp(cont2_scaled)

        contribution1 = torch.nan_to_num(response * torch.exp(cont1_scaled) / div)
        contribution2 = torch.nan_to_num(response * torch.exp(cont2_scaled) / div)

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
                                                                                       False,
                                                                                       training_iter=training_iter)

        sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model1_1D_max_seen(sub2, sub2_like, train_x_sub2,
                                                                                       train_y_sub2,
                                                                                       x_sub2[next_query_indices[1]],
                                                                                       contribution2,
                                                                                       False,
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

        # No bottom-up: do not update the parent mean from children's posteriors

        master = hmodel.update_kernel_parameters(master, sub1, sub2)

        train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins,
                                                          response)
        master.set_train_data(train_x_hier, (train_y_hier - torch.mean(train_y_hier))/torch.std(train_y_hier), strict=False)
        #master.set_train_data(train_x_hier, train_y_hier/max_seen_resp_2D, strict=False)
        """
        train_x_sub1, train_x_sub2 = train_x_hier[:, 0], train_x_hier[:, 1]"""

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            # Find optimal model hyperparameters
            master.train()
            likelihood.train()

            # start = time.time()
            """master, likelihood = master.Hoptimize(likelihood, training_iter, train_x_hier,
                                                  train_y_hier / max_seen_resp_2D,
                                                  verbose=False)"""
            master, likelihood = master.Hoptimize(likelihood, training_iter, train_x_hier,
                                                  (train_y_hier - torch.mean(train_y_hier))/torch.std(train_y_hier),
                                                  verbose=False)

            master.eval()
            likelihood.eval()
            sub1.eval()
            sub1_like.eval()

            sub2.eval()
            sub2_like.eval()

            # Make a prediction, observed_pred = likelihood, prediction_mean = mu
            observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

        instantaneous_regret, next_query_pins_exploration_2D = models.get_instantaneous_regret(hierar_y_mu,
                                                                                               ground_truth_max_hier,
                                                                                               test_x_hier,
                                                                                               x_hier, y_hier)

        exploitation_score_2D = models.get_exploitation_score(next_query_value_mean, ground_truth_max_hier)

        better_exploration_score.append(instantaneous_regret)
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

        noi = str(noise).replace('.', ',')

        vi.contour_plot_1D(master.sub_models, [x_sub1, x_sub2], [y_sub1, y_sub2], [train_y_sub1, train_y_sub2],
                           f'/contour/Contour_init_{nbr_rand_init}_train_iter_{training_iter}_eps_{e}_k_{k}_g_{g}_nu_{n}_noise_{noi}',
                           model_name.lower(), folder_of_the_day, data_name, parent=master, query=q,
                           visualize=visualize)



    if final:
        return master, sub1, sub2, better_exploration_score, better_exploitation_score, heatmap_rep, child_1_r2, child_2_r2
    else:
        return better_exploration_score, better_exploitation_score, heatmap_rep, child_1_r2, child_2_r2


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals, nu_vals,
                       data_name, data_creation_func, eps, hierarchical_model, multi, seed, children=[], visualize=True, noise=0.1, alpha=0, disable_tqdm=False, acq_func='ucb'):
    if hierarchical_model == hmodel.Efficient_UCB_Hierarchical_GP:
        model_name = "Efficient"

    elif hierarchical_model == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
        model_name = "Lossless_Efficient"

    elif hierarchical_model == hmodel.Lossless_ConstantMean_UCB_Hierarchical_GP:
        model_name = "Ablation_No_Child_Prior"

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)

    os.makedirs(workspace, exist_ok=True)
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

                """if seed:
                    seed = np.arange(nbr_repetition)
                else:
                    seed = [False]*nbr_repetition"""

                if multi:
                    with mp.Pool(processes=nbr_repetition - 1) as pool:

                        # running the repetitions in parallel except for the last one
                        for i in range(nbr_repetition - 1):
                            p = pool.apply_async(run_repetition, (
                            kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model,
                            data_creation_func, eps, model_name, folder_of_the_day, data_name, False, children, visualize, seed[i], noise, disable_tqdm),
                            kwds={'acq_func': acq_func})
                            processes.append(p)

                        # must run the final block manually to allow return of the models
                        try:
                            master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                                hierarchical_model,
                                data_creation_func, eps, model_name, folder_of_the_day, data_name, final=True,
                                children=children, visualize=visualize, seed=seed[-1], noise=noise, disable_tqdm=disable_tqdm, acq_func=acq_func)
                        except:
                            try:
                                master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                    kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                                    hierarchical_model,
                                    data_creation_func, eps, model_name, folder_of_the_day, data_name, final=True,
                                    children=children, visualize=visualize, seed=seed[-1], noise=noise, disable_tqdm=disable_tqdm, acq_func=acq_func)
                            except:
                                try:
                                    master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                        kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                                        hierarchical_model,
                                        data_creation_func, eps, model_name, folder_of_the_day, data_name, final=True,
                                        children=children, visualize=visualize, seed=seed[-1], noise=noise, disable_tqdm=disable_tqdm, acq_func=acq_func)
                                except:
                                    continue

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
                        master, sub1, sub2, rep_exploration_score, rep_exploitation_score, heatmap_rep, child_1_r2, child_2_r2 = run_repetition(
                                kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter,
                                hierarchical_model,
                                data_creation_func, eps, model_name, folder_of_the_day, data_name, final=True,
                                children=children, visualize=visualize, seed=seed[i], noise=noise, disable_tqdm=disable_tqdm, acq_func=acq_func)

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
                noi = str(noise).replace('.', ',')

                torch.save(master.state_dict(),
                           f'{data_name}/{model_name.lower()}{folder_of_the_day}/models/{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}_eps_{e}_k_{k}_g_{g}_nu_{n}_noise_{noi}.pth')

                y = np.mean(better_exploration_score, axis=0)
                y = np.insert(y, 0, np.zeros((2 - len(children))*nbr_rand_init))[:nbr_query]
                over_explor.append(y)
                std = np.std(better_exploration_score, axis=0) / np.sqrt(len(better_exploration_score))
                std = np.insert(std, 0, np.zeros((2 - len(children))*nbr_rand_init))[:nbr_query]
                plt.plot(y, label='Instantaneous Regret')
                plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)
                """y = np.mean(better_exploitation_score, axis=0)
                y = np.insert(y, 0, np.zeros((2 - len(children))*nbr_rand_init))[:nbr_query]

                over_exploit.append(y)"""

                r2 = vi.heatmap_r_score(heatmap_data, y_hier)
                r2 = np.insert(r2, 0, np.zeros(((2 - len(children)) *nbr_rand_init, 1)), axis=1)[:, :nbr_query]
                r2_avg = np.mean(r2, axis=0)
                r2_std = np.std(r2, axis=0) / np.sqrt(len(r2))
                r2_std = np.insert(r2_std, 0, np.zeros((2 - len(children)) *nbr_rand_init))[:nbr_query]
                plt.plot(r2_avg, label="Parent R2")
                plt.fill_between(range(len(r2_std)), r2_avg-r2_std, r2_avg + r2_std, alpha=0.4)

                all_children = np.concatenate([child_1_r2_data, child_2_r2_data])
                all_children = np.insert(all_children, 0, np.zeros(((2 - len(children))*nbr_rand_init, 1)), 1)[:, :nbr_query]

                avg_child = np.mean(all_children, axis=0)
                std_child = np.std(all_children, axis=0) / np.sqrt(len(all_children))

                plt.plot(avg_child, label='Child Avg R2')
                plt.fill_between(range(len(avg_child)), avg_child - std_child, avg_child + std_child, alpha=0.4)

                plt.legend()
                plt.ylim(-0.1, 1.1)
                plt.ylabel('Performance')
                plt.xlabel('Query Number')

                plt.title(f'{model_name} HGP-BO {nbr_repetition} repetitions with kappa {k} Gamma {g} Nu {n} Init {nbr_rand_init}')
                plt.savefig(
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}_eps_{e}_k_{k}_g_{g}_nu_{n}_noise_{noi}.svg')
                plt.savefig(
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/png/{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}_eps_{e}_k_{k}_g_{g}_nu_{n}_noise_{noi}.png')

                plt.close()

                vi.model_heatmap(heatmap_data[:, -1, :], x_hier, y_hier,
                                 f'Heatmap_{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}_eps_{e}_k_{k}_g_{g}_nu_{n}_noise_{noi}',
                                 model_name.lower(), folder_of_the_day, data_name)
                vi.model_contour_3d(heatmap_data[:, -1, :], x_hier, y_hier,
                                 f'Parent_Contour_{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}_eps_{e}_k_{k}_g_{g}_nu_{n}_noise_{noi}',
                                 model_name.lower(), folder_of_the_day, data_name)

                data = np.mean(heatmap_data[:, -1, :], axis=0)

                re_output = np.reshape(data, y_hier.shape)
                df = pd.DataFrame(re_output)
                df.to_csv(
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}_eps_{e}_k_{k}_g_{g}_nu_{n}_noise_{noi}.csv')
                df = pd.DataFrame(y_hier)
                df.to_csv(
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/True_State_Space_Values.csv')

                auc = np.sum(y + avg_child + r2_avg)
                if not disable_tqdm:
                    print(f"explor {y[-1] * 100} + {std[-1] * 100}")
                    print(f"r2 avg {r2_avg[-1] * 100} + {r2_std[-1] * 100}")
                    print(f"child r2 {avg_child[-1] * 100} + {std_child[-1] * 100}")

                    print(f"AUC {np.sum(auc) * 100}")
                    print(f'\n{data_name} {model_name} Kappa {k} Gamma {g} Nu {n} eps_{e}_ complete!\n')

                df = pd.DataFrame(
                    [f'kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_eps_{e}_init_{nbr_rand_init}_train_iter_{training_iter}',
                     y[-1], r2_avg[-1], avg_child[-1], auc])
                df.index = ['name', 'instantaneous_regret', 'parent_r2', 'avg_child_r2', 'auc']
                df.to_csv(f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}_eps_{e}_k_{k}_g_{g}_nu_{n}_noise_{noi}.csv')

                list_models.append([f'kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_eps_{e}_init_{nbr_rand_init}_train_iter_{training_iter}', master, y, r2_avg, avg_child])

            # Joint Section
            """joint_performance(over_exploit, over_explor, kappa, gamma, nu_vals, folder_of_the_day, dimension, nbr_query,
                              nbr_repetition, data_name, model_name)

            joint_plots(over_exploit, over_explor, kappa, gamma, nu_vals, folder_of_the_day, dimension, nbr_query,
                        nbr_repetition, data_name, model_name)"""

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

    warnings.filterwarnings('ignore')

    dimension = 32
    nbr_query = 100
    training_iter = 10  # Found through HP Testing
    nbr_repetition = 10
    k_vals = [7.5]
    g_vals = [2]
    nu_vals = [0.5]  # Found through HP Testing
    multi = True
    acq = 'ucb'
    h_model = [hmodel.Lossless_ConstantMean_UCB_Hierarchical_GP]
    process = []
    nbr_rand_init = 3  # Found through HP Testing
    seed = [False]*nbr_repetition
    noises = [0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7, 0.8, 0.9]
    over_explor = []
    over_r2 = []
    over_child_r2 = []
    over_auc = []
    for dataset_num in [2,10,11,12,13]:
        for h in h_model:
            data_name, data_creation_func, eps = get_dataset_info(dataset_num)

            if h == hmodel.Efficient_UCB_Hierarchical_GP:
                model_name = "Efficient"

            elif h == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
                model_name = "Lossless_Efficient"

            elif h == hmodel.Lossless_ConstantMean_UCB_Hierarchical_GP:
                model_name = "Ablation_No_Child_Prior"

            current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
            current_dateday = datetime.now().strftime("%Y-%m-%d")
            workspace = f"{data_name}/{model_name.lower()}"
            folder_of_the_day = '/data-' + str(current_dateday)

            name, master, explor, r2, child_r2 = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                               nu_vals, data_name, data_creation_func, eps, h, multi, seed, noise=0.1, visualize=True, disable_tqdm=False, acq_func=acq)[0]
