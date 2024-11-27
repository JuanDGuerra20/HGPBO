import gpytorch
import models_3d as models
import hmodel_3d as hmodel
from dataset_actions_3d import *
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
import multiprocessing as mp


def joint_plots(joint_exploit, joint_explor, kappa, g_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition,
                data_name):
    for i, gamma in enumerate(g_vals):
        plt.plot(joint_exploit[i], label=f'gamma {gamma}')

    k = str(kappa).replace('.', ',')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploitation Score')
    plt.title(f'Joint Norm_Efficient Propagation HGPBO {nbr_repetition} Exploitation Kappa {kappa}')
    plt.savefig(
        f'{data_name}/efficient_3D{folder_of_the_day}/differentiable_plots/Joint_Norm_Efficient_Propagation_HGPBO_{nbr_repetition}_Exploitation_kappa_{k}')
    plt.close()

    for i, gamma in enumerate(g_vals):
        plt.plot(joint_explor[i], label=f'gamma {gamma}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploration Score')
    plt.title(f'Joint Norm_Efficient Propagation HGPBO {nbr_repetition} Exploration Kappa {kappa}')
    plt.savefig(
        f'{data_name}/efficient_3D{folder_of_the_day}/differentiable_plots/Joint_Norm_Efficient_Propagation_HGPBO_{nbr_repetition}_Exploration_kappa{k}')
    plt.close()

def run_repetition(kappa, gamma, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model, data_creation_func, eps, final=False):
    
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

    for q in range(nbr_query):
        if q == 0:
            # Need to initialize the model - Will be random in this method
            train_x_sub1, train_y_sub1 = select_random_queries(nbr_rand_init, x_sub1, y_sub1)
            train_x_sub2, train_y_sub2 = select_random_queries(nbr_rand_init, x_sub2, y_sub2)
            train_x_sub3, train_y_sub3 = select_random_queries(nbr_rand_init, x_sub3, y_sub3)
            train_x_hier, train_y_hier = hierarchical_select_random_queries(nbr_rand_init, x_hier, y_hier)
            max_seen_resp_1_1D = torch.max(train_y_sub1)
            max_seen_resp_2_1D = torch.max(train_y_sub2)
            max_seen_resp_3_1D = torch.max(train_y_sub3)

            max_seen_resp_2D = torch.max(train_y_hier)

            """
            train_y_sub1 = train_y_sub1 / max_seen_resp_1_1D
            train_y_sub2 = train_y_sub2 / max_seen_resp_2_1D
            train_y_hier = train_y_hier / max_seen_resp_2D"""

            # Need to modify this section such that the model is receiving the partial contribution
            train_x_sub1, train_y_sub1 = update_training_data(train_x_sub1, train_y_sub1,
                                                                train_x_hier[:, 0], train_y_hier)
            train_x_sub2, train_y_sub2 = update_training_data(train_x_sub2, train_y_sub2,
                                                                train_x_hier[:, 1], train_y_hier)
            train_x_sub3, train_y_sub3 = update_training_data(train_x_sub3, train_y_sub3,
                                                                train_x_hier[:, 2], train_y_hier)

            sub1_like = gpytorch.likelihoods.GaussianLikelihood()
            sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1 / max_seen_resp_1_1D, sub1_like, sub1_qc)

            sub2_like = gpytorch.likelihoods.GaussianLikelihood()
            sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2 / max_seen_resp_2_1D, sub2_like, sub2_qc)

            sub3_like = gpytorch.likelihoods.GaussianLikelihood()
            sub3 = models.ExactGPModel(train_x_sub3, train_y_sub3 / max_seen_resp_3_1D, sub3_like, sub3_qc)

            sub1.eval()
            sub2.eval()
            sub3.eval()

            sub1_like.eval()
            sub2_like.eval()
            sub3_like.eval()

            for i in range(len(train_x_sub1)):
                sub1_qc = sub1.increment_q_n(sub1_qc, train_x_sub1[i], x_sub1)
                sub2_qc = sub2.increment_q_n(sub2_qc, train_x_sub2[i], x_sub2)
                sub3_qc = sub3.increment_q_n(sub3_qc, train_x_sub3[i], x_sub3)


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

            prior_hierarchical_kernel = hmodel.hierarchical_kernel("add_kernel", [sub1, sub2, sub3])
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            master = hmodel.Efficient_UCB_Hierarchical_GP(train_x_hier, train_y_hier / max_seen_resp_2D, likelihood,
                                                            prior_hierarchical_kernel,
                                                            prior_map / prior_map_max, kernel_op='add_kernel',
                                                            sub_models=[sub1, sub2, sub3],
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
        cont2 = y_mu_point_b + gamma * torch.nan_to_num(y_conf_point_b / torch.sqrt(y_qc_b))
        cont3 = y_mu_point_c + gamma * torch.nan_to_num(y_conf_point_c / torch.sqrt(y_qc_c))

        contribution1 = torch.nan_to_num(response * torch.exp(cont1) / (torch.exp(cont1) + torch.exp(cont2) + torch.exp(cont3)))
        contribution2 = torch.nan_to_num(response * torch.exp(cont2) / (torch.exp(cont1) + torch.exp(cont2) + torch.exp(cont3)))
        contribution3 = torch.nan_to_num(response * torch.exp(cont3) / (torch.exp(cont1) + torch.exp(cont2) + torch.exp(cont3)))


        response_1, max_seen_resp_1_1D = models.update_max_seen_response_no_norm(contribution1,
                                                                                    max_seen_resp_1_1D)
        response_2, max_seen_resp_2_1D = models.update_max_seen_response_no_norm(contribution2,
                                                                                    max_seen_resp_2_1D)
        response_3, max_seen_resp_3_1D = models.update_max_seen_response_no_norm(contribution3,
                                                                                    max_seen_resp_3_1D)

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
                                                                                contribution1, max_seen_resp_1_1D,
                                                                                training_iter=training_iter)

        sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model1_1D_max_seen(sub2, sub2_like, train_x_sub2,
                                                                                train_y_sub2,
                                                                                x_sub2[next_query_indices[1]],
                                                                                contribution2, max_seen_resp_2_1D,
                                                                                training_iter=training_iter)

        sub3, sub3_like, train_x_sub3, train_y_sub3 = hmodel.update_model1_1D_max_seen(sub3, sub3_like,
                                                                                        train_x_sub3,
                                                                                        train_y_sub3,
                                                                                        x_sub3[
                                                                                            next_query_indices[
                                                                                                2]],
                                                                                        contribution3,
                                                                                        max_seen_resp_3_1D,
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
        master.set_train_data(train_x_hier, train_y_hier/max_seen_resp_2D, strict=False)

        """
        train_x_sub1, train_x_sub2 = train_x_hier[:, 0], train_x_hier[:, 1]"""


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
        pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, master.likelihood)
        master_like = master.likelihood(pred)
        heatmap_rep.append(master_like.mean.detach().cpu().numpy())
    
    
    if final:
        return master, sub1, sub2, sub3, better_exploration_score, better_exploitation_score, heatmap_rep
    else:
        return better_exploration_score, better_exploitation_score, heatmap_rep


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals, data_name, data_creation_func,
                           eps, hierarchical_model):
    
    if hierarchical_model == hmodel.Efficient_UCB_Hierarchical_GP:
        model_name = "Efficient"

    elif hierarchical_model == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
        model_name = "Lossless_Efficient"

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/efficient_3D"
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

    x_sub1, y_sub1, x_sub2, y_sub2, x_sub3, y_sub3, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)


    list_prior_map = []
    list_objective_mean_map = []
    final_exploitation_metric = []
    final_exploration_metric = []

    for kappa in k_vals:
        over_exploit = []
        over_explor = []
        for gamma in g_vals:
            better_exploration_score = []
            better_exploitation_score = []
            heatmap_data = []
            processes = []

            with mp.Pool(processes=nbr_repetition - 1) as pool:
                for i in range(nbr_repetition - 1):
                    p = pool.apply_async(run_repetition, (kappa, gamma, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model, data_creation_func, eps, ))
                    processes.append(p)
                
                master, sub1, sub2, sub3, rep_exploration_score, rep_exploitation_score, heatmap_rep = run_repetition(kappa, gamma, nbr_query, nbr_rand_init, dimension, training_iter, hierarchical_model, data_creation_func, eps, final=True)

                better_exploration_score.append(rep_exploration_score)
                better_exploitation_score.append(rep_exploitation_score)
                heatmap_data.append(heatmap_rep)

                for proc in processes:
                    rep_exploration_score, rep_exploitation_score, heatmap_rep = proc.get()

                    better_exploration_score.append(rep_exploration_score)
                    better_exploitation_score.append(rep_exploitation_score)
                    heatmap_data.append(heatmap_rep)
            
            for heat in heatmap_data:
                print(len(heat))
                print(type(heat))
                for thing in heat:
                    print(len(thing))
                    print(type(thing))
            
            heatmap_data = np.array(heatmap_data)
            k = str(kappa).replace('.', ',')
            g = str(gamma).replace('.', ',')

            # Currently only takes the last model of the repetitions, currently too lazy to fix
            vi.contour_plot_1D(master.sub_models, x_sub1, [y_sub1 / torch.max(y_sub1), y_sub2 / torch.max(y_sub2), y_sub3 / torch.max(y_sub3)],
                               f'/contour/Contour_{data_name}_HGP-BO_{nbr_repetition}_repetitions_dim_{dimension}_kappa_{k}_gamma_{g}',
                               model_name.lower(), folder_of_the_day, data_name)
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

            r2 = vi.heatmap_r_score(heatmap_data, y_hier)
            plt.plot(r2, label="Heatmap R2")

            plt.legend()
            plt.ylim(0, 1.1)


            plt.title(f'Norm_Efficient HGP-BO {nbr_repetition} repetitions with kappa {k} gamma {g}')
            plt.savefig(
                f'{data_name}/efficient_3D{folder_of_the_day}/differentiable_plots/Norm_Efficient_Prop_{data_name}_HGP-BO_{nbr_repetition}_repetitions_kappa_{k}_gamma_{g}')
            plt.close()
        # Joint Section

        joint_plots(over_exploit, over_explor, kappa, g_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition, data_name)

if __name__ == '__main__':

    dimension = 10
    nbr_query = 80
    training_iter = 5
    nbr_repetition = 5
    nbr_rand_init = 5
    k_vals = [5, 6]
    g_vals = [3, 4, 5]

    h_model = [hmodel.Efficient_UCB_Hierarchical_GP]
    process = []

    for h in h_model:
        for dataset_num in [5]:
            data_name, data_creation_func, eps = get_dataset_info(dataset_num)

            p = mp.Process(target=training_procedure, args=(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals, data_name, data_creation_func,
                               eps, h,))
            process.append(p)
            p.start()
            print(f"ID of process: {p.pid}")
