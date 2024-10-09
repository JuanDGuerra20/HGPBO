import gpytorch
import three_d_models as models
import hmodel_3d as hmodel
from dataset_actions_3d import *
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm


def joint_plots(joint_exploit, joint_explor, k_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition,
                data_name):
    for i, kappa in enumerate(k_vals):
        plt.plot(joint_exploit[i], label=f'kappa {kappa}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploitation Score')
    plt.title(f'Joint Norm_Efficient Propagation HGPBO {nbr_repetition} Exploitation')
    plt.savefig(
        f'{data_name}/efficient_3D{folder_of_the_day}/differentiable_plots/Joint_Norm_Efficient_Propagation_HGPBO_{nbr_repetition}_Exploitation')
    plt.close()

    for i, kappa in enumerate(k_vals):
        plt.plot(joint_explor[i], label=f'kappa {kappa}')

    plt.legend()
    plt.xlabel(f'Nbr Queries')
    plt.ylim((0, 1.1))
    plt.ylabel(f'Exploration Score')
    plt.title(f'Joint Norm_Efficient Propagation HGPBO {nbr_repetition} Exploration')
    plt.savefig(
        f'{data_name}/efficient_3D{folder_of_the_day}/differentiable_plots/Joint_Norm_Efficient_Propagation_HGPBO_{nbr_repetition}_Exploration')
    plt.close()


def training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, data_name, data_creation_func,
                           eps):

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"C:/Users/preda/PycharmProjects/HierarchicalGPBO/3D_testing/{data_name}/efficient_3D"
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

    ground_truth_max_hier = torch.max(y_hier)

    over_exploit = []
    over_explor = []
    heatmap_data = []
    prior_map = torch.zeros(dimension, dimension, dimension)
    list_prior_map = []
    list_objective_mean_map = []

    for kappa in k_vals:
        better_exploration_score = []
        better_exploitation_score = []
        heatmap_data = []
        for repetition in range(nbr_repetition):
            max_seen_resp_2D = 0
            max_seen_resp_1_1D = 0
            max_seen_resp_2_1D = 0
            max_seen_resp_3_1D = 0

            for q in tqdm(range(nbr_query)):
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
                    sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1 / max_seen_resp_1_1D, sub1_like)

                    sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                    sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2 / max_seen_resp_2_1D, sub2_like)

                    sub3_like = gpytorch.likelihoods.GaussianLikelihood()
                    sub3 = models.ExactGPModel(train_x_sub3, train_y_sub3 / max_seen_resp_3_1D, sub3_like)

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

                    _, y_reg1 = observed_pred1.confidence_region()
                    _, y_reg2 = observed_pred2.confidence_region()
                    _, y_reg3 = observed_pred3.confidence_region()

                    for i in range(len(prior_map)):
                        for j in range(len(prior_map)):
                            for k in range(len(prior_map)):
                                prior_map[i, j, k] = (y_reg1[i] + y_reg2[j] + y_reg3[k]) / 3

                    prior_map_max = torch.max(prior_map)

                    prior_hierarchical_kernel = hmodel.hierarchical_kernel("add_kernel", [sub1, sub2, sub3])
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    master = hmodel.Efficient_UCB_Hierarchical_GP(train_x_hier, train_y_hier / max_seen_resp_2D, likelihood,
                                                                  prior_hierarchical_kernel,
                                                                  prior_map / prior_map_max, kernel_op='add_kernel',
                                                                  sub_models=[sub1, sub2, sub3],
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
                y_mu_point_c = hmodel.get_y_mu_point_value(next_query_pins[2], y_mu3, x_sub3)

                y_conf_point_a = hmodel.get_y_mu_point_value(next_query_pins[0], y_conf1, x_sub1)
                y_conf_point_b = hmodel.get_y_mu_point_value(next_query_pins[1], y_conf2, x_sub2)
                y_conf_point_c = hmodel.get_y_mu_point_value(next_query_pins[2], y_conf3, x_sub3)

                next_query_value_random, max_seen_resp_2D = models.update_max_seen_response_no_norm(next_query_value_random,
                                                                                            max_seen_resp_2D)

                response = torch.tensor(next_query_value_random)

                cont1 = y_mu_point_a + kappa * y_conf_point_a
                cont2 = y_mu_point_b + kappa * y_conf_point_b
                cont3 = y_mu_point_c + kappa * y_conf_point_c

                contribution1 = response * np.exp(cont1) / (np.exp(cont1) + np.exp(cont2) + np.exp(cont3))
                contribution2 = response * np.exp(cont2) / (np.exp(cont1) + np.exp(cont2) + np.exp(cont3))
                contribution3 = response * np.exp(cont3) / (np.exp(cont1) + np.exp(cont2) + np.exp(cont3))


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

                _, y_reg1 = observed_pred1.confidence_region()
                _, y_reg2 = observed_pred2.confidence_region()
                _, y_reg3 = observed_pred3.confidence_region()

                for i in range(len(prior_map)):
                    for j in range(len(prior_map)):
                        for k in range(len(prior_map)):
                            prior_map[i, j, k] = (y_reg1[i] + y_reg2[j] + y_reg3[k]) / 3

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

            print(f'\nRepetition {repetition} complete!\n')

        # Currently only takes the last model of the repetitions, currently too lazy to fix
        vi.contour_plot_1D(master.sub_models, x_sub1, [y_sub1 / torch.max(y_sub1), y_sub2 / torch.max(y_sub2), y_sub3 / torch.max(y_sub3)],
                           f'/contour/Contour_{data_name}_Efficient_HGP-BO_{nbr_repetition}_repetitions_dim_{dimension}_kappa_{str(kappa).replace(".", ",")}',
                           'efficient_3D', folder_of_the_day, data_name)
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

        plt.legend()
        plt.ylim(0, 1.1)
        k = str(kappa).replace('.', ',')

        plt.title(f'Norm_Efficient HGP-BO {nbr_repetition} repetitions with kappa value {k}')
        plt.savefig(
            f'{data_name}/efficient_3D{folder_of_the_day}/differentiable_plots/Norm_Efficient_Prop_{data_name}_HGP-BO_{nbr_repetition}_repetitions_kappa_{k}')
        plt.close()
        """
        vi.model_heatmap(heatmap_data, x_hier, y_hier,
                         f'/Heatmap_{data_name}_Norm_Efficient_HGP-BO_{nbr_repetition}_repetitions_dim_{dimension}_kappa_{k}',
                         "efficient_3D", folder_of_the_day, data_name)"""

    # Joint Section

    joint_plots(over_exploit, over_explor, k_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition, data_name)

if __name__ == '__main__':

    dimension = 10
    nbr_query = 40
    training_iter = 5
    nbr_repetition = 2
    nbr_rand_init = 5
    k_vals = [5]

    for dataset_num in [4]:
        data_name, data_creation_func, eps = get_dataset_info(dataset_num)

        training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, data_name, data_creation_func,
                           eps)
