import gpytorch
import synthetic_models as models

import hmodel_synthetic as hmodel
from dataset_actions import *
from datetime import datetime
import visualization_information as vi
import hmodel_gpu as hgpu
from tqdm import tqdm

def joint_plots(joint_exploit, joint_explor, k_vals, folder_of_the_day, dimension, nbr_query, nbr_repetition, data_name):
    for i, kappa in enumerate(k_vals):
        plt.plot(list(range(nbr_query)), joint_exploit[i], label=f'Kappa {str(kappa)}')
    plt.xlabel('Number of Queries')
    plt.ylabel('Exploitation score')
    plt.title(f'Exploitation Score of Changing HGPBO model')
    plt.legend()
    plt.ylim((0, 1.1))
    plt.savefig(
        f"{data_name}/changing{folder_of_the_day}/differentiable_plots/Joint_Exploit_{data_name}_Change_HGP-BO_{nbr_repetition}_repetitions_eps0,75_dim_{dimension}.png")
    plt.close()

    for i, kappa in enumerate(k_vals):
        plt.plot(list(range(nbr_query)), joint_explor[i], label=f'Kappa {str(kappa)}')
    plt.xlabel('Number of Queries')
    plt.ylabel('Exploration score')
    plt.title(f'Exploration Score of Changing HGPBO model')
    plt.legend()
    plt.ylim((0, 1.1))
    plt.savefig(
        f"{data_name}/changing{folder_of_the_day}/differentiable_plots/Joint_Explor_{data_name}_Change_HGP-BO_{nbr_repetition}_repetitions_eps0,75_dim_{dimension}.png")
    plt.close()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}\n")

    dimension = 10
    max_seen_resp_2D = 0
    max_seen_resp_1_1D = 0
    max_seen_resp_2_1D = 0
    prior_map = torch.zeros(dimension, dimension, device=device)
    list_prior_map = []
    list_objective_mean_map = []

    dataset_num = 3
    if dataset_num == 1:
        raise AssertionError("Haven't really set it all up for the first dataset")
    elif dataset_num == 2:
        data_name = 'synthetic_tests'
        data_creation_func = generate_sin_cos_dataset
    elif dataset_num == 3:
        data_name = 'synthetic3'
        data_creation_func = generate_synthetic3_dataset
    else:
        raise AssertionError("Dataset number invalid")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"C:/Users/preda/PycharmProjects/HierarchicalGPBO/model_testing/{data_name}/changing"
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

    nbr_query = 30
    training_iter = 5
    nbr_repetition = 10
    nbr_rand_init = 5
    eps = 2

    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    ground_truth_max_1 = torch.max(y_sub1)
    ground_truth_max_2 = torch.max(y_sub2)
    ground_truth_max_hier = torch.max(y_hier)

    x_sub1_cuda, x_sub2_cuda = x_sub1.to(device), x_sub2.to(device)
    y_sub1_cuda, y_sub2_cuda = y_sub1.to(device), y_sub2.to(device)
    test_x_hier_cuda = test_x_hier.to(device)
    x_hier_cuda = x_hier.to(device)
    y_hier_cuda = y_hier.to(device)

    max_seen_resp_1_1D_cuda = torch.tensor(max_seen_resp_1_1D).to(device)
    max_seen_resp_2_1D_cuda = torch.tensor(max_seen_resp_2_1D).to(device)

    ks = [1, 2, 3, 4, 5, 6]
    over_exploit = []
    over_explor = []
    heatmap_data = []

    for kappa in ks:
        better_exploration_score = []
        better_exploitation_score = []
        for repetition in range(nbr_repetition):
            for q in tqdm(range(nbr_query)):
                #print(f'Query num: {q}')
                if q == 0:
                    # Need to initialize the model - Will be random in this method
                    train_x_sub1, train_y_sub1 = select_random_queries(nbr_rand_init, x_sub1_cuda, y_sub1_cuda)
                    train_x_sub2, train_y_sub2 = select_random_queries(nbr_rand_init, x_sub2_cuda, y_sub2_cuda)
                    train_x_hier, train_y_hier = hierarchical_select_random_queries(nbr_rand_init, x_hier_cuda,
                                                                                    y_hier_cuda)
                    max_seen_resp_1_1D_cuda = torch.max(train_y_sub1)
                    max_seen_resp_2_1D_cuda = torch.max(train_y_sub2)
                    max_seen_resp_2D_cuda = torch.max(train_y_hier)

                    train_y_sub1 = train_y_sub1 / max_seen_resp_1_1D_cuda
                    train_y_sub2 = train_y_sub2 / max_seen_resp_2_1D_cuda
                    train_y_hier = train_y_hier / max_seen_resp_2D_cuda

                    # Need to modify this section such that the model is receiving the partial contribution
                    train_x_sub1, train_y_sub1 = update_training_data(train_x_sub1, train_y_sub1, train_x_hier[:, 0],
                                                                      train_y_hier)
                    train_x_sub2, train_y_sub2 = update_training_data(train_x_sub2, train_y_sub2, train_x_hier[:, 1],
                                                                      train_y_hier)

                    sub1_like = gpytorch.likelihoods.GaussianLikelihood()
                    sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1, sub1_like)
                    sub1_like = sub1_like.to(device)
                    sub1 = sub1.to(device)

                    sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                    sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like)
                    sub2_like = sub2_like.to(device)
                    sub2 = sub2.to(device)

                    sub1.eval()
                    sub2.eval()

                    sub1_like.eval()
                    sub2_like.eval()
                    with gpytorch.settings.lazily_evaluate_kernels(state=False):
                        observed_pred1 = models.make_prediction(sub1, x_sub1_cuda, sub1_like)
                        observed_pred2 = models.make_prediction(sub2, x_sub2_cuda, sub2_like)
                    y_mu1 = observed_pred1.mean
                    y_mu2 = observed_pred2.mean

                    for i in range(len(prior_map)):
                        for j in range(len(prior_map)):
                            prior_map[i, j] = y_mu1[i] + y_mu2[j]

                    prior_map_max = torch.max(prior_map)

                    prior_hierarchical_kernel = hmodel.hierarchical_kernel("add_kernel", sub1, sub2)
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    master = hgpu.GPU_Efficient_UCB_Hierarchical_GP(train_x_hier, train_y_hier, likelihood,
                                                                  prior_hierarchical_kernel,
                                                                  prior_map / prior_map_max, kernel_op='add_kernel',
                                                                  sub_models=[sub1, sub2], kappa=kappa, device=device)

                    likelihood = likelihood.to(device)
                    master = master.to(device)

                    master.eval()
                    likelihood.eval()

                    with gpytorch.settings.lazily_evaluate_kernels(state=False):
                        observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier_cuda, likelihood)

                acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred)

                next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier_cuda)

                next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins,
                                                                                             test_x_hier_cuda,
                                                                                             y_hier_cuda)

                y_mu_point_a = hmodel.get_y_mu_point_value(next_query_pins[0], y_mu1, x_sub1_cuda)
                y_mu_point_b = hmodel.get_y_mu_point_value(next_query_pins[1], y_mu2, x_sub2_cuda)

                response_1, response_2 = hmodel.compute_responses(y_mu_point_a, y_mu_point_b, next_query_value_random)

                response_1, max_seen_resp_1_1D_cuda = models.update_max_seen_response(response_1, max_seen_resp_1_1D_cuda)
                response_2, max_seen_resp_2_1D_cuda = models.update_max_seen_response(response_2, max_seen_resp_2_1D_cuda)
                next_query_value_random, max_seen_resp_2D_cuda = models.update_max_seen_response(next_query_value_random,
                                                                                            max_seen_resp_2D_cuda)

                response = torch.tensor(next_query_value_random, device=device)
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
                                                                                      x_sub1_cuda[next_query_indices[0]],
                                                                                      response_1,
                                                                                      training_iter=training_iter)

                sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model1_1D(sub2, sub2_like, train_x_sub2,
                                                                                      train_y_sub2,
                                                                                      x_sub2_cuda[next_query_indices[1]],
                                                                                      response_2,
                                                                                      training_iter=training_iter)

                sub1.eval()
                sub1_like.eval()

                sub2.eval()
                sub2_like.eval()

                # Make a prediction, observed_pred = likelihood
                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    observed_pred1 = models.make_prediction(sub1, x_sub1_cuda, sub1_like)
                    observed_pred2 = models.make_prediction(sub2, x_sub2_cuda, sub2_like)

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

                # TODO: adjust the response that will be propagated by contribution stuff

                train_x_sub1, train_y_sub1 = update_training_data(train_x_sub1, train_y_sub1, next_query_pins[0],
                                                                  response)
                train_x_sub2, train_y_sub2 = update_training_data(train_x_sub2, train_y_sub2, next_query_pins[1],
                                                                  response)
                """
                train_x_sub1, train_x_sub2 = train_x_hier[:, 0], train_x_hier[:, 1]"""
                sub1.set_train_data(train_x_sub1, train_y_sub1, strict=False)
                sub2.set_train_data(train_x_sub2, train_y_sub2, strict=False)

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
                    observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier_cuda, likelihood)

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
                print(f'Exploitation_Score: {torch.round(exploitation_score_2D, decimals=4)}')
"""
                better_exploration_score.append(exploration_score_2D)
                better_exploitation_score.append(exploitation_score_2D)

            pred = hmodel.make_Hierarchique_prediction(master, test_x_hier_cuda, master.likelihood)
            master_like = master.likelihood(pred)
            heatmap_data.append(master_like.mean)
            print(f'\nRepetition {repetition} complete!\n')

        # Currently only takes the last model of the repetitions, currently too lazy to fix
        """vi.contour_plot_1D(master.sub_models, x_sub1, [y_sub1, y_sub2],
                           f'/contour/Contour_{data_name}_Norm_Efficient_HGP-BO_{nbr_repetition}_repetitions_eps0,75_dim_{dimension}_kappa_{kappa}',
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
        plt.title(f'Norm_Efficient HGP-BO {nbr_repetition} repetitions with kappa value {kappa}')
        plt.savefig(
            f'{data_name}/efficient_3D{folder_of_the_day}/differentiable_plots/Norm_Efficient_Prop_{data_name}_HGP-BO_{nbr_repetition}_repetitions_kappa_{kappa}_eps_0,75')
        plt.close()

        vi.model_heatmap(heatmap_data, x_hier, y_hier,
                         f'/Heatmap_{data_name}_Norm_Efficient_HGP-BO_{nbr_repetition}_repetitions_eps0,75_dim_{dimension}_kappa_{kappa}',
                         "efficient_3D", folder_of_the_day, data_name)

    # Joint Section

    joint_plots(over_exploit, over_explor, ks, folder_of_the_day, dimension, nbr_query, nbr_repetition)

"""
