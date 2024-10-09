import gpytorch
import torch
import numpy as np
import time
from datetime import datetime

from dataset_actions import *
import models
import hierarchical_model as hmodel
import visualization_information as vi

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

training_iter = 10
nbr_query = 50
q = 0
nbr_rdm_points_ini = 5       # Number of random points to initialize the model
nbr_rdm_points_data = 20     # Number of random points from the Ground Truth (GT) EMG responses to create a dataset
kappa = 5
max_seen_resp = 0            # To store the maximum response observed, and normalize between 0 and 1 (arbitrary choice)
nbr_repetition = 5          # Number of repetition of the entire process to average the results
nbr_pins = 100               # Number of pins in he input space (2*matrix of 2*5 pins)

name_code = 'HGP_BO-test6-priorMAP-1model1D'
current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
current_dateday = datetime.now().strftime("%Y-%m-%d")

workspace_folder = (r'C:\Users\preda\PycharmProjects\HierarchicalGPBO') #path to folder

os.chdir(workspace_folder)

folder_of_the_day = (str(workspace_folder) + '/data-' + str(name_code) + str(current_dateday))

if os.path.exists(folder_of_the_day):
    print('Data folder is ready')
else:
    os.mkdir(folder_of_the_day)
    print('Data folder created')

def run_training_and_eval(nbr_query, nbr_repetition, kappa):
    list_prior_map = []
    # -----------------------------------------------------------------------------------------------------
    # Start running the entire process for nbr_repetition
    for repetition in range(nbr_repetition):
        startTime_repetition = time.time()
        print("repetition: ", repetition)
        prior_map_max = 0
        max_seen_resp_2D = 0
        max_seen_resp_1_1D = 0
        max_seen_resp_2_1D = 0
        response = torch.zeros(1)
        response1 = torch.zeros(1)
        response2 = torch.zeros(1)

        # Start training and evaluating over a number of query
        for q in range(nbr_query):
            if q == 0:
                # Initialize 2 training data set
                train_x1_1D, train_y1_1D = random_initialization_1D(nbr_rdm_points_ini, trainsC,
                                                                    max_seen_resp_1_1D, emg=EMG)

                likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
                model1 = models.ExactGPModel(train_x1_1D, train_y1_1D, likelihood1)

                model1.eval()
                likelihood1.eval()

                # Make a prediction, observed_pred = likelihood
                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    # Prediction of Hierarchical Model
                    observed_pred1 = models.make_prediction(model1, test_x_1D, likelihood1)

                y_mu1 = observed_pred1.mean

                for i in range(len(prior_map)):
                    for j in range(len(prior_map)):
                        prior_map[i][j] = y_mu1[i] + y_mu1[j]
                prior_map_max = torch.max(prior_map)
                list_prior_map.append(np.array(prior_map / prior_map_max))

                # If first query, random initialization of the model
                train_x_2D, train_y_2D = hmodel.random_initialization(nbr_rdm_points_ini, EMG, trainsC,
                                                                      max_seen_resp_2D, DT)
                train_x_2D = torch.tensor(train_x_2D, dtype=torch.float64)
                train_y_2D = torch.tensor(train_y_2D, dtype=torch.float64)

                prior_hierarchical_kernel = hmodel.hierarchical_kernel('add_kernel', model1, model1)

                # Initialize likelihood and model
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                model = hmodel.Hierarchical_GP(train_x_2D, train_y_2D, likelihood, prior_hierarchical_kernel,
                                               (prior_map / prior_map_max))

                # Get into evaluation (predictive posterior) mode
                model.eval()
                likelihood.eval()
                # Make a prediction, observed_pred = likelihood
                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    # Prediction of Hierarchical Model
                    observed_pred = hmodel.make_Hierarchique_prediction(model, test_x_2D, likelihood)
                list_objective_mean_map.append(np.array(np.reshape(observed_pred.mean, (10, 10))))

            vi.comparison(list_prior_map, list_objective_mean_map, q, Xmean_1D)

            # acquisition map based on 1D model prediction
            acquisition_map, hierar_y_mu = models.get_acquisition_map(kappa, observed_pred)

            # Get the coordinates of the next pins to exploite
            next_query_pins = models.get_next_query_pins(acquisition_map, test_x_2D)

            # Get the next query value
            next_query_value_random, next_query_value_mean = models.get_next_query_value(next_query_pins, X_2D, Y_2D)

            y_mu_point_a = hmodel.get_y_mu_point_value(next_query_pins[0:2], y_mu1, test_x_1D)
            y_mu_point_b = hmodel.get_y_mu_point_value(next_query_pins[2:4], y_mu1, test_x_1D)

            # next_query_value_random is the response of the system (emg)
            response_1, response_2 = hmodel.compute_responses(y_mu_point_a, y_mu_point_b, next_query_value_random)

            response_1, max_seen_resp_1_1D = models.update_max_seen_response(response_1, max_seen_resp_1_1D)
            response_2, max_seen_resp_1_1D = models.update_max_seen_response(response_2, max_seen_resp_1_1D)
            next_query_value_random, max_seen_resp_2D = models.update_max_seen_response(next_query_value_random,
                                                                                        max_seen_resp_2D)

            response = torch.tensor(next_query_value_random)
            response_1 = response_1.clone().detach()
            response_2 = response_2.clone().detach()

            # update unitary model with response_1 and response_2
            model1, likelihood1, train_x1_1D, train_y1_1D = hmodel.update_model1_1D(model1, likelihood1, train_x1_1D,
                                                                                    train_y1_1D, next_query_pins,
                                                                                    response_1,
                                                                                    response_2,
                                                                                    training_iter=training_iter)

            model1.eval()
            likelihood1.eval()

            # Make a prediction, observed_pred = likelihood
            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                observed_pred1 = models.make_prediction(model1, test_x_1D, likelihood1)

            y_mu1 = observed_pred1.mean


            # creating the prior map from the prediction of the submodel which is required for parent
            for i in range(len(prior_map)):
                for j in range(len(prior_map)):
                    prior_map[i][j] = y_mu1[i] + y_mu1[j]
            prior_map_max = torch.max(prior_map)

            list_prior_map.append(np.array(prior_map / prior_map_max))
            list_objective_mean_map.append(np.array(np.reshape(observed_pred.mean, (10, 10))))

            model.mean_module.map = torch.nn.Parameter(prior_map / prior_map_max)

            model = hmodel.update_kernel_parameters(model, model1, model1)

            # Update training data by adding next_query_value to the train dataset
            train_x_2D, train_y_2D = update_training_data(train_x_2D, train_y_2D, next_query_pins, response)

            # Update the model with the new training data
            model.set_train_data(train_x_2D, train_y_2D,
                                 strict=False)  # strict = False to add_kernel inputs with different shape

            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                # Find optimal model hyperparameters
                model.train()
                likelihood.train()

                model, likelihood = hmodel.Hoptimize(model, likelihood, training_iter, train_x_2D, train_y_2D,
                                                     verbose=False)
                # Get into evaluation (predictive posterior) mode
                model.eval()
                likelihood.eval()

                # Make a prediction, observed_pred = likelihood, prediction_mean = mu
                observed_pred = hmodel.make_Hierarchique_prediction(model, test_x_2D, likelihood)

            # -----------------------------------------------------------------------------------------------#
            # 2D HIERARCHICAL MODEL
            # -----------------------------------------------------------------------------------------------#
            # Compute exploration and exploitation score
            exploration_score_2D, next_query_pins_exploration_2D = models.get_exploration_score(hierar_y_mu,
                                                                                                ground_truth_max_2D,
                                                                                                test_x_2D,
                                                                                                X_2D, Y_2D)
            exploitation_score_2D = models.get_exploitation_score(next_query_value_mean, ground_truth_max_2D)
            # Record results and parameters
            # Query and repetitions
            array_repetition_2D[q + nbr_query * repetition] = repetition
            array_query_2D[q + nbr_query * repetition] = q
            # Scores
            array_exploration_score_2D[q + nbr_query * repetition] = exploration_score_2D
            array_exploitation_score_2D[q + nbr_query * repetition] = exploitation_score_2D
            # Pins selected
            array_pins_coord_2D[q + nbr_query * repetition][:] = next_query_pins
            array_pins_coord_exploration_2D[q + nbr_query * repetition][:] = next_query_pins_exploration_2D
            # Mean for each score and confidence level of a query over all rep
            array_explr_mean_2D[q] += (exploration_score_2D)
            array_explt_mean_2D[q] += (exploitation_score_2D)
            # -----------------------------------------------------------------------------------------------#

    # Compute the mean on each line which accumulated value over repetitions
    for q in range(nbr_query):
        array_explr_mean_2D[q] = array_explr_mean_2D[q] / nbr_repetition
        if array_explr_mean_2D[q] > 1:
            print("I hate this")
        array_explt_mean_2D[q] = array_explt_mean_2D[q] / nbr_repetition
        if array_explt_mean_2D[q] > 1:
            print("I hate this")

    array_pins_count_2D = count_pin(array_pins_coord_2D, test_x_2D)
    array_pins_count_exploration_2D = count_pin(array_pins_coord_exploration_2D, test_x_2D)

    data_dict_Hmodel = {'query_number': array_query_2D,
                        'exploration_score': array_exploration_score_2D,
                        'exploitation_score': array_exploitation_score_2D,
                        'pins_coord_exploration': array_pins_coord_exploration_2D,
                        'pins_count_exploration': array_pins_count_exploration_2D,
                        'pins_coord_exploitation': array_pins_coord_2D,
                        'pins_count_exploitation': array_pins_count_2D,
                        'exploration_score_mean_repetition': array_explr_mean_2D,
                        'exploitation_score_mean_repetition': array_explt_mean_2D,
                        'kappa_value': kappa,
                        'nbr_of_query_per_rep': nbr_query,
                        'nbr_of_repetition': nbr_repetition,
                        'pts_of_ini': nbr_rdm_points_ini,
                        'emg': EMG,
                        'prior_map': list_prior_map,
                        'objective_mean_map': list_objective_mean_map}

    return data_dict_Hmodel

if __name__ == '__main__':
    startTime = time.time()
    max_seen_resp_2D = 0
    max_seen_resp_1_1D = 0
    max_seen_resp_2_1D = 0
    prior_map = torch.zeros(10, 10)
    list_prior_map = []
    list_objective_mean_map = []

    # Get datas
    trainsC = Trains(clean_thresh=0.06)

    # 1D dataset and models
    # ---------------------------------------------
    # Create 1D dataset
    X_1D, Y_1D, Xmean_1D, Ymean_1D = make_dataset_1d(trainsC)
    # Get ground truth value, max value and possible pins coordinates for testing
    test_x_1D = torch.tensor(Xmean_1D)
    ground_truth_max_1D = np.max(Ymean_1D)
    print('GT max: ', ground_truth_max_1D)

    # 2D dataset
    # -------------------------------------------------
    # Create 2D dataset
    X_2D, Y_2D, Xmean_2D, Ymean_2D = make_dataset_2d(trainsC)
    print('Shape X', X_2D.shape)
    print('Shape Y', Y_2D.shape)
    print('Shape Xmean', Xmean_2D.shape)
    print('Shape Ymean', Ymean_2D.shape)

    ground_truth_max_2D = np.max(Ymean_2D)
    print('GT max: ', ground_truth_max_2D)
    trainsC.plot_response_matrix()

    test_x_2D = torch.tensor(Xmean_2D)
    # --------------------------------------------------

    # Prepare data storage
    # -------------------------------------------------------------------------
    (array_repetition_2D, array_query_2D, array_exploration_score_2D, array_exploitation_score_2D, array_explr_mean_2D,
     array_explt_mean_2D,
     array_confUp_explr_mean_2D, array_confUp_explt_mean_2D, array_confLow_explr_mean_2D, array_confLow_explt_mean_2D,
     array_pins_coord_2D, array_pins_coord_exploration_2D, array_pins_count_exploration_2D,
     array_pins_count_2D) = create_data_storage(nbr_query, nbr_repetition, nbr_pins)

    for kappa in [2, 3, 4, 5, 6]:

        print(f'\nKAPPA = {kappa}\n')
        # Running the training and evaluatino pipeline to get all the information on the model
        data_dict_Hmodel = run_training_and_eval(nbr_query, nbr_repetition, kappa)

        # Name format for saving data
        file_name_H = 'HGPBO_data_' + current_datetime + '_k' + str(f'{kappa:02}') + '_emg' + str(EMG) + 'iniPts' + str(f'{nbr_rdm_points_ini:02}')
        # Save data
        save_data(file_name_H, folder_of_the_day, workspace_folder, data_dict_Hmodel)
        # Reset data storage for next loop (new kappa or nbr_ini, etc)
        data_dict_Hmodel = reset_data_storage(data_dict_Hmodel)

    # Go to the folder of the day and plot data
    os.chdir(folder_of_the_day)
    for files in os.listdir(folder_of_the_day):
        print(files)
        if files != ("time_of_run.json"):
            if files.endswith(".json"):
                with open(files, 'r') as dict_json_file:
                    data_dict = json.load(dict_json_file)
                    vi.prior_map_visualization(data_dict, nbr_query, Xmean_1D, 2, 5, nbr_repetition)
    # Go back to workspace
    os.chdir(workspace_folder)

    vi.prior_map_visualization(data_dict_Hmodel, nbr_query, Xmean_1D, 2, 5, nbr_repetition)

    # Go to the folder of the day and plot data
    os.chdir(folder_of_the_day)

    for files in os.listdir(folder_of_the_day):
        print(files)
        if files != ("time_of_run.json"):
            if files.endswith(".json"):
                with open(files, 'r') as dict_json_file:
                    data_dict = json.load(dict_json_file)
                    vi.visualization_function(data_dict, files, data_dict)

    # Go back to workspace
    os.chdir(workspace_folder)

    # Go to the folder of the day and plot data
    os.chdir(folder_of_the_day)

    # Plot all exploration score
    vi.exploration_visualization_function(folder_of_the_day, nbr_query, nbr_rdm_points_data, nbr_repetition)

    # Go back to workspace
    os.chdir(workspace_folder)

    # Go to the folder of the day and plot data
    os.chdir(folder_of_the_day)

    # Plot all exploitation score
    vi.exploitation_visualization_function(folder_of_the_day, nbr_repetition)

    # Go back to workspace
    os.chdir(workspace_folder)