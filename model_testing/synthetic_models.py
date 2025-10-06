"""
This file will have the task of defining the models we will use in this project and the functions to modify/use them
"""
import torch
import sys
import os

from dataset_actions import *
import gpytorch
import time
import math
import visualization_information as vi
import hmodel_synthetic as hmodel

"""
Class definitions mainly GP models in this file
"""


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Simple GP classifier from the GPytorch library
    """

    def __init__(self, train_x, train_y, likelihood, nu=2.5, query_counter=None):
        """
        Initialize the Exact GP model.

        Parameters:
        - train_x (torch.Tensor): Training input data. If 1D [x, y], if 2D [x1, y1, x2, y2]
        - train_y (torch.Tensor): Training output data: EMG values
        - likelihood: Likelihood function.
        """
        if len(train_y) == 1:
            scaled_y = train_y/max(train_y)
        else:
            scaled_y = (train_y - torch.min(train_y)) / (torch.max(train_y) - torch.min(train_y))

        super(ExactGPModel, self).__init__(train_x, scaled_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu))
        self.query_counter = query_counter
        self.env_max_seen = torch.max(train_y)
        self.env_ind = list(range(0, len(train_x)))

        self.bif_max_seen = torch.tensor(-9999999, dtype=torch.float64)
        self.bif_ind = []

    def forward(self, x):
        """
        Forward pass of the GP model.

        Parameters:
        - x (torch.Tensor): Input data, can be train or test dataset

        Returns:
        - gpytorch.distributions.MultivariateNormal: Predictive distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def increment_q_n(model, query_c, query, domain):
        for x in range(len(domain)):
            if domain[x] == query:
                query_c[x] += 1
                break

        model.query_counter = query_c
        return query_c

    def update_max_seen_response_no_norm(self, next_query_value_random, max_seen_resp, env=False):
        if env:
            if next_query_value_random > self.env_max_seen:
                self.env_max_seen = next_query_value_random
        else:
            if next_query_value_random > self.bif_max_seen:
                self.bif_max_seen = next_query_value_random
        # next_query_value_random = next_query_value_random / max_seen_resp
        return next_query_value_random
    
    def update_training_data(self, train_x, train_y, next_query_pins, next_query_value, env=False):
        """
        Update the training data with the new query values.

        Parameters:
        - train_x (torch.Tensor): Training input data.
        - train_y (torch.Tensor): Training output data.
        - next_query_pins (torch.Tensor): Coordinates of the next query pins.
        - next_query_value (float): Value for the next query.

        Returns:
        - train_x (torch.Tensor): Updated training input data. (pin coordinates)
        - train_y (torch.Tensor): Updated training output data. (pin value)
        """
        """train_x = list(train_x)
        train_y = list(train_y)
        train_x.append(next_query_pins)
        train_y.append(next_query_value)
        train_x = torch.stack(train_x)
        train_y = torch.as_tensor(train_y)"""  # torch.as_tensor avoids copying the tensor in memory again, unlike torch.tensor
        if next_query_pins.ndim != train_x.ndim:
            next_query_pins = next_query_pins.unsqueeze(0)
        if next_query_value.ndim != train_y.ndim:
            next_query_value = next_query_value.unsqueeze(0)

        train_x = torch.cat((train_x, next_query_pins))
        train_y = torch.cat((train_y, next_query_value))
        
        next_q_val = len(self.env_ind) + len(self.bif_ind)
        if env:
            self.env_ind.append(next_q_val)
        else:
            self.bif_ind.append(next_q_val)

        return train_x, train_y


"""
Functions
"""


def get_next_query_pins(acquisition_map, coord_pins):
    """
    Determine the next query pins based on the acquisition map.
    This is for the hierarchical model!!!

    Parameters:
    - acquisition_map (torch.Tensor): Acquisition map.
    - coord_pins (numpy.ndarray): Coordinates of pins.

    Returns:
    - next_query_pins (torch.Tensor): Coordinates of the next query pins.
    """
    # get next query pin based on the acquisition map, max return the max value, argmax return indices of the max value
    possible_next_query = torch.where(acquisition_map.reshape(len(acquisition_map))==torch.max(acquisition_map.reshape(len(acquisition_map))))
    # randomly choose a query if there are multiple max values
    if len(possible_next_query[0])>1: # si plusieurs fois la valeur max, choisir random parmis ces valeurs max
        indice_next_query = np.random.randint(len(possible_next_query[0]))             # récupère l'indice de la next query aléatoirement parmis les indices offrant la max value
        next_query = possible_next_query[0][indice_next_query]                         # coordonnées x,y correspondant à la val max sélectionnée
    else:
        next_query = possible_next_query
    next_query_pins = torch.as_tensor(coord_pins[next_query], dtype=torch.float64)
    next_query_pins = torch.squeeze(next_query_pins)    # récupère les coord des pins et la valeur correpsondante
    return next_query_pins


def get_next_query_value(next_query_pins, X, Y, nbr_rdm_points_data=20, noise=0):
    """
    Get the new value for the next query pins to update training data.

    Parameters:
    - next_query_pins (torch.Tensor): Coordinates of the next query pins.
    - Y (list): List of values for the corresponding pins.
                Per pins, possible values =  nbr_rdm_points_data

    Returns:
    - new_query_value_random (float): Randomly chosen new value to add_kernel to the training dataset.
    - new_query_value_mean (float): Mean value of the corresponding pins to compute exploitation score.
    """
    # next_query_pins = next_query_pins.to(torch.int)
    new_training_values_tampon = np.zeros(nbr_rdm_points_data)
    reshape_y = torch.reshape(Y, (-1, 1))
    i = 0
    for indices, pins in enumerate(X):
        # find pins of ((x, y), (x, y)) coordinates in X
        if next_query_pins.ndim == 0:
            # This is for when we are trying to train the children
            if pins == next_query_pins:
                new_training_values_tampon[i] = reshape_y[indices]
                i += 1
        else:
            if pins[0] == next_query_pins[0] and pins[1] == next_query_pins[1]:
                new_training_values_tampon[i] = reshape_y[indices]
                i += 1

    # To deal with number of Y in the dataset that is variable in 2D dataset (always 20 in 1D dataset)
    # (most of the time is 10 in 2D dataset because they took 10 emg responses from monkeys)
    # but it can be 11 or 9. More elegant way is to use len(ys) in make_dataset function
    # but here it works by taking fixing the lenght of new_training_values_tampon to 11
    # and taking the real lenght of non zero elements, then using a new array
    len_non_zero = np.count_nonzero(new_training_values_tampon)
    new_training_values = np.zeros(len_non_zero)
    for x in range(len_non_zero):
        new_training_values[x] = new_training_values_tampon[x]

    new_query_value_mean = np.mean(new_training_values)
    new_query_value_random = np.random.choice(new_training_values)


    new_query_value_mean += np.random.normal(0, noise, size=new_query_value_mean.shape) * (np.max(reshape_y.numpy())-np.min(reshape_y.numpy()))
    new_query_value_random += np.random.normal(0, noise, size=new_query_value_random.shape) * (np.max(reshape_y.numpy())-np.min(reshape_y.numpy()))

    return new_query_value_random, new_query_value_mean


def optimize(model, likelihood, training_iter, train_x, train_y, verbose=True, hierarchical=False):
    """
    Optimize the GP model.

    Parameters:
    - model: GP model.
    - likelihood: Likelihood function.
    - training_iter (int): Number of optimization iterations.
    - train_x (torch.Tensor): Training input data. . If 1D [x, y], if 2D [x1, y1, x2, y2]
    - train_y (torch.Tensor): Training output data: EMG values
    - verbose (bool): Whether to print optimization progress.

    Returns:
    - model: Optimized GP model.
    - likelihood: Optimized likelihood function.
    """
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.sum().backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
    return model, likelihood


def get_acquisition_map(kappa, observed_pred, query_count):
    """
    Compute the acquisition map for Bayesian optimization.
    Here UCB function : a(x;k)=μ(x)+kσ(x)

    Parameters:
    - k: kappa (float): Manage the trade-off between exploration and exploitation.
    - observed_pred (gpytorch.distributions.MultivariateNormal): Predictive posterior distribution.

    Returns:
    - acquisition_map (torch.Tensor): Acquisition map.
    - y_mu (torch.Tensor): Mean of the predictive distribution.
    """
    # get the mean (mu) and the variance (sigma2) of the model
    y_mu = observed_pred.mean
    y_sigma2 = observed_pred.stddev

    # compute acquisition map
    acquisition_map = y_mu + kappa * torch.nan_to_num(y_sigma2/torch.sqrt(query_count))  # here UCB acquisition function
    # print(f'Average UCB Ratio mean/({kappa} * std): {torch.mean(y_mu/(kappa * y_sigma2))}')
    return acquisition_map, y_mu


def get_next_query_pins_from_1D(acquisition_map, coord_pins):
    """
    Determine the next query pins based on the acquisition map.

    Parameters:
    - acquisition_map (torch.Tensor): Acquisition map.
    - coord_pins (numpy.ndarray): Coordinates of pins.

    Returns:
    - next_query_pins (torch.Tensor): Coordinates of the next query pins.
    """
    # get next query pin based on the acquisition map, max return the max value, argmax return indices of the max value
    possible_next_query = torch.topk(acquisition_map, 2)  # Gets the top 2 candidates for acquisition
    indice1 = possible_next_query.indices[0]
    indice2 = possible_next_query.indices[1]
    next_query_pins1 = torch.as_tensor(indice1, dtype=torch.float64)
    next_query_pins2 = torch.as_tensor(indice2, dtype=torch.float64)
    next_query_pins = torch.cat((next_query_pins1, next_query_pins2), 0)
    return next_query_pins


def get_next_query_value_1D(next_query_pins, X, Y, nbr_rdm_points_data=20):
    """
    Get the new value for the next query pins to update training data.

    Parameters:
    - next_query_pins (torch.Tensor): Coordinates of the next query pins.
    - Y (list): List of values for the corresponding pins.
                Per pins, possible values =  nbr_rdm_points_data

    Returns:
    - new_query_value_random (float): Randomly chosen new value to add_kernel to the training dataset.
    - new_query_value_mean (float): Mean value of the corresponding pins to compute exploitation score.
    """
    new_training_values = np.zeros(nbr_rdm_points_data)
    i = 0
    for indices, pins in enumerate(X):
        if pins[0] == next_query_pins[0] and pins[1] == next_query_pins[1]:  # find pins of (x, y) coordinates in X
            new_training_values[i] = Y[indices]
            i += 1
    new_query_value_mean = np.mean(new_training_values)
    new_query_value_random = np.random.choice(new_training_values)
    return new_query_value_random, new_query_value_mean


def make_prediction(model, x, likelihood):
    """
    Make a prediction on the next query.

    Parameters:
    - model: GP model.
    - x (torch.Tensor): Input data. (pins coordinates)

    Returns:
    - observed_pred (gpytorch.distributions.MultivariateNormal): Predicted distribution.
    """
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x))
        return observed_pred


def get_confidence_exploration(tensor_lower, tensor_upper, pins_exploration, test_coord):
    """
    Get confidence bounds for exploration.

    Parameters:
    - tensor_lower (torch.Tensor): Lower confidence bounds tensor. (from make_prediction)
    - tensor_upper (torch.Tensor): Upper confidence bounds tensor.
    - pins_exploration (torch.Tensor): Coordinates of the pins for exploration.
    - test_coord: Array of test coordinates.

    Returns:
    - conf_low (torch.Tensor): Lower confidence bound for the specified exploration pins.
    - conf_up (torch.Tensor): Upper confidence bound for the specified exploration pins.
    """
    for indices, pins in enumerate(test_coord):
        if pins_exploration[0] == pins[0] and pins_exploration[1] == pins[1]:  # find pins of (x, y) coordinates in X
            conf_low = tensor_lower[indices]
            conf_up = tensor_upper[indices]
    return conf_low, conf_up


def get_confidence_exploitation(tensor_lower, tensor_upper, pins_exploitation, test_coord):
    """
    Get confidence bounds for exploitation.

    Parameters:
    - tensor_lower (torch.Tensor): Lower confidence bounds tensor.
    - tensor_upper (torch.Tensor): Upper confidence bounds tensor.
    - pins_exploitation (torch.Tensor): Coordinates of the pins for exploitation.
    - test_coord: Array of test coordinates.

    Returns:
    - conf_low (torch.Tensor): Lower confidence bound for the specified exploitation pins.
    - conf_up (torch.Tensor): Upper confidence bound for the specified exploitation pins.
    """
    for indices, pins in enumerate(test_coord):
        if pins_exploitation[0] == pins[0] and pins_exploitation[1] == pins[1]:  # find pins of (x, y) coordinates in X
            conf_low = tensor_lower[indices]
            conf_up = tensor_upper[indices]
    return conf_low, conf_up


def get_exploration_score(y_mu, ground_truth_max, coord_pins, X, Y, nbr_rdm_points_data=20):
    """
    Compute the exploration score.

    Parameters:
    - y_mu (torch.Tensor): Mean of the predictive distribution.
    - ground_truth_max (float): Maximum ground truth value.
    - coord_pins (numpy.ndarray): Coordinates of pins.
    - Y (list): List of values for the corresponding pins.

    Returns:
    - exploration_score (float): Exploration score.
    - next_query_pins (torch.Tensor): Coordinates of the next query pins to explore.

    Comments: X and Y represent the coord and respective values of GT, size of nbr_rdm_points_data
    """
    new_training_values_tampon = []
    mu = y_mu
    argmax_mu = torch.where(mu.reshape(len(mu)) == torch.max(mu.reshape(len(mu))))

    # randomly choose a query if there are multiple max values
    if len(argmax_mu[0]) > 1:  # si plusieurs fois la valeur max, choisir random parmis ces valeurs max
        indice_next_query = np.random.randint(len(
            argmax_mu[0]))  # récupère l'indice de la next query aléatoirement parmis les indices offrant la max value
        next_query = argmax_mu[0][indice_next_query]  # coordonnées x,y correspondant à la val max sélectionnée
        next_query_pins = torch.as_tensor(coord_pins[next_query],
                                          dtype=torch.float64)  # récupère les coord des pins et la valeur correpsondante
    else:
        next_query = argmax_mu[0][0]
        next_query_pins = torch.as_tensor(coord_pins[next_query], dtype=torch.float64)

    for indices_x, pins in enumerate(X):
        # find pins of ((x, y), (x, y)) coordinates in X
        for indices_y, sub_pin in enumerate(pins):
            if sub_pin[0] == next_query_pins[0] and sub_pin[1] == next_query_pins[1]:
                new_training_values_tampon.append(Y[indices_x, indices_y])

    # To deal with number of Y in the dataset that is variable in 2D dataset (always 20 in 1D dataset)
    # (most of the time is 10 in 2D dataset because they took 10 emg responses from monkeys)
    # but it can be 11 or 9. More elegant way is to use len(ys) in make_dataset function
    # but here it works by taking fixing the length of new_training_values_tampon to 11
    # and taking the real length of non zero elements, then using a new array
    new_training_values_tampon = np.array(new_training_values_tampon)
    mean_value = np.mean(new_training_values_tampon)
    max_Y = torch.max(Y)
    min_Y = torch.min(Y)/max_Y
    exploration_score = mean_value / max_Y
    resized_explor = (exploration_score - min_Y)/(1 - min_Y)
    return resized_explor, next_query_pins


def get_exploitation_score(mean_value, ground_truth_max):
    """
    Compute the exploitation score.

    Parameters:
    - mean_value (float): Mean value of the corresponding selected pin.
    - ground_truth_max (float): Maximum ground truth value.

    Returns:
    - exploitation_score (float): Exploitation score.
    """
    exploitation_score = mean_value / ground_truth_max
    return exploitation_score


def train_model1D(model, likelihood, training_iter, train_x, train_y, test_x, GT_max, kappa, nbr_repetition, nbr_query,
                  nbr_rdm_points_ini, nbr_pins, emg, trainsC, Y):
    # Prepare data storage
    # -------------------------------------------------------------------------
    # Array for saving data of each query
    array_repetition = np.zeros(nbr_query * nbr_repetition)
    array_query = np.zeros(nbr_query * nbr_repetition)
    # Array for the exploitation and exploration scores
    array_exploration_score = np.zeros(nbr_query * nbr_repetition)
    array_exploitation_score = np.zeros(nbr_query * nbr_repetition)
    # Array for the mean of the #query over all repetitions
    array_explr_mean = np.zeros(nbr_query)
    array_explt_mean = np.zeros(nbr_query)
    # Array for the mean confidence low and up of each score of the #query
    array_confUp_explr_mean = np.zeros(nbr_query)
    array_confUp_explt_mean = np.zeros(nbr_query)
    array_confLow_explr_mean = np.zeros(nbr_query)
    array_confLow_explt_mean = np.zeros(nbr_query)
    # Array for exploration and exploitation pins selected
    array_pins_coord = np.zeros(((nbr_query * nbr_repetition), 2))
    array_pins_coord_exploration = np.zeros(((nbr_query * nbr_repetition), 2))
    # Array for counting each exploration and exploitation pins selected
    array_pins_count_exploration = np.zeros(nbr_pins)
    array_pins_count = np.zeros(nbr_pins)
    # -------------------------------------------------------------------------

    # Start running the entire process for nbr_repetition
    for repetition in range(nbr_repetition):
        startTime_repetition = time.time()
        max_seen_resp = 0

        # Start training and evaluating over a number of query
        for q in range(nbr_query):
            if q == 0:  # If first query
                # Get into evaluation (predictive posterior) mode
                model.eval()
                likelihood.eval()

                # Make a prediction, observed_pred = likelihood, prediction_mean = mu
                observed_pred = make_prediction(model, test_x, likelihood)

            acquisition_map, y_mu = get_acquisition_map(kappa, observed_pred)

            # Get the coordinates of the next pins to exploite
            next_query_pins = get_next_query_pins(acquisition_map, test_x)

            # Get the 20 values in the GT dataset corresponding to the next pin
            # Random is for update training data, random value choose between the 20 values of the GT
            # Mean is for the exploitation score computation, mean of the 20 values of the GT
            next_query_value_random, next_query_value_mean = get_next_query_value(next_query_pins, Y)

            # Update the max seen response if necessary
            if (next_query_value_random > max_seen_resp) or (max_seen_resp == 0):
                max_seen_resp = next_query_value_random
            next_query_value_random = next_query_value_random / max_seen_resp

            # Update training data by adding next_query_value to the train dataset
            train_x, train_y = update_training_data(train_x, train_y, next_query_pins, next_query_value_random)

            # Update the model with the new training data
            model.set_train_data(train_x, train_y, strict=False)  # strict = False to add_kernel inputs with different shape

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            model, likelihood = optimize(model, likelihood, training_iter, train_x, train_y)

            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            # Make a prediction, observed_pred = likelihood, prediction_mean = y_mu
            observed_pred = make_prediction(model, test_x, likelihood)

            # -----------------------------------------------------------------------------------------------#
            # Compute exploration and exploitation score
            exploration_score, next_query_pins_exploration = get_exploration_score(y_mu, GT_max, test_x, Y)
            exploitation_score = get_exploitation_score(next_query_value_mean, GT_max)
            # Record results and parameters
            # Query and repetitions
            array_repetition[q + nbr_query * repetition] = repetition
            array_query[q + nbr_query * repetition] = q
            # Scores and confidence
            array_exploration_score[q + nbr_query * repetition] = exploration_score
            array_exploitation_score[q + nbr_query * repetition] = exploitation_score
            # Pins selected
            array_pins_coord[q + nbr_query * repetition][:] = next_query_pins
            array_pins_coord_exploration[q + nbr_query * repetition][:] = next_query_pins_exploration
            # Mean for each score and confidence level of a query over all rep
            array_explr_mean[q] += (array_exploration_score[q + nbr_query * repetition])
            array_explt_mean[q] += (array_exploitation_score[q + nbr_query * repetition])
            # -----------------------------------------------------------------------------------------------#

    executionTime_repetitions = (time.time() - startTime_repetition)

    # Compute the mean of score and confidence level on each line which accumulated value over repetitions
    for q in range(nbr_query):
        array_explr_mean[q] = array_explr_mean[q] / nbr_repetition
        array_explt_mean[q] = array_explt_mean[q] / nbr_repetition
        array_confUp_explr_mean[q] = array_explr_mean[q] + (
                    np.std(array_explr_mean, axis=0) / nbr_repetition) / math.sqrt(
            nbr_repetition)  # mean +/- (SD/sqrt(query))
        array_confUp_explt_mean[q] = array_explt_mean[q] + (
                    np.std(array_explt_mean, axis=0) / nbr_repetition) / math.sqrt(nbr_repetition)
        array_confLow_explr_mean[q] = array_explr_mean[q] - (
                    np.std(array_explr_mean, axis=0) / nbr_repetition) / math.sqrt(nbr_repetition)
        array_confLow_explt_mean[q] = array_explt_mean[q] - (
                    np.std(array_explt_mean, axis=0) / nbr_repetition) / math.sqrt(nbr_repetition)

    array_pins_count = count_pin(array_pins_coord, test_x)
    array_pins_count_exploration = count_pin(array_pins_coord_exploration, test_x)

    data_dict = {'query_number': array_query,
                 'exploration_score': array_exploration_score,
                 'exploitation_score': array_exploitation_score,
                 'pins_coord_exploration': array_pins_coord_exploration,
                 'pins_count_exploration': array_pins_count_exploration,
                 'pins_coord_exploitation': array_pins_coord,
                 'pins_count_exploitation': array_pins_count,
                 'exploration_score_mean_repetition': array_explr_mean,
                 'exploitation_score_mean_repetition': array_explt_mean,
                 'kappa_value': kappa,
                 'nbr_of_query_per_rep': nbr_query,
                 'nbr_of_repetition': nbr_repetition,
                 'pts_of_ini': nbr_rdm_points_ini
                 }

    return model, train_x, train_y, max_seen_resp, data_dict, executionTime_repetitions


def update_max_seen_response(next_query_value_random, max_seen_resp):
    if (next_query_value_random > max_seen_resp) or (max_seen_resp == 0):
        max_seen_resp = next_query_value_random
    next_query_value_random = next_query_value_random / max_seen_resp
    return next_query_value_random, max_seen_resp

def update_max_seen_response_no_norm(next_query_value_random, max_seen_resp):
    if (next_query_value_random > max_seen_resp) or (max_seen_resp == 0):
        max_seen_resp = next_query_value_random
    # next_query_value_random = next_query_value_random / max_seen_resp
    return next_query_value_random, max_seen_resp

def startup_children(child, child_like, train_x_child, train_y_child, x_child, y_child, child_qc, nbr_query, training_iter, noise, kappa):
    over = []
    for q in range(nbr_query):
        if q == 0:
            max_seen_resp = torch.max(train_y_child)

            optimizer = torch.optim.Adam(child.parameters(), lr=1e-3)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(child_like, child)

            child.eval()
            child_like.eval()

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            observed_pred = make_prediction(child, x_child, child_like)

        acquisition_map, hierar_y_mu = get_acquisition_map(kappa, observed_pred, child_qc)

        next_query_pins = get_next_query_pins(acquisition_map, x_child)

        next_query_value_random, next_query_value_mean = get_next_query_value(next_query_pins,
                                                                                     x_child,
                                                                                     y_child, noise=noise)

        next_query_value_random, max_seen_resp = update_max_seen_response_no_norm(next_query_value_random,
                                                                                    max_seen_resp)

        response = torch.tensor(next_query_value_random)

        train_x_child, train_y_child = child.update_training_data(train_x_child, train_y_child, next_query_pins, response,
                                                      env=True)  # has to be 1D dataset
        # Update the model with the new training data

        div_y = (train_y_child - torch.min(y_child))/(torch.max(y_child) - torch.min(y_child))

        child.set_train_data(train_x_child, div_y, strict=False)

        """
        train_x_child, train_x_sub2 = train_x_hier[:, 0], train_x_hier[:, 1]"""

        child.train()
        child_like.train()
        for i in range(training_iter):
            # Find optimal model hyperparameters
            optimizer.zero_grad()

            output = child(train_x_child)

            loss = -mll(output, train_y_child)

            loss.backward()
            optimizer.step()
            # Get into evaluation (predictive posterior) mode
        child.eval()
        child_like.eval()
        with gpytorch.settings.lazily_evaluate_kernels(state=False):

            c1_r2 = vi.child_contour_r2([child], x_child,
                                        [y_child / torch.max(y_child)])
        over.append(c1_r2[0])
    print(over)
    return child, child_like, train_x_child, train_y_child, child_qc

def train_submodels(child, child_like, train_x_child, train_y_child, x_child, y_child, child_qc, nbr_query, training_iter, noise, kappa):
    for q in range(nbr_query):
        if q == 0:
            # Need to initialize the model - Will be random in this method

            max_seen_response = torch.max(train_y_child)
            child.eval()
            child_like.eval()

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            observed_pred = make_prediction(child, x_child, child_like)

        acquisition_map, y_mu = get_acquisition_map(kappa, observed_pred, child_qc)

        next_query = torch.argmax(acquisition_map)

        q_x, q_y = x_child[next_query], y_child[next_query]

        q_y += (torch.max(y_child)-torch.min(y_child)) * np.random.normal(0, noise)

        response, max_seen_response = update_max_seen_response_no_norm(q_y, max_seen_response)
        child_qc = child.increment_q_n(child_qc, q_x, x_child)
        child, child_like, train_x_child, train_y_child = hmodel.update_model1_1D_max_seen(child, child_like, train_x_child, train_y_child, q_x, response, env=True, training_iter=training_iter)
        child.eval()
        child_like.eval()
    return child, child_like, train_x_child, train_y_child, child_qc
name_code = 'HGP_BO-test6-priorMAP-1model1D'
