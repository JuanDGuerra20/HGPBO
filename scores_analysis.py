import numpy as np
import torch


def get_exploration_score_model_1D(y_mu, ground_truth_max, coord_pins, X, Y, nbr_rdm_points_data=20):
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
    values_for_mean = np.zeros(nbr_rdm_points_data)
    i = 0
    mu = y_mu
    argmax_mu = torch.where(mu.reshape(len(mu)) == torch.max(mu.reshape(len(mu))))

    # randomly choose a query if there are multiple max values
    if len(argmax_mu[0]) > 1:  # If there are several max values, choose randomly among these max values
        indice_next_query = np.random.randint(len(
            argmax_mu[0]))  # Retrieves the index of the next query randomly among the indices offering the max value
        next_query = argmax_mu[0][indice_next_query]  # Coordinates (x, y) corresponding to the selected max value
        next_query_pins = torch.as_tensor(coord_pins[next_query],
                                          dtype=torch.float64)  # Retrieves the coordinates of the pins and the corresponding value

    else:
        next_query = argmax_mu[0][0]
        next_query_pins = torch.as_tensor(coord_pins[next_query], dtype=torch.float64)

    for indices, pins in enumerate(X):
        if pins[0] == next_query_pins[0] and pins[1] == next_query_pins[1]:  # find pins of (x, y) coordinates in X
            values_for_mean[i] = Y[indices]
            i += 1

    mean_value = np.mean(values_for_mean)
    exploration_score = mean_value / ground_truth_max
    return exploration_score, next_query_pins


def get_exploitation_score_model_1D(mean_value, ground_truth_max):
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
