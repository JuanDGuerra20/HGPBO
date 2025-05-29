import torch
import matplotlib.pyplot as plt
from torch import nn
from dataset_actions import *
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from seaborn import heatmap
import warnings
from scipy.stats import linregress



class HashTable:

    # Create empty bucket list of given size
    def __init__(self, size):
        self.size = size
        self.hash_table = self.create_buckets()

    def create_buckets(self):
        return [[] for _ in range(self.size)]

    # Insert values into hash map
    def set_val(self, key, val):

        # Get the index from the key
        # using hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key to be inserted
            if record_key[0] == key[0] and record_key[1] == key[1] and record_key[2] == key[2]:
                found_key = True
                break

        # If the bucket has same key as the key to be inserted,
        # Update the key value
        # Otherwise append the new key-value pair to the bucket
        if found_key:
            bucket[index] = (key, val)
        else:
            bucket.append((key, val))

    # Return searched value with specific key
    def get_val(self, key):

        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key being searched
            if record_key[0] == key[0] and record_key[1] == key[1]:
                found_key = True
                break

        # If the bucket has same key as the key being searched,
        # Return the value found
        # Otherwise indicate there was no record found
        if found_key:
            return record_val
        else:
            return "No record found"

    # Remove a value with specific key
    def delete_val(self, key):

        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key to be deleted
            if record_key == key:
                found_key = True
                break
        if found_key:
            bucket.pop(index)
        return

    def __str__(self):
        return "".join(str(item) for item in self.hash_table)


class NN_baseline(nn.Module):

    def __init__(self, dims, test_x, device="cpu"):
        super().__init__()

        self.device = device
        self.hash_map = HashTable(np.prod(test_x.shape[:-1]))
        for i in range(len(test_x)):
            one_hot = torch.zeros(len(test_x))
            one_hot[i] = 1
            self.hash_map.set_val(str(test_x[i]), one_hot)
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential()

        for i in range(1, len(dims)):
            self.linear_stack.append(nn.Linear(dims[i-1], dims[i]))
            self.linear_stack.append(nn.ReLU())
        self.linear_stack.append(nn.Linear(dims[-1], 1))

    def forward(self, x):
        mapped_x = []
        for ind_x in x:
            mapped_x.append(self.hash_map.get_val(str(ind_x)))
            if len(mapped_x[-1]) != 100:
                print(ind_x.dtype)
                print(ind_x)
                print(mapped_x[-1])
        flattened_x = self.flatten(torch.tensor(np.array(mapped_x), device=self.device))
        logits = self.linear_stack(flattened_x)
        return logits

    def get_acquisition_map(self, test_x_hier):
        predictions = self.forward(test_x_hier)

        return predictions


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
    next_query_pins = torch.as_tensor(coord_pins[next_query], dtype=torch.float)
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
    new_training_values_tampon = torch.zeros(nbr_rdm_points_data)
    reshape_y = torch.reshape(Y, (-1, 1))
    i = 0
    for indices, pins in enumerate(X):
        # find pins of ((x, y), (x, y)) coordinates in X
        if pins[0] == next_query_pins[0] and pins[1] == next_query_pins[1]:
            new_training_values_tampon[i] = reshape_y[indices]
            i += 1

    # To deal with number of Y in the dataset that is variable in 2D dataset (always 20 in 1D dataset)
    # (most of the time is 10 in 2D dataset because they took 10 emg responses from monkeys)
    # but it can be 11 or 9. More elegant way is to use len(ys) in make_dataset function
    # but here it works by taking fixing the lenght of new_training_values_tampon to 11
    # and taking the real lenght of non zero elements, then using a new array
    len_non_zero = np.count_nonzero(new_training_values_tampon)
    new_training_values = torch.zeros(len_non_zero)
    for x in range(len_non_zero):
        new_training_values[x] = new_training_values_tampon[x]

    new_query_value_mean = torch.mean(new_training_values)
    new_query_value_random = np.random.choice(new_training_values)

    new_query_value_mean += torch.normal(0, noise*(torch.max(reshape_y)-torch.min(reshape_y)), size=new_query_value_mean.shape)
    new_query_value_random += torch.normal(0, noise*(torch.max(reshape_y)-torch.min(reshape_y)), size=new_query_value_random.shape)

    return torch.tensor(new_query_value_random, dtype=torch.float), torch.tensor(new_query_value_mean, dtype=torch.float)

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
    new_training_values_tampon = np.zeros(nbr_rdm_points_data)
    i = 0
    mu = y_mu
    argmax_mu = torch.where(mu.reshape(len(mu)) == torch.max(mu.reshape(len(mu))))

    # randomly choose a query if there are multiple max values
    if len(argmax_mu[0]) > 1:  # si plusieurs fois la valeur max, choisir random parmis ces valeurs max
        indice_next_query = np.random.randint(len(
            argmax_mu[0]))  # récupère l'indice de la next query aléatoirement parmis les indices offrant la max value
        next_query = argmax_mu[0][indice_next_query]  # coordonnées x,y correspondant à la val max sélectionnée
        next_query_pins = torch.as_tensor(coord_pins[next_query],
                                          dtype=torch.float)  # récupère les coord des pins et la valeur correpsondante
    else:
        next_query = argmax_mu[0][0]
        next_query_pins = torch.as_tensor(coord_pins[next_query], dtype=torch.float)

    for indices_x, pins in enumerate(X):
        # find pins of ((x, y), (x, y)) coordinates in X
        for indices_y, sub_pin in enumerate(pins):
            if sub_pin[0] == next_query_pins[0] and sub_pin[1] == next_query_pins[1]:
                new_training_values_tampon[i] = Y[indices_x, indices_y]
                i += 1

    # To deal with number of Y in the dataset that is variable in 2D dataset (always 20 in 1D dataset)
    # (most of the time is 10 in 2D dataset because they took 10 emg responses from monkeys)
    # but it can be 11 or 9. More elegant way is to use len(ys) in make_dataset function
    # but here it works by taking fixing the length of new_training_values_tampon to 11
    # and taking the real length of non zero elements, then using a new array
    len_non_zero = np.count_nonzero(new_training_values_tampon)
    new_training_values = np.zeros(len_non_zero)
    for x in range(len_non_zero):
        new_training_values[x] = new_training_values_tampon[x]

    mean_value = np.mean(new_training_values)
    exploration_score = mean_value / ground_truth_max
    return exploration_score, next_query_pins

def heatmap_r_score(data, z):
    data_avg = np.mean(data, axis=0)

    r_scores = []

    z = z.reshape(data_avg[0].shape)
    r_over = []
    """for d in range(len(data)):
        r_scores_list = []
        for i in range(len(data[d])):
            r_scores.append((linregress(z, data_avg[d][i]).rvalue) ** 2)
            r_scores_list.append(r_scores)
        r_over.append(r_scores_list)

    r_std = np.std(r_over, axis=0)"""

    for q in range(len(data_avg)):
        r_scores.append((linregress(z, data_avg[q]).rvalue)**2)

    return r_scores