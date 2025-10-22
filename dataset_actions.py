import os
import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import torch
import json
from json import JSONEncoder
from collections import Counter
import math
import seaborn as sns

"""
Global Variables
Look into the possibility of passing these as inputs to functions if they are used across many files
could make the code more modular/better
"""

CHS = [13, 17, 21, 18, 22, 2, 6, 10, 14, 9]
DTS = [0, 10, 20, 40, 60, 80, 100]
EMG = 4
DT = 60
N_EMGS = 7
xy2ch = [[2, 6, 10, 14, 9],
         [13, 17, 21, 18, 22]]
ch2xy = {}
for ch in CHS:
    x, y = np.where(np.array(xy2ch) == ch)
    ch2xy[ch] = [x[0], y[0]]
nbr_rdm_points_ini = 5


class Trains:
    """
    Class to train the models and setup the environment of stimulations and everything
    Helps create the dataset
    """

    def __init__(self, path_to_data=None, clean_thresh=None, verbose=False):
        """
        :param emg: What does this mean?
        :param N_EMGS: what dose this mean?
        :param CHS: the channel numbers that are used
        :param DTS: time delay between the stimulations
        :param path_to_data: self-explanatory
        :param clean_thresh:
        :param verbose: self-explanatory
        """
        self.chs = CHS
        self.n_ch = len(self.chs)
        self.dts = DTS

        # conditions are
        # 0 : seulement channel A
        # 1: channel A, delay 40ms, channel B
        # 2: channel A, delay 20ms, channel B
        # 3: channel A, delay 10ms, channel B
        # 4: channel A et channel B simultaneous
        # 5: channel B, delay 10ms, channel A
        # 6: channel B, delay 20ms, channel A
        # 7: channel B, delay 40ms, channel A
        # 8:  seulement channel B
        # each cell contains a 20x733 matrix (20 stimulations, 733 time series
        # emg response)
        if path_to_data:
            filtmat = loadmat(os.path.join(path_to_data, 'FilteredPairedTrains.mat'))
        else:
            filtmat = loadmat('FilteredPairedTrains.mat')
        filtdata = filtmat['gfilt_resp']

        # Let's build the datastructure
        # trains[chi][chj][deltat] will contain all timeseries for pair chi, chj
        # with time delay delta_t
        # trains[ch_i][ch_i] will contain single channel pulses
        self.N_EMGS = N_EMGS
        self.emgdct = {}
        for emg in range(N_EMGS):
            # Pairs
            dct = {ch1: {ch2: {} for ch2 in self.chs} for ch1 in self.chs}
            for i, ch1 in enumerate(self.chs):
                for j, ch2 in enumerate(self.chs):
                    # We deal with dt=0 separately, since it should be
                    # symmetric! (ch1,ch2 = ch2,ch1 since stimulations
                    # are done simultaneously (dt=0))
                    if i == j:
                        dct[ch1][ch1][0] = {'data': filtdata[emg, i, j, 0]}
                    else:
                        data = filtdata[emg, i, j, 0]
                        if data.size == 0:
                            data = filtdata[emg, j, i, 0]
                        dct[ch1][ch2][0] = dct[ch2][ch1][0] = {'data': data}
                    ch = dct[ch1][ch2][0]['data']
                    maxs = ch.max(axis=1)
                    dct[ch1][ch2][0]['maxs'] = dct[ch2][ch1][0]['maxs'] = maxs
                    dct[ch1][ch2][0]['meanmax'] = dct[ch2][ch1][0]['meanmax'] = maxs.mean()
                    dct[ch1][ch2][0]['stdmax'] = dct[ch2][ch1][0]['stdmax'] = maxs.std()
                    # Then we deal with dt!=0
                    for k, dt in enumerate(self.dts[1:], 1):
                        dct[ch1][ch2][dt] = {'data': filtdata[emg, i, j, k]}
                        # We also precompute the meanmax and stdmax
                        # statistics
                        ch = dct[ch1][ch2][dt]['data']
                        if ch.size != 0:
                            maxs = ch.max(axis=1)
                            dct[ch1][ch2][dt]['maxs'] = maxs
                            dct[ch1][ch2][dt]['meanmax'] = maxs.mean()
                            dct[ch1][ch2][dt]['stdmax'] = maxs.std()
            self.emgdct[emg] = dct

        if clean_thresh:
            # we remove all resps whose max is > clean_thresh
            count = 0
            for emg in range(N_EMGS):
                for ch1 in self.chs:
                    for ch2 in self.chs:
                        for dt in self.dts:
                            if (dt == 10 or dt == 20) and ch1 == ch2:
                                continue
                            maxs = self.emgdct[emg][ch1][ch2][dt]['maxs']
                            i = 0
                            while i < len(maxs):
                                if maxs[i] > clean_thresh:
                                    if verbose:
                                        print("Removing resp with max {}".format(maxs[i]))
                                        print("emg {}, ch1 {}, ch2 {}, dt {}".format(emg, ch1, ch2, dt))
                                    count += 1
                                    for other_emg in range(N_EMGS):
                                        data = self.emgdct[other_emg][ch1][ch2][dt]['data']
                                        data = np.delete(data, i, 0)
                                        self.emgdct[other_emg][ch1][ch2][dt]['data'] = data
                                        maxs = data.max(axis=1)
                                        self.emgdct[other_emg][ch1][ch2][dt]['maxs'] = maxs
                                        self.emgdct[other_emg][ch1][ch2][dt]['meanmax'] = maxs.mean()
                                        self.emgdct[other_emg][ch1][ch2][dt]['stdmax'] = maxs.std()
                                i += 1
            if verbose:
                print("Removed {} resps in total from dataset".format(count))
        self.trains = self.emgdct[emg]

    def get_emgdct(self, emg):
        return self.emgdct[emg]

    def plot_response_matrix(self, emg=EMG, syn=None, dt=DT, tau=40, title=True, plot_green=True, plot_red=True):
        if syn is None:
            syn = (emg, None)
            emg1 = emg2 = emg
        else:
            emg1, emg2 = syn
        fig = plt.figure()
        if title:
            plt.suptitle("Matrix of responses for EMG1={},EMG2={} with dt={} (1stch left, 2ndch top)".format(*syn, dt))
        gs = gridspec.GridSpec(12, 12)
        # We first plot the 1d responses on left and top
        for i, ch in enumerate(self.chs):
            # top
            ax = plt.subplot(gs[0, i + 2])
            ax.set_ylim([0, 0.05])
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title(ch2xy[ch])
            ax.plot(self.emgdct[emg2][ch][ch][0]['data'].T)
            ax.text(0, 0, "{:.2}".format(self.emgdct[emg2][ch][ch][0]['meanmax']))

            # channel labels top
            ax = plt.subplot(gs[1, i + 2])
            ax.text(0, 0.25, ch2xy[ch])
            ax.axis('off')

            # left side
            ax = plt.subplot(gs[i + 2, 0])
            ax.set_ylim([0, 0.05])
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_ylabel(ch2xy[ch])
            ax.plot(self.emgdct[emg1][ch][ch][0]['data'].T)
            ax.text(0, 0, "{:.2}".format(self.emgdct[emg1][ch][ch][0]['meanmax']))

            # channel labels left
            ax = plt.subplot(gs[i + 2, 1])
            ax.text(0.25, 0.25, ch2xy[ch])
            ax.axis('off')

        maxch1, maxch2 = self.max_ch_2d(syn=syn, dt=dt)
        maxr = self.synergy_meanmax(*syn, maxch1, maxch2, dt, tau=tau)
        for i, ch1 in enumerate(self.chs):
            for j, ch2 in enumerate(self.chs):
                ax = plt.subplot(gs[i + 2, j + 2])
                ax.set_ylim([0, 0.05])
                ax.set_xticks([])
                ax.set_yticks([])
                bbox = None
                data = self.synergy(*syn, ch1, ch2, dt, tau=tau)
                if data.size != 0:
                    plt.plot(data.T)
                mm = float(self.synergy_meanmax(*syn, ch1, ch2, dt, tau=tau))
                # if ch1==maxch1 and ch2==maxch2:
                #     bbox = dict(facecolor='green', alpha=0.5)
                if plot_green and mm > maxr - 0.002:
                    bbox = dict(facecolor='green', )
                elif plot_red and mm > maxr - 0.005 and mm < maxr - 0.002:
                    bbox = dict(facecolor='red', alpha=0.5)
                plt.text(0, 0, "{:.2}".format(mm), bbox=bbox)
        plt.show()
        plt.close()

    def max_ch_2d(self, emg=2, dt=DT, syn=None):
        if syn is None:
            syn = (emg, None)
        grid = self.build_f_grid(syn=syn, dt=dt)
        x, y = np.unravel_index(grid.argmax(), grid.shape)
        return [self.chs[x], self.chs[y]]

    def build_f_grid(self, emg=2, syn=None, dt=DT, f='meanmax'):
        if syn is None:
            syn = (emg, None)
        z_grid = np.zeros((self.n_ch, self.n_ch))
        for i, ch1 in enumerate(self.chs):
            for j, ch2 in enumerate(self.chs):
                # TODO: now with synergies we can only do f='meanmax'
                #      maybe changing this later if needed
                z_grid[i][j] = self.synergy_meanmax(*syn, ch1, ch2, dt)
                # self.emgdct[emg][ch1][ch2][dt][f]
        return z_grid

    def get_resp(self, emg, dt, ch1, ch2, filldiag=True):
        return self.synergy(emg, emg, ch1, ch2, dt, b=0, filldiag=filldiag)

    def synergy(self, emg1, emg2, ch1, ch2, dt=0, tau=40, a=1, b=1, filldiag=True):
        # synergy is just a linear combination of emgs
        # We use these to define a new cost function (max synergy
        # instead of max of a particular channel)
        # dt is dt between stim pulses
        # tau is how much we shift resp2
        if emg2 is None:
            emg2 = 0
            b = 0
        if filldiag and ch1 == ch2 and (dt == 10 or dt == 20):
            # We don't have values on the diag for dt=10 and dt=20,
            # so we get them from dt=0 if filldiag is True
            dt = 0
        resps1 = self.emgdct[emg1][ch1][ch2][dt]['data']
        resps2 = self.emgdct[emg2][ch1][ch2][dt]['data']
        # resps are 1466 (resps1.shape[1]) ticks ts, which last 300ms
        dtidx = int(tau / 300 * resps1.shape[1]) + 1
        resps2_shifted = np.zeros_like(resps2)
        resps2_shifted[:, :-dtidx] = resps2[:, dtidx:]
        return a * resps1 + b * resps2_shifted

    def synergy_meanmax(self, emg1, emg2, ch1, ch2, dt=0, tau=40, a=1, b=1):
        syn = self.synergy(emg1, emg2, ch1, ch2, dt, tau, a, b)
        if syn.size == 0:
            return 0
        else:
            return syn.max(axis=1).mean()


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def make_dataset_1d(trainsC, emg=EMG, n=20):
    """
    Makes a 1D dataset from the stimulations based on n random samples of the channel
    :param trainsC: the class for creating datasets
    :param emg: specific EMG signal we want
    :param n: number of random samples per channel we want to generate
    :return: dataset of x,y, avg x, and avg y
    """
    # Get the training data for the specified EMG signal.
    trains = trainsC.get_emgdct(emg=emg)

    X = []  # List to store input data points (channel coordinates).
    Y = []  # List to store output data points (EMG signal values).
    Xmean = []  # List to store mean input data points.
    Ymean = []  # List to store mean output data points.
    Yvars = []  # List to store variance of output data points.

    # Loop through all channels and sample 'n' data points per channel.
    for ch in CHS:
        # Randomly sample 'n' maximum values from the data of the current channel.
        ys = random.sample(trains[ch][ch][0]['data'].max(axis=1).tolist(), n)
        Y.extend(ys)  # Add the sampled maximum values to the output data list.

        # Get the variance of the maximum values for the current channel.
        var = trains[ch][ch][0]['stdmax'] ** 2
        Yvars.extend([var] * len(ys))  # Add the variance to the variance list.

        xy = ch2xy[ch]  # Get the (x, y) coordinates of the current channel.
        X.extend([xy] * len(ys))  # Add the coordinates to the input data list.

        # Get the mean value and add_kernel it to the mean lists.
        Ymean.append(trains[ch][ch][0]['meanmax'])
        Xmean.extend([xy])

    # If mean is False, convert the input and output lists into numpy arrays and return.
    X = np.array(X)
    Y = np.array(Y).reshape((-1, 1))
    Xmean = np.array(Xmean)
    Ymean = np.array(Ymean)
    return X, Y, Xmean, Ymean


def random_initialization_1D(random_sample, trainsC, max_seen_resp, dt=DT, emg=EMG, max_update=False):
    """
    Perform random initialization for training points.

    Parameters:
    - random_sample (int): Number of random samples.
    - emg: EMG data.
    - trainsC: Training configuration.

    Returns:
    - X: List of input data.
    - Y: List of output data.
    """
    # Initial random training points
    X = []
    Y = []
    for _ in range(random_sample):
        ch = random.choice(CHS)
        X.append(ch2xy[ch])
        response = random.choice(trainsC.emgdct[emg][ch][ch][dt]['data'].max(axis=1))

        # Add of if statement and max seen resp in the case of several initialization points
        if (response > max_seen_resp) or (max_seen_resp == 0):
            max_seen_resp = response

        # TODO: see if this normalization makes sense and if not do it correctly
        Y.append(response)  # need to Normalization between 0 and 1

    X = torch.tensor(X, dtype=torch.float64)
    if max_update:
        Y = torch.tensor(Y, dtype=torch.float64)

    else:
        Y = torch.tensor(Y, dtype=torch.float64)/max_seen_resp

    return X, Y


def update_training_data(train_x, train_y, next_query_pins, next_query_value):
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

    return train_x, train_y


def count_pin(selected_pin_array, x):
    # TODO: verify, maybe not working properly for HGP-BO
    """
    Count the occurrences of each pin in the selected pin array.

    Parameters:
    - selected_pin_array (torch.Tensor): Array containing selected pin to count.
    - x (torch.Tensor): Array of pin coordinates, e.g., see above

    Returns:
    - arr_count (np.ndarray): Array containing the count of occurrences for each pin.
    """
    arr = tuple(map(tuple, (selected_pin_array)))  # set the selected_pin_array as tuple to be iterable by Counter
    count_occurences = Counter(arr)  # return the counting value but not the zeros
    count_occurences = {torch.tensor(k): int(v) for k, v in count_occurences.items()}

    dict_pins = {}  # set a new dict with key as all pins coordinates and value 0
    for i in x:
        dict_pins[i] = 0

    for key1 in dict_pins:  # compare both dict and replace value of dict_pins by corresponding count_occurences value if both key are the same
        for key2 in count_occurences:
            if (key1[0] == key2[0] and key1[1] == key2[1]):
                dict_pins[key1] = count_occurences.get(key2)

    result = dict_pins.items()
    result_list = list(result)
    arr_count = np.array(result_list, dtype="object")  # changing from tensor to array to be serialize and saved as json

    for i in range(len(arr_count)):
        arr_count[i][0] = np.array(arr_count[i][0])

    return arr_count


def reset_data_storage(dict_of_array):
    """
    Reset data storage for the next repetition.

    Parameters:
    - dict_of_array: dictionnary of arrays to reset

    Returns:
    - dict_of_array: : dictionnary of arrays set to zero
    """

    for keys, array in dict_of_array.items():
        if keys == 'kappa_value' or keys == 'nbr_of_repetition' or keys == 'nbr_of_query_per_rep' or keys == 'pts_of_ini' or keys == 'emg':
            pass
        elif type(array) == list:
            array = []
        else:
            array.fill(0)

    return dict_of_array


def save_data(file_name, folder_of_the_day, workspace_folder, data_dict):
    # Go to the folder of the day and save data
    os.chdir(folder_of_the_day)
    with open(str(file_name) + '.json', 'w') as file:
        json.dump(data_dict, file, cls=NumpyArrayEncoder)
    # Go back to workspace
    os.chdir(workspace_folder)


def create_data_storage(nbr_query, nbr_repetition, nbr_pins):
    # TODO: convert all the returned arrays into a dictionary to make things easier
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
    array_pins_coord = np.zeros(((nbr_query * nbr_repetition), 4))
    array_pins_coord_exploration = np.zeros(((nbr_query * nbr_repetition), 4))
    # Array for counting each exploration and exploitation pins selected
    array_pins_count_exploration = np.zeros(nbr_pins)
    array_pins_count = np.zeros(nbr_pins)

    return array_repetition, array_query, array_exploration_score, array_exploitation_score, array_explr_mean, array_explt_mean, array_confUp_explr_mean, array_confUp_explt_mean, array_confLow_explr_mean, array_confLow_explt_mean, array_pins_coord, array_pins_coord_exploration, array_pins_count_exploration, array_pins_count


def create_data_storage_1D(nbr_query, nbr_repetition, nbr_pins):
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

    return array_repetition, array_query, array_exploration_score, array_exploitation_score, array_explr_mean, array_explt_mean, array_confUp_explr_mean, array_confUp_explt_mean, array_confLow_explr_mean, array_confLow_explt_mean, array_pins_coord, array_pins_coord_exploration, array_pins_count_exploration, array_pins_count


def make_dataset_2d(trainsC, emg=EMG, syn=None, dt=DT, n=None):
    if syn is not None:
        assert type(syn) is tuple, "syn should be a tuple of 2 emgs. (eg. (0,4))"
    else:
        syn = (emg, None)
    trains = trainsC.get_emgdct(emg)
    X = []
    Y = []
    Xmean = []
    Ymean = []
    for i, ch1 in enumerate(CHS):
        for j, ch2 in enumerate(CHS):
            xych1 = ch2xy[ch1]
            xych2 = ch2xy[ch2]
            # Note that here we are only taking the pairs (ch1,ch2)
            # But will miss (ch2,ch1). These are the same data (for
            # dt=0),but the GP should still have this info
            # However, with 2000 pts its already slow enough so we
            # don't include them for now (this is prob bad though)
            # train time: 4000 pts = 10 mins
            #             2000 pts = 2 mins
            emg1, emg2 = syn
            ys = trainsC.synergy(emg1, emg2, ch1, ch2, dt).max(axis=1)
            if n:
                ys = random.sample(trains[ch1][ch2][dt]['data'].max(axis=1).tolist(), n)
            Y.extend(ys)
            X.extend([xych1 + xych2] * len(ys))
            # we also make a small dataset with means
            Ymean.append(ys.mean())
            Xmean.extend([xych1 + xych2])
    Xmean = np.array(Xmean)
    Ymean = np.array(Ymean).reshape((-1, 1))
    X = np.array(X)
    Y = np.array(Y).reshape((-1, 1))
    return X, Y, Xmean, Ymean


def random_hp_values_generation(lower, upper, num_hp: int):
    """
    Imitates the random hp selection from randomized grid search in SKLearn as it is optimal over standard
        will be useful for finding the best kappa, nu values for the kernels and acquisition functions

    Runtime might seem awful cuz of the while loop but will be O(num_hp^2) maybe there's an optimal way of doing
        but honestly this is good enough for our uses since the hp draw will never be huge
    :param lower: the lower range from which we can draw the hp
    :param upper: the upper range from which we can draw the hp
    :param num_hp: the number of hp variations to select
    :return: a list of randomly selected hp values to attempt
    """

    hp_values = []

    for i in range(num_hp):
        candidate = random.uniform(lower, upper)
        while candidate in hp_values:
            candidate = random.uniform(lower, upper)

        hp_values.append(candidate)

    return hp_values


def select_random_queries(num_queries, x, y, seed=False, noise=0):
    if type(seed) != bool:
        np.random.seed(seed)

    indices = np.random.randint(len(x), size=(num_queries))
    train_x = x[indices]
    train_y = y[indices]
    train_y += (torch.max(y)-torch.min(y)) * np.random.normal(0, noise, size=train_y.shape)

    return train_x, train_y


def hierarchical_select_random_queries(num_queries, x, y, seed=False, noise=0):
    if type(seed) != bool:
        np.random.seed(seed)
    indices_X = np.random.randint(len(x), size=(num_queries))
    indices_Y = np.random.randint(len(x), size=(num_queries))

    train_x = x[indices_X, indices_Y]
    train_y = y[indices_X, indices_Y]
    train_y += torch.normal(0, noise, size=train_y.shape, device=train_y.device)*(y.max()-y.min())

    return train_x, train_y


def make_test_sub(num_queries, x):
    indices_X = np.random.randint(len(x), size=(num_queries))

    test_x = x[indices_X]
    return test_x


def make_test_hierarchical(num_queries, x_vals):
    test_x = torch.zeros((num_queries, 2))
    for i in range(num_queries):
        index1, index2 = np.random.randint(0, len(x_vals), (2))
        test_x[i][0] = x_vals[index1][index2][0]
        test_x[i][1] = x_vals[index1][index2][1]

    return test_x


def generate_sin_cos_dataset(dimension, eps):
    x_sub1 = torch.linspace(0, 2, dimension).double()
    y_sub1 = torch.sin(x_sub1 * (2 * math.pi)) + torch.randn(x_sub1.size()) * math.sqrt(0.04)
    y_sub1 = y_sub1.double()

    x_sub2 = torch.linspace(0, 2, dimension).double()
    y_sub2 = torch.cos(x_sub2 * (2 * math.pi)) + torch.randn(x_sub1.size()) * math.sqrt(0.04)
    y_sub2 = y_sub2.double()

    x_hier = torch.zeros((dimension, dimension, 2)).double()
    torch.autograd.set_detect_anomaly(True)

    for i in range(len(x_sub1)):
        for j in range(len(x_sub2)):
            x_hier[i, j, 0] = x_sub1[i]
            x_hier[i, j, 1] = x_sub2[j]

    y_hier = torch.zeros((dimension, dimension))

    test_x = make_test_sub(5, x_sub1)
    # test_x_hier = make_test_hierarchical(10, x_hier)
    test_x_hier = torch.reshape(x_hier, (-1, 2))

    for i in range(len(y_sub1)):
        for j in range(len(y_sub2)):
            y_hier[i, j] = (y_sub1[i] + y_sub2[j]) / (
                (abs(x_sub1[i] - x_sub2[j]) + eps))  # Adding a convolution and need epsilon
            # the added 1e-10 will make it a multiplier if they are right on each other and no effect if far

    y_hier = y_hier.double()

    return x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier


def generate_diff_dataset(dimension, eps):
    x_sub1 = torch.linspace(0, 2, dimension).double()
    y_sub1 = torch.zeros(x_sub1.shape)

    for i, x in enumerate(x_sub1):
        y_sub1[i] = -(x - 2) ** 2 + 2
    y_sub1 = y_sub1.double()

    x_sub2 = torch.linspace(0, 2, dimension).double()
    y_sub2 = torch.zeros(x_sub2.shape)

    for i, x in enumerate(x_sub1):
        y_sub2[i] = -(x - 1 / 2) ** 4 + 2

    y_sub2 = y_sub2.double()

    x_hier = torch.zeros((dimension, dimension, 2)).double()

    for i in range(len(x_sub1)):
        for j in range(len(x_sub2)):
            x_hier[i, j, 0] = x_sub1[i]
            x_hier[i, j, 1] = x_sub2[j]

    y_hier = torch.zeros((dimension, dimension))

    test_x = make_test_sub(5, x_sub1)
    # test_x_hier = make_test_hierarchical(10, x_hier)
    test_x_hier = torch.reshape(x_hier, (-1, 2))

    for i in range(len(y_sub1)):
        for j in range(len(y_sub2)):
            y_hier[i, j] = (y_sub1[i] + y_sub2[j]) / (
                ((x_sub1[i] - x_sub2[j]) ** 2 + eps))  # Adding a convolution and need epsilon
            # the added 1e-10 will make it a multiplier if they are right on each other and no effect if far

    y_hier = y_hier.double()

    return x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier


def generate_synthetic3_dataset(dimension, eps):
    x_sub1 = torch.linspace(0, 2, dimension).double()
    y_sub1 = torch.sin(x_sub1 * (2 * math.pi)) / (x_sub1 - 1) + 2
    y_sub1 = torch.where(y_sub1 == -torch.inf, 6.283, y_sub1)

    y_sub1 = y_sub1.double()

    x_sub2 = torch.linspace(0, 2, dimension).double()
    y_sub2 = torch.sin(x_sub2 * (2 * math.pi)) / (x_sub2 - 2) + 2
    y_sub2 = torch.where(y_sub2 == -torch.inf, 6.283, y_sub2)  # solved the overflow by limits

    y_sub2 = y_sub2.double()

    x_hier = torch.zeros((dimension, dimension, 2)).double()

    for i in range(len(x_sub1)):
        for j in range(len(x_sub2)):
            x_hier[i, j, 0] = x_sub1[i]
            x_hier[i, j, 1] = x_sub2[j]

    y_hier = torch.zeros((dimension, dimension))

    test_x = make_test_sub(5, x_sub1)
    # test_x_hier = make_test_hierarchical(10, x_hier)
    test_x_hier = torch.reshape(x_hier, (-1, 2))

    for i in range(len(y_sub1)):
        for j in range(len(y_sub2)):
            y_hier[i, j] = (y_sub1[i] + y_sub2[j]) / (
                ((x_sub1[i] - x_sub2[j]) ** 2 + eps))  # Adding a convolution and need epsilon
            # the added 1e-10 will make it a multiplier if they are right on each other and no effect if far

    y_hier = y_hier.double()

    return x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier

def generate_sub_dataset(dimension, eps):
    x_sub1 = torch.linspace(0, 2, dimension).double()
    y_sub1 = torch.sin(x_sub1 * (2 * math.pi))
    y_sub1 = y_sub1.double()

    x_sub2 = torch.linspace(0, 2, dimension).double()
    y_sub2 = torch.sin(x_sub2 * (2 * math.pi))

    y_sub2 = y_sub2.double()

    x_hier = torch.zeros((dimension, dimension, 2)).double()

    for i in range(len(x_sub1)):
        for j in range(len(x_sub2)):
            x_hier[i, j, 0] = x_sub1[i]
            x_hier[i, j, 1] = x_sub2[j]

    y_hier = torch.zeros((dimension, dimension))

    test_x = make_test_sub(5, x_sub1)
    test_x_hier = torch.reshape(x_hier, (-1, 2))

    for i in range(len(y_sub1)):
        for j in range(len(y_sub2)):
            y_hier[i, j] = y_sub1[i]  - y_sub2[j] + 5e-5

    y_hier = y_hier.double()

    return x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier

def generate_3d_dataset(dimension, eps):
    x_sub1 = torch.linspace(0, 4, dimension).double()
    y_sub1 = torch.zeros(x_sub1.shape)

    for i, x in enumerate(x_sub1):
        y_sub1[i] = -(x - 2) ** 5
    y_sub1 = y_sub1.double()

    x_sub2 = torch.linspace(0, 4, dimension).double()
    y_sub2 = torch.sin(x_sub2)**3
    #y_sub2 = torch.where(y_sub2 == -torch.inf, 6.283, y_sub2)  # solved the overflow by limits

    y_sub2 = y_sub2.double()

    x_sub3 = torch.linspace(0, 4, dimension).double()
    y_sub3 = (torch.log(x_sub3 + 1) + 1) / (x_sub3 + 1)
    y_sub3 = y_sub3.double()

    x_hier = torch.zeros((dimension, dimension, dimension, 3)).double()

    for i in range(len(x_sub1)):
        for j in range(len(x_sub2)):
            for k in range(len(x_sub3)):
                x_hier[i, j, k, 0] = x_sub1[i]
                x_hier[i, j, k, 1] = x_sub2[j]
                x_hier[i, j, k, 2] = x_sub3[k]

    y_hier = torch.zeros((dimension, dimension, dimension))

    test_x = make_test_sub(5, x_sub1)
    # test_x_hier = make_test_hierarchical(10, x_hier)
    test_x_hier = torch.reshape(x_hier, (-1, 3))

    for i in range(len(y_sub1)):
        for j in range(len(y_sub2)):
            for k in range(len(y_sub3)):
                y_hier[i, j] = (y_sub1[i] + y_sub2[j] + y_sub3[k])**2 / (
                    (((x_sub1[i] - x_sub2[j]) + (x_sub1 - x_sub3) + (x_sub2 - x_sub3)) ** 2 + eps))

    y_hier = y_hier.double()

    return x_sub1, y_sub1, x_sub2, y_sub2, x_sub3, y_sub3, x_hier, y_hier, test_x, test_x_hier


def generate_3d_2_dataset(dimension, eps):
    x_sub1 = torch.linspace(0, 4, dimension).double()
    y_sub1 = torch.zeros(x_sub1.shape)

    for i, x in enumerate(x_sub1):
        y_sub1[i] = -(x - 2) ** 5
    y_sub1 = y_sub1.double()

    x_sub2 = torch.linspace(0, 4, dimension).double()
    y_sub2 = torch.sin(x_sub2)**3
    #y_sub2 = torch.where(y_sub2 == -torch.inf, 6.283, y_sub2)  # solved the overflow by limits

    y_sub2 = y_sub2.double()

    x_sub3 = torch.linspace(0, 4, dimension).double()
    y_sub3 = (torch.log(x_sub3 + 1) + 1) / (x_sub3 + 1)
    y_sub3 = y_sub3.double()

    x_hier = torch.zeros((dimension, dimension, dimension, 3)).double()

    for i in range(len(x_sub1)):
        for j in range(len(x_sub2)):
            for k in range(len(x_sub3)):
                x_hier[i, j, k, 0] = x_sub1[i]
                x_hier[i, j, k, 1] = x_sub2[j]
                x_hier[i, j, k, 2] = x_sub3[k]

    y_hier = torch.zeros((dimension, dimension, dimension))

    test_x = make_test_sub(5, x_sub1)
    # test_x_hier = make_test_hierarchical(10, x_hier)
    test_x_hier = torch.reshape(x_hier, (-1, 3))

    for i in range(len(y_sub1)):
        for j in range(len(y_sub2)):
            for k in range(len(y_sub3)):
                y_hier[i, j] = (y_sub1[i] + y_sub2[j] + y_sub3[k])**2 + 2

    y_hier = y_hier.double()

    return x_sub1, y_sub1, x_sub2, y_sub2, x_sub3, y_sub3, x_hier, y_hier, test_x, test_x_hier

def generate_modularity_dataset(dimension, eps):
    x_sub1 = torch.linspace(1, 3, dimension).double()
    y_sub1 = torch.zeros(x_sub1.shape)

    for i, x in enumerate(x_sub1):
        y_sub1[i] = -(x - 2) ** 2 + 2
    y_sub1 = y_sub1.double()

    x_sub2 = torch.linspace(0, 2, dimension).double()
    y_sub2 = torch.sin(x_sub2 * (2 * math.pi)) / (x_sub2 - 1) + 2
    y_sub2 = torch.where(y_sub2 == -torch.inf, 6.283, y_sub2)  # solved the overflow by limits

    y_sub2 = y_sub2.double()

    x_hier = torch.zeros((dimension, dimension, 2)).double()

    for i in range(len(x_sub1)):
        for j in range(len(x_sub2)):
            x_hier[i, j, 0] = x_sub1[i]
            x_hier[i, j, 1] = x_sub2[j]

    y_hier = torch.zeros((dimension, dimension))

    test_x = make_test_sub(5, x_sub1)
    # test_x_hier = make_test_hierarchical(10, x_hier)
    test_x_hier = torch.reshape(x_hier, (-1, 2))

    for i in range(len(y_sub1)):
        for j in range(len(y_sub2)):
            y_hier[i, j] = (y_sub1[i] + y_sub2[j]) / (
                ((x_sub1[i] - x_sub2[j]) ** 2 + eps))  # Adding a convolution and need epsilon
            # the added 1e-10 will make it a multiplier if they are right on each other and no effect if far

    y_hier = y_hier.double()

    return x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier

def generate_exponential_nonlinearity_dataset(dimension, eps):
    x_sub1 = torch.linspace(0, 2, dimension).double()
    y_sub1 = torch.sin(x_sub1)
    y_sub1 = y_sub1.double()

    x_sub2 = torch.linspace(0, 2, dimension).double()
    y_sub2 = torch.tanh(x_sub2)

    y_sub2 = y_sub2.double()

    x_hier = torch.zeros((dimension, dimension, 2)).double()

    for i in range(len(x_sub1)):
        for j in range(len(x_sub2)):
            x_hier[i, j, 0] = x_sub1[i]
            x_hier[i, j, 1] = x_sub2[j]

    y_hier = torch.zeros((dimension, dimension))

    test_x = make_test_sub(5, x_sub1)
    # test_x_hier = make_test_hierarchical(10, x_hier)
    test_x_hier = torch.reshape(x_hier, (-1, 2))

    for i in range(len(y_sub1)):
        for j in range(len(y_sub2)):
            y_hier[i, j] = torch.pow((y_sub1[i] + y_sub2[j])/2, eps)  # Adding a convolution and need epsilon
            # the added 1e-10 will make it a multiplier if they are right on each other and no effect if far

    y_hier = y_hier.double()

    return x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier

def generate_exponential_inside_nonlinearity_dataset(dimension, eps):
    x_sub1 = torch.linspace(0, 2, dimension).double()
    y_sub1 = torch.sin(x_sub1)
    y_sub1 = y_sub1.double()

    x_sub2 = torch.linspace(0, 2, dimension).double()
    y_sub2 = torch.tanh(x_sub2)

    y_sub2 = y_sub2.double()

    x_hier = torch.zeros((dimension, dimension, 2)).double()

    for i in range(len(x_sub1)):
        for j in range(len(x_sub2)):
            x_hier[i, j, 0] = x_sub1[i]
            x_hier[i, j, 1] = x_sub2[j]

    y_hier = torch.zeros((dimension, dimension))

    test_x = make_test_sub(5, x_sub1)
    # test_x_hier = make_test_hierarchical(10, x_hier)
    test_x_hier = torch.reshape(x_hier, (-1, 2))

    for i in range(len(y_sub1)):
        for j in range(len(y_sub2)):
            y_hier[i, j] = (torch.pow(y_sub1[i], eps) + y_sub2[j])/2  # Adding a convolution and need epsilon
            # the added 1e-10 will make it a multiplier if they are right on each other and no effect if far

    y_hier = y_hier.double()

    return x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier

def generate_mult_factor_nonlinearity_dataset(dimension, eps):
    x_sub1 = torch.linspace(0, 2, dimension).double()
    y_sub1 = torch.sin(x_sub1)
    y_sub1 = y_sub1.double()

    x_sub2 = torch.linspace(0, 2, dimension).double()
    y_sub2 = torch.tanh(x_sub2)

    y_sub2 = y_sub2.double()

    x_hier = torch.zeros((dimension, dimension, 2)).double()

    for i in range(len(x_sub1)):
        for j in range(len(x_sub2)):
            x_hier[i, j, 0] = x_sub1[i]
            x_hier[i, j, 1] = x_sub2[j]

    y_hier = torch.zeros((dimension, dimension))

    test_x = make_test_sub(5, x_sub1)
    # test_x_hier = make_test_hierarchical(10, x_hier)
    test_x_hier = torch.reshape(x_hier, (-1, 2))

    for i in range(len(y_sub1)):
        for j in range(len(y_sub2)):
            y_hier[i, j] = y_sub1[i] + y_sub2[j] + eps * y_sub1[i] * y_sub2[j]  # Adding a convolution and need epsilon
            # the added 1e-10 will make it a multiplier if they are right on each other and no effect if far

    y_hier = y_hier.double()

    return x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier

def generate_b_mult_factor_nonlinearity_dataset(dimension, eps):
    x_sub1 = torch.linspace(0, 2, dimension).double()
    y_sub1 = torch.sin(x_sub1)
    y_sub1 = y_sub1.double()

    x_sub2 = torch.linspace(0, 2, dimension).double()
    y_sub2 = torch.tanh(x_sub2)

    y_sub2 = y_sub2.double()

    x_hier = torch.zeros((dimension, dimension, 2)).double()

    for i in range(len(x_sub1)):
        for j in range(len(x_sub2)):
            x_hier[i, j, 0] = x_sub1[i]
            x_hier[i, j, 1] = x_sub2[j]

    y_hier = torch.zeros((dimension, dimension))

    test_x = make_test_sub(5, x_sub1)
    # test_x_hier = make_test_hierarchical(10, x_hier)
    test_x_hier = torch.reshape(x_hier, (-1, 2))

    b1, b2 = eps
    for i in range(len(y_sub1)):
        for j in range(len(y_sub2)):
            y_hier[i, j] = b1*y_sub1[i] + b2*y_sub2[j]   # Adding a convolution and need epsilon

    y_hier = y_hier.double()

    return x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier


def get_dataset_info(dataset_num, alpha=1):
    if dataset_num == 1:
        data_name = 'sin_cos'
        data_creation_func = generate_diff_dataset
        eps = 1 / 5
    elif dataset_num == 2:
        data_name = 'synthetic_tests'
        data_creation_func = generate_diff_dataset
        eps = 3 / 2

    elif dataset_num == 3:
        data_name = 'synthetic3'
        data_creation_func = generate_synthetic3_dataset
        eps = 2
    elif dataset_num == 4:
        data_name = 'first_3D'
        data_creation_func = generate_3d_dataset
        eps = 10
    elif dataset_num == 5:
        data_name = 'second_3D'
        data_creation_func = generate_3d_2_dataset
        eps = 2
    elif dataset_num == 6:
        data_name = 'modular_2D'
        data_creation_func = generate_modularity_dataset
        eps = 5
    elif dataset_num == 7:
        data_name = 'nonlinearity_exponential'
        data_creation_func = generate_exponential_nonlinearity_dataset
        eps = alpha
    elif dataset_num == 7.5:
        data_name = 'nonlinearity_exponential_inside'
        data_creation_func = generate_exponential_inside_nonlinearity_dataset
        eps = alpha
    elif dataset_num == 8:
        data_name = 'nonlinearity_mult_factor'
        data_creation_func = generate_mult_factor_nonlinearity_dataset
        eps = alpha
    elif dataset_num == 9:
        data_name = 'nonlinearity_beta_mult_factor'
        data_creation_func = generate_b_mult_factor_nonlinearity_dataset
        eps = alpha
    elif dataset_num == 10:
        data_name = 'sub_2D'
        data_creation_func = generate_sub_dataset
        eps = alpha
    else:
        raise AssertionError("Dataset number invalid")

    return data_name, data_creation_func, eps

if __name__ == '__main__':
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = generate_sin_cos_dataset(10, 3 / 4)

    sns.heatmap(y_hier)
    plt.show()
