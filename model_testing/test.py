import laferriere_script_synthetic as laf
import efficient_general_2d as gen
import gpytorch


import synthetic_models as models
import hmodel_synthetic as hmodel
from dataset_actions import *
from datetime import datetime
import visualization_information as vi
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from seaborn import heatmap

from model_testing.efficient_synthetic_script import joint_performance

if __name__ == '__main__':
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    multi = True
    dimension = 15
    noise = 0.1
    nbr_repetition = 20
    nbr_query = 100
    seed = np.random.randint(99999, size=nbr_repetition)
    k_vals_list = [1, 2, 3, 4, 5, 6, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]
    g_vals_list = [1, 2, 3, 4, 5, 6, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]
    nu_vals = [0.5]  # Found through HP Testing
    nbr_rand_init = 6  # Found through HP Testing
    training_iter = 10  # Found through HP Testing

    k_vals = [9.5]
    g_vals = [3]
    for dataset_num in [2, 3]:


        data_name, data_creation_func, eps = get_dataset_info(dataset_num)

        """if h == hmodel.Efficient_UCB_Hierarchical_GP:
            model_name = "Efficient"

        elif h == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
            model_name = "Lossless_Efficient"

        current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
        current_dateday = datetime.now().strftime("%Y-%m-%d")
        workspace = f"{data_name}/{model_name.lower()}"
        folder_of_the_day = '/data-' + str(current_dateday)"""
        name, master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2 = \
            gen.training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals_list,
                                   g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                   noise=noise)[0]
        name, master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2 = \
            gen.training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                   g_vals_list, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                   noise=noise)[0]
