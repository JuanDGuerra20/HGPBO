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
    multi = False
    dimension = 15
    noise_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    nbr_repetition = 30
    nbr_query = 100
    seed = np.random.randint(99999, size=nbr_repetition)

    for dataset_num in [2,3]:
        for noise in noise_list:

            print(f"Entering Noise {noise}")

            data_name, data_creation_func, eps = get_dataset_info(dataset_num)

            """if h == hmodel.Efficient_UCB_Hierarchical_GP:
                model_name = "Efficient"
    
            elif h == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
                model_name = "Lossless_Efficient"
    
            current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
            current_dateday = datetime.now().strftime("%Y-%m-%d")
            workspace = f"{data_name}/{model_name.lower()}"
            folder_of_the_day = '/data-' + str(current_dateday)"""
            training_iter = 10  # Found through HP Testing
            k_vals = [2]
            g_vals = [6]
            nu_vals = [0.5]  # Found through HP Testing
            nbr_rand_init = 6  # Found through HP Testing

            name, master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2 = \
                laf.training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                   g_vals,nu_vals, data_name, data_creation_func, eps, h_model, multi, seed, noise=noise)[0]

            training_iter = 10  # Found through HP Testing
            k_vals = [9.5]
            g_vals = [10]
            nu_vals = [0.5]  # Found through HP Testing
            nbr_rand_init = 6  # Found through HP Testing
            name, master, better_exploration_score, better_exploitation_score, r2, child_1_r2, child_2_r2 = \
                gen.training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed, noise=noise)[0]