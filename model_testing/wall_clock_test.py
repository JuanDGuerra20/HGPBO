import laferriere_script_synthetic as laf
import efficient_general_2d as gen
import vanilla_gpbo_synthetic as van
import deep_gp_synthetic as dgp
import add_gp_ucb_synthetic as add_ucb

import hmodel_synthetic as hmodel
from dataset_actions import *
import multiprocessing as mp
from tqdm import tqdm


if __name__ == '__main__':
    dimension = 32
    nbr_query = 100
    training_iter = [5, 10, 15, 20]  # Found through HP Testing
    nbr_repetition = 10
    base_k = [7.5]
    base_g = [3.5]
    base_train = 10
    base_rand = 3
    k_vals = [4, 5, 6, 7, 7.5, 8, 9]
    g_vals = [1, 2, 3, 3.5, 4, 5, 6]
    nu_vals = [0.5]  # Found through HP Testing
    multi = True
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    process = []
    nbr_rand_init = [1, 2, 3, 5, 10, 15, 20]  # Found through HP Testing

    seed = [False] * nbr_repetition
    """seed = np.array([791104038, 558883516,  75533178, 730586104,  64343038, 353199330,
       138876529, 594536092, 713725275, 642158682, 287397414, 156569942,
       554978049, 860858855, 899218178])"""
    #seed = np.random.randint(999999999, size=nbr_repetition)
    datasets = [11]
    processes = []
    data_name, data_creation_func, eps = get_dataset_info(dataset_num)


    for i in range(1, 11):
