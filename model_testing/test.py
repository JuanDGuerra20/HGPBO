import laferriere_script_synthetic as laf
import efficient_general_2d as gen
import vanilla_gpbo_synthetic as van

import hmodel_synthetic as hmodel
from dataset_actions import *
import multiprocessing as mp


if __name__ == '__main__':
    dimension = 32
    nbr_query = 100
    training_iter = 10  # Found through HP Testing
    nbr_repetition = 30
    k_vals = [7.5]
    g_vals = [3]
    nu_vals = [0.5]  # Found through HP Testing
    multi = False
    h_model = [hmodel.Lossless_Efficient_UCB_Hierarchical_GP]
    process = []
    nbr_rand_init = 3  # Found through HP Testing
    """seed = np.array([901112484, 798576827, 862109006, 256960071, 67686131, 960919614,
                     542146925, 225453837, 328655096, 167690914, 578139702, 126081086,
                     445226178, 339718381, 278636500, 570547118, 459828174, 673392709,
                     56896553, 749380297, 635521450, 19699771, 351850900, 520687372,
                     833438344, 355138099, 382604277, 40529313, 441069895, 797772191])
    seed = [False] * nbr_repetition"""
    seed = np.random.randint(999999999, size=nbr_repetition)
    datasets = [3, 2, 6, 10]
    processes = []
    processes_laf = []
    processes_van = []
    with mp.Pool(processes=len(datasets)) as pool:
        for h in h_model:
            for dataset_num in datasets:
                data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                p = pool.apply_async(gen.training_procedure, (nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals,
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, [], True, 0.1, 0,))
                processes.append(p)
                p = pool.apply_async(laf.training_procedure,
                                     (nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                      g_vals,
                                      nu_vals, data_name, data_creation_func,
                                      eps, h, multi, seed, [], True, 0.1, 0,))
                processes.append(p)
                p = pool.apply_async(van.training_procedure, (nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                   data_name, data_creation_func,
                                   eps, ))
                processes.append(p)

        for i, p in enumerate(processes):
            try:
                p.get()
            except:
                print(f"Process {i} failed")
