import laferriere_script_synthetic as laf
import efficient_general_2d as gen
import vanilla_gpbo_synthetic as van

import hmodel_synthetic as hmodel
from dataset_actions import *
import multiprocessing as mp
from tqdm import tqdm


if __name__ == '__main__':
    dimension = 32
    nbr_query = 100
    training_iter = [5, 10, 15, 20]  # Found through HP Testing
    nbr_repetition = 15
    base_k = [7.5]
    base_g = [3.5]
    base_train = 10
    base_rand = 3
    k_vals = [4, 5, 6, 7, 7.5, 8, 9, 10]
    g_vals = [1, 2, 3, 3.5, 4, 5, 6]
    nu_vals = [0.5]  # Found through HP Testing
    multi = False
    h_model = [hmodel.Lossless_Efficient_UCB_Hierarchical_GP]
    process = []
    nbr_rand_init = [1, 2, 3, 5, 10, 15, 20]  # Found through HP Testing
    """seed = np.array([901112484, 798576827, 862109006, 256960071, 67686131, 960919614,
                     542146925, 225453837, 328655096, 167690914, 578139702, 126081086,
                     445226178, 339718381, 278636500, 570547118, 459828174, 673392709,
                     56896553, 749380297, 635521450, 19699771, 351850900, 520687372,
                     833438344, 355138099, 382604277, 40529313, 441069895, 797772191])
    seed = [False] * nbr_repetition"""
    seed = np.array([791104038, 558883516,  75533178, 730586104,  64343038, 353199330,
       138876529, 594536092, 713725275, 642158682, 287397414, 156569942,
       554978049, 860858855, 899218178])
    #seed = np.random.randint(999999999, size=nbr_repetition)
    datasets = [3, 2, 6, 10]
    processes = []
    with mp.Pool(processes=len(datasets)) as pool:
        for h in h_model:
            # Kappa block
            for dataset_num in datasets:
                data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                p = pool.apply_async(gen.training_procedure,
                                     (nbr_query, nbr_repetition, base_rand, dimension, base_train, k_vals,
                                      base_g,
                                      nu_vals, data_name, data_creation_func,
                                      eps, h, multi, seed, [], True, 0.1, 0, True,))
                processes.append(p)
                p = pool.apply_async(laf.training_procedure,
                                     (nbr_query, nbr_repetition, base_rand, dimension, base_train, k_vals,
                                      base_g,
                                      nu_vals, data_name, data_creation_func,
                                      eps, h, multi, seed, [], True, 0.1, True,))
                processes.append(p)
                p = pool.apply_async(van.training_procedure,
                                     (nbr_query, nbr_repetition, 1, dimension, base_train, k_vals,
                                      data_name, data_creation_func,
                                      eps, seed, True))
                processes.append(p)
            for i, p in enumerate(processes):
                try:
                    p.get()
                except:
                    print(f"Kappa Process {i} failed")
            processes = []

            print(f"\n=====================================================")
            print(f"Kappa Complete")
            print(f"=====================================================\n")
            # Gamme Block
            for dataset_num in datasets:
                data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                p = pool.apply_async(gen.training_procedure,
                                     (nbr_query, nbr_repetition, base_rand, dimension, base_train, base_k,
                                      g_vals,
                                      nu_vals, data_name, data_creation_func,
                                      eps, h, multi, seed, [], True, 0.1, 0, True,))
                processes.append(p)
                p = pool.apply_async(laf.training_procedure,
                                     (nbr_query, nbr_repetition, base_rand, dimension, base_train, base_k,
                                      g_vals,
                                      nu_vals, data_name, data_creation_func,
                                      eps, h, multi, seed, [], True, 0.1, True,))
                processes.append(p)
            for i, p in enumerate(processes):
                try:
                    p.get()
                except:
                    print(f"Gamma Process {i} failed")
            processes = []
            print(f"\n=====================================================")
            print(f"Gamma Complete")
            print(f"=====================================================\n")
            # Rand Init Block
            for rand_init in nbr_rand_init:
                for dataset_num in datasets:
                    data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                    p = pool.apply_async(gen.training_procedure,
                                         (nbr_query, nbr_repetition, rand_init, dimension, base_train, base_k,
                                          base_g,
                                          nu_vals, data_name, data_creation_func,
                                          eps, h, multi, seed, [], True, 0.1, 0, True,))
                    processes.append(p)
                    p = pool.apply_async(laf.training_procedure,
                                         (nbr_query, nbr_repetition, rand_init, dimension, base_train, base_k,
                                          base_g,
                                          nu_vals, data_name, data_creation_func,
                                          eps, h, multi, seed, [], True, 0.1, True,))
                    processes.append(p)
            for i, p in enumerate(processes):
                try:
                    p.get()
                except:
                    print(f"Rand Init Process {i} failed")
            processes = []
            print(f"\n=====================================================")
            print(f"Rand Init Complete")
            print(f"=====================================================\n")
            # Train Iteration block

            for train_iter in training_iter:
                for dataset_num in datasets:
                    data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                    p = pool.apply_async(gen.training_procedure, (nbr_query, nbr_repetition, base_rand, dimension, train_iter, base_k,
                                           base_g,
                                           nu_vals, data_name, data_creation_func,
                                           eps, h, multi, seed, [], True, 0.1, 0, True,))
                    processes.append(p)
                    p = pool.apply_async(laf.training_procedure,
                                         (nbr_query, nbr_repetition, base_rand, dimension, train_iter, base_k,
                                          base_g,
                                          nu_vals, data_name, data_creation_func,
                                          eps, h, multi, seed, [], True, 0.1, True,))
                    processes.append(p)
                    p = pool.apply_async(van.training_procedure, (nbr_query, nbr_repetition, 1, dimension, train_iter, base_k,
                                       data_name, data_creation_func,
                                       eps, seed, True))
                    processes.append(p)

            for i, p in enumerate(processes):
                try:
                    p.get()
                except:
                    print(f"Training Iter Process {i} failed")
                processes = []

            print(f"\n=====================================================")
            print(f"Training Iter Complete")
            print(f"=====================================================\n")

