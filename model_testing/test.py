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
    nbr_repetition = 1
    base_k = [7.5]
    base_g = [3.5]
    base_train = 10
    base_rand = 3
    k_vals = [4, 5, 6, 7, 7.5, 8, 9]
    g_vals = [1, 2, 3, 3.5, 4, 5, 6]
    nu_vals = [0.5]  # Found through HP Testing
    multi = False
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
    for dataset_num in datasets:
        data_name, data_creation_func, eps = get_dataset_info(dataset_num)
        """gen.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, base_train, base_k,
                               base_g,
                               nu_vals, data_name, data_creation_func,
                               eps, h_model, multi, seed, [], True, 0.1, 0, False)

        add_ucb.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, base_train,
                                   base_k, nu_vals, data_name, data_creation_func,
                                   eps, multi, seed, True, 0.1, False)
        laf.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, base_train, base_k,
                               base_g,
                               nu_vals, data_name, data_creation_func,
                               eps, h_model, multi, seed, [], True, 0.1, False, )
        van.training_procedure(nbr_query, nbr_repetition, 1, dimension, base_train, base_k,
                               data_name, data_creation_func,
                               eps, seed, False)"""
        dgp.training_procedure(nbr_query, nbr_repetition, 1, dimension, base_train, base_k,
                               data_name, data_creation_func,
                               eps, seed, depth=3, disable_tqdm=False)

        print("======================================================")
        print(f"Dataset {data_name} complete")
        print("======================================================")

        """# Kappa block
        for kappa in k_vals:
            for dataset_num in datasets:
                data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                gen.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, base_train, [kappa],
                                       base_g,
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, [], True, 0.1, 0, True)
                laf.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, base_train, [kappa],
                                       base_g,
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, [], True, 0.1, True, )
                van.training_procedure(nbr_query, nbr_repetition, 1, dimension, base_train, [kappa],
                                       data_name, data_creation_func,
                                       eps, seed, True)
                dgp.training_procedure(nbr_query, nbr_repetition, 1, dimension, base_train, [kappa],
                                       data_name, data_creation_func,
                                       eps, seed, depth=2, disable_tqdm=True)
                add_ucb.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, base_train,
                                           [kappa], nu_vals, data_name, data_creation_func,
                                           eps, multi, seed, True, 0.1, True)


        print(f"\n=====================================================")
        print(f"Kappa Complete")
        print(f"=====================================================\n")
        # Gamme Block
        for gamma in g_vals:
            for dataset_num in datasets:
                data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                gen.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, base_train, base_k,
                                       [gamma],
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, [], True, 0.1, 0, True)
                laf.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, base_train, base_k,
                                       [gamma],
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, [], True, 0.1, True, )
        print(f"\n=====================================================")
        print(f"Gamma Complete")
        print(f"=====================================================\n")
        # Rand Init Block
        for rand_init in nbr_rand_init:
            for dataset_num in datasets:
                data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                gen.training_procedure(nbr_query, nbr_repetition, rand_init, dimension, base_train, base_k,
                                       base_g,
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, [], True, 0.1, 0, True)
                laf.training_procedure(nbr_query, nbr_repetition, rand_init, dimension, base_train, base_k,
                                       base_g,
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, [], True, 0.1, True, )
                add_ucb.training_procedure(nbr_query, nbr_repetition, rand_init, dimension, base_train,
                                           base_k, nu_vals, data_name, data_creation_func,
                                           eps, multi, seed, True, 0.1, True)
        print(f"\n=====================================================")
        print(f"Rand Init Complete")
        print(f"=====================================================\n")
        # Train Iteration block

        for train_iter in training_iter:
            for dataset_num in datasets:
                data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                gen.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, train_iter, base_k,
                                       base_g,
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, [], True, 0.1, 0, True)
                laf.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, train_iter, base_k,
                                       base_g,
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, [], True, 0.1, True, )
                van.training_procedure(nbr_query, nbr_repetition, 1, dimension, train_iter, base_k,
                                       data_name, data_creation_func,
                                       eps, seed, True)
                dgp.training_procedure(nbr_query, nbr_repetition, 1, dimension, train_iter, base_k,
                                       data_name, data_creation_func,
                                       eps, seed, depth=2, disable_tqdm=True)
                add_ucb.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, train_iter,
                                           base_k, nu_vals, data_name, data_creation_func,
                                           eps, multi, seed, True, 0.1, True)


        print(f"\n=====================================================")
        print(f"Training Iter Complete")
        print(f"=====================================================\n")"""

