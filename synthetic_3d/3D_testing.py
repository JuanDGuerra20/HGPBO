import ucb_efficient_general as gen
import laferriere as laf
import vanilla_3D_general as van
import deep_gp_3d as deep_gp
import add_gp_ucb_3d as add_ucb
import hmodel_3d as hmodel
from dataset_actions_3d import *


if __name__ == '__main__':
    dimension = 10
    nbr_query = 100
    training_iter = [5, 10, 15, 20]  # Found through HP Testing
    nbr_repetition = 10
    base_k = [7]
    base_g = [3]
    base_train = 10
    base_rand = 3
    k_vals = [4, 5, 6, 7, 8, 9]
    g_vals = [1, 2, 3, 4, 5, 6]
    nu_vals = [0.5]  # Found through HP Testing
    multi = True
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
    datasets = [6]
    processes = []
    for h in h_model:
        # Kappa block
        #for kappa in k_vals:
        for dataset_num in datasets:
            data_name, data_creation_func, eps = get_dataset_info(dataset_num)
            """gen.training_procedure(nbr_query, nbr_repetition, 2, dimension, 20, [8], [5], nu_vals, data_name, data_creation_func,
                   eps, h, multi, seed, children=[], visualize=True, noise=0.1, acq_func="ucb")
            gen.training_procedure(nbr_query, nbr_repetition, 2, dimension, 20, [8], [5], nu_vals, data_name,
                                   data_creation_func,
                                   eps, h, multi, seed, children=[], visualize=True, noise=0.1, acq_func="ei")
            gen.training_procedure(nbr_query, nbr_repetition, 2, dimension, 20, [8], [5], nu_vals, data_name, data_creation_func,
                   eps, h, multi, seed, children=[], visualize=True, noise=0.1, acq_func="pi")
            laf.training_procedure(nbr_query, nbr_repetition, 3, dimension, 5, [7], [3], nu_vals, data_name, data_creation_func,
                   eps, h, multi, seed, children=[], visualize=True, noise=0.1)
            van.training_procedure(nbr_query, nbr_repetition, 1, dimension, 10, [6], data_name, data_creation_func,
                   eps, 0.1)
            deep_gp.training_procedure(nbr_query, nbr_repetition, 1, dimension, 100,
                   [6], data_name, data_creation_func, eps, seed=seed, multi=multi)"""
            add_ucb.training_procedure(nbr_query, nbr_repetition, 1, dimension, 10,
                   [6], nu_vals, data_name, data_creation_func, eps, multi, seed)

        """print(f"\n=====================================================")
        print(f"Kappa Complete")
        print(f"=====================================================\n")
        # Gamme Block
        for gamma in g_vals:
            for dataset_num in datasets:
                data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                gen.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, base_train, base_k, [gamma],
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, children=[], visualize=True, noise=0.1)
                laf.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, base_train, base_k, [gamma],
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, children=[], visualize=True, noise=0.1)
        print(f"\n=====================================================")
        print(f"Gamma Complete")
        print(f"=====================================================\n")
        # Rand Init Block
        for rand_init in nbr_rand_init:
            for dataset_num in datasets:
                data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                gen.training_procedure(nbr_query, nbr_repetition, rand_init, dimension, base_train, base_k, base_g,
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, children=[], visualize=True, noise=0.1)
                laf.training_procedure(nbr_query, nbr_repetition, rand_init, dimension, base_train, base_k, base_g,
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, children=[], visualize=True, noise=0.1)
                deep_gp.training_procedure(nbr_query, nbr_repetition, rand_init, dimension, base_train,
                                           base_k, data_name, data_creation_func, eps, seed=seed, multi=multi)
                add_ucb.training_procedure(nbr_query, nbr_repetition, rand_init, dimension, base_train,
                                           base_k, nu_vals, data_name, data_creation_func, eps, multi, seed)
        print(f"\n=====================================================")
        print(f"Rand Init Complete")
        print(f"=====================================================\n")
        # Train Iteration block

        for train_iter in training_iter:
            for dataset_num in datasets:
                data_name, data_creation_func, eps = get_dataset_info(dataset_num)
                gen.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, train_iter, base_k, base_g,
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, children=[], visualize=True, noise=0.1)
                laf.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, train_iter, base_k, base_g,
                                       nu_vals, data_name, data_creation_func,
                                       eps, h, multi, seed, children=[], visualize=True, noise=0.1)
                van.training_procedure(nbr_query, nbr_repetition, 1, dimension, train_iter, base_k, data_name,
                                       data_creation_func,
                                       eps, 0.1)
                deep_gp.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, train_iter,
                                           base_k, data_name, data_creation_func, eps, seed=seed, multi=multi)
                add_ucb.training_procedure(nbr_query, nbr_repetition, base_rand, dimension, train_iter,
                                           base_k, nu_vals, data_name, data_creation_func, eps, multi, seed)

        print(f"\n=====================================================")
        print(f"Training Iter Complete")
        print(f"=====================================================\n")"""