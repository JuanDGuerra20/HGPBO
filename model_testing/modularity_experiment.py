from efficient_general_2d import *

if __name__ == '__main__':
    dimension = 10
    nbr_query = 80
    training_iter = 5
    nbr_repetition = 1
    nbr_rand_init = 5
    k_vals = [2]
    g_vals = [6]
    nu_vals = [0.5]
    multi = False
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP

    # Setting up for dataset number 1
    data_name, data_creation_func, eps = get_dataset_info(2)

    parent_1 = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                                  nu_vals, data_name, data_creation_func, eps, h, multi)

    parent_1 = parent_1[0][1]
    # Setting up for dataset number 2
    data_name, data_creation_func, eps = get_dataset_info(dataset_num)

    parent_2 = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                                  nu_vals, data_name, data_creation_func, eps, h, multi)

    parent_2 = parent_2[0][1]

    child_11, child_12 = parent_1.sub_models

    child_21, child_22 = parent_2.sub_models

