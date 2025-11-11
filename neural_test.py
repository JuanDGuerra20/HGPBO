from dataset_actions import *
import general_neural as gen
import laferriere_script as laf
import vanilla_script as van
import update_hmodel as hmodel
import multiprocessing as mp

if __name__ == "__main__":
    max_seen_resp_2D = 0
    max_seen_resp_1_1D = 0
    max_seen_resp_2_1D = 0
    list_prior_map = []
    list_objective_mean_map = []

    trainsC = Trains(clean_thresh=0.06)
    X_1D, Y_1D, Xmean_1D, Ymean_1D = make_dataset_1d(trainsC)
    test_x_1D = torch.tensor(Xmean_1D)
    test_y_1D = torch.tensor(Ymean_1D)
    ground_truth_max_1D = np.max(Ymean_1D)

    x_sub1 = torch.from_numpy(X_1D.copy())
    x_sub2 = torch.from_numpy(X_1D.copy())

    y_sub1 = torch.from_numpy(Y_1D[:, 0].copy())
    y_sub2 = torch.from_numpy(Y_1D[:, 0].copy())

    X_2D, Y_2D, Xmean_2D, Ymean_2D = make_dataset_2d(trainsC)

    ground_truth_max_2D = np.max(Ymean_2D)

    x_hier = torch.from_numpy(X_2D.copy())

    y_hier = torch.from_numpy(Y_2D[:, 0].copy())

    # trainsC.plot_response_matrix()
    test_x_hier = torch.tensor(Xmean_2D)
    test_y_hier = torch.tensor(Ymean_2D)

    nbr_query = 100
    training_iter = 10
    nbr_repetition = 30
    nbr_rand_init = 3
    k_vals = [4]  # Found through HP Testing
    g_vals = [3]  # Found through HP Testing
    nu_vals = [0.5]
    multi = False
    hierarchical_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    process = []
    seed = np.array([85153, 48849, 68588, 89899, 60426, 53629, 85382, 61427, 75115,
       64493, 56460, 55906, 78223, 47519, 41817, 48206,  4283, 57623,
       38230, 79533, 19244,  1846,  2211, 44014, 64025, 69297, 98026,
       28352, 35718, 94998])
    #seed = np.random.randint(999999, size=nbr_repetition)

    gen.training_procedure(nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals,
                             hierarchical_model, multi, seed, [], True,)

    """with mp.Pool(processes=3) as pool:
        p = pool.apply_async(gen.training_procedure, (nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals, hierarchical_model, multi, seed, [], True,))
        process.append(p)

        p = pool.apply_async(laf.training_procedure,
                             (nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals, g_vals, nu_vals,
                              hierarchical_model, multi, seed, [], True,))
        process.append(p)

        p = pool.apply_async(van.training_procedure,
                             (nbr_query, nbr_repetition, nbr_rand_init, training_iter, k_vals,))
        process.append(p)

        for i, proc in enumerate(process):
            try:
                proc.get()
            except:
                print(f"process {i} failed")"""