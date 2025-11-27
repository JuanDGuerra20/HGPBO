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
    training_iter = [5, 10, 15, 20]  # Found through HP Testing
    nbr_repetition = 10
    base_k = [4]
    base_g = [3]
    base_train = 10
    base_rand = 3
    k_vals = [4, 5, 6, 7, 8, 9]
    g_vals = [1, 2, 3, 4, 5, 6]
    nu_vals = [0.5]  # Found through HP Testing
    multi = True
    h_model = [hmodel.Lossless_Efficient_UCB_Hierarchical_GP]
    process = []
    nbr_rand_init = [1, 2, 3, 5, 10, 15, 20]
    seed = np.array([791104038, 558883516,  75533178, 730586104,  64343038, 353199330,
       138876529, 594536092, 713725275, 642158682, 287397414, 156569942,
       554978049, 860858855, 899218178])
    #seed = np.random.randint(999999, size=nbr_repetition)
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
    with mp.Pool(processes=10) as pool:
        for h in h_model:
            # Kappa block
            for kappa in k_vals:
                gen.training_procedure(nbr_query, nbr_repetition, base_rand, base_train, [kappa], base_g, nu_vals, h, multi, seed)
                laf.training_procedure(nbr_query, nbr_repetition, base_rand, base_train, [kappa], base_g, nu_vals, h, multi, seed)
                van.training_procedure(nbr_query, nbr_repetition, 1, base_train, [kappa], seed)


            print(f"\n=====================================================")
            print(f"Kappa Complete")
            print(f"=====================================================\n")
            # Gamme Block
            for gamma in g_vals:
                gen.training_procedure(nbr_query, nbr_repetition, base_rand, base_train, base_k, [gamma], nu_vals, h,
                                       multi, seed)
                laf.training_procedure(nbr_query, nbr_repetition, base_rand, base_train, base_k, [gamma], nu_vals, h,
                                       multi, seed)
            print(f"\n=====================================================")
            print(f"Gamma Complete")
            print(f"=====================================================\n")
            # Rand Init Block
            for rand_init in nbr_rand_init:
                gen.training_procedure(nbr_query, nbr_repetition, rand_init, base_train, base_k, base_g, nu_vals, h,
                                       multi, seed)
                laf.training_procedure(nbr_query, nbr_repetition, rand_init, base_train, base_k, base_g, nu_vals, h,
                                       multi, seed)
            print(f"\n=====================================================")
            print(f"Rand Init Complete")
            print(f"=====================================================\n")
            # Train Iteration block

            for train_iter in training_iter:
                gen.training_procedure(nbr_query, nbr_repetition, base_rand, train_iter, base_k, base_g, nu_vals, h,
                                       multi, seed)
                laf.training_procedure(nbr_query, nbr_repetition, base_rand, train_iter, base_k, base_g, nu_vals, h,
                                       multi, seed)
                van.training_procedure(nbr_query, nbr_repetition, 1, train_iter, base_k, seed)
            print(f"\n=====================================================")
            print(f"Training Iter Complete")
            print(f"=====================================================\n")

