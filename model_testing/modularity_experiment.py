import multiprocessing as mp


from efficient_general_2d import *

def two_pretrained(k_vals, g_vals, nu_vals):
    dimension = 15
    nbr_query = 100
    training_iter = 10
    nbr_repetition = 10
    nbr_rand_init = 6

    multi = False
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    seed = [False]*nbr_repetition
    noise = 0.1

    avg_explor_1 = []
    avg_exploit_1 = []
    avg_r2_1 = []
    avg_r2_c1_1 = []
    avg_r2_c2_1 = []

    avg_explor_2 = []
    avg_exploit_2 = []
    avg_r2_2 = []
    avg_r2_c1_2 = []
    avg_r2_c2_2 = []

    avg_explor_mod = []
    avg_exploit_mod = []
    avg_r2_mod = []
    avg_r2_c1_mod = []
    avg_r2_c2_mod = []

    for i in range(nbr_repetition):
        # Setting up for dataset number 1
        data_name, data_creation_func, eps = get_dataset_info(2)
        data_name = "modular_2D"

        try:
            func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                       noise=noise, visualize=False)[0]
        except:
            try:
                func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                       noise=noise, visualize=False)[0]
            except:
                try:
                    func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                seed,
                                                noise=noise, visualize=False)[0]
                except:
                    try:
                        func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed,
                                                    noise=noise, visualize=False)[0]
                    except:
                        func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed,
                                                    noise=noise, visualize=False)[0]

        name_1, parent_1, explor_1, exploit_1, r2_1, child_1_r2, child_2_r2 = func_1

        avg_explor_1.append(np.mean(explor_1, axis=0))
        avg_exploit_1.append(np.mean(exploit_1, axis=0))
        avg_r2_1.append(r2_1)
        avg_r2_c1_1.append(child_1_r2)
        avg_r2_c2_1.append(child_2_r2)

        # Setting up for dataset number 2
        data_name, data_creation_func, eps = get_dataset_info(3)
        data_name = "modular_2D"

        try:
            func_2 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                       noise=noise, visualize=False)[0]
        except:
            try:
                func_2 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                       noise=noise, visualize=False)[0]
            except:
                try:
                    func_2 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                seed,
                                                noise=noise, visualize=False)[0]
                except:
                    try:
                        func_2 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed,
                                                    noise=noise, visualize=False)[0]
                    except:
                        func_2 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed,
                                                    noise=noise, visualize=False)[0]
        name_2, parent_2, explor_2, exploit_2, r2_2, child_1_r2, child_2_r2 = func_2

        avg_explor_2.append(np.mean(explor_2, axis=0))
        avg_exploit_2.append(np.mean(exploit_2, axis=0))
        avg_r2_2.append(r2_2)

        avg_r2_c1_2.append(child_1_r2)
        avg_r2_c2_2.append(child_2_r2)

        child_11, child_12 = parent_1.sub_models

        child_21, child_22 = parent_2.sub_models

        # =======================================================
        # Modular Section

        modular_children = [child_11, child_21]

        data_name, data_creation_func, eps = get_dataset_info(6)
        data_name = "modular_2D"

        try:
            func_mod = training_procedure(nbr_query, 1, 0, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                       children=modular_children, noise=noise, visualize=False)[0]
            name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

            avg_explor_mod.append(np.mean(explor_mod, axis=0))
            avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
            avg_r2_mod.append(r2_mod)

            avg_r2_c1_mod.append(child_1_r2)
            avg_r2_c2_mod.append(child_2_r2)
        except:
            try:
                func_mod = training_procedure(nbr_query, 1, 0, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                       children=modular_children, noise=noise, visualize=False)[0]
                name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                avg_explor_mod.append(np.mean(explor_mod, axis=0))
                avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                avg_r2_mod.append(r2_mod)

                avg_r2_c1_mod.append(child_1_r2)
                avg_r2_c2_mod.append(child_2_r2)
            except:
                try:
                    func_mod = training_procedure(nbr_query, 1, 0, dimension, training_iter, k_vals,
                                                  g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                  seed,
                                                  children=modular_children, noise=noise, visualize=False)[0]
                    name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                    avg_explor_mod.append(np.mean(explor_mod, axis=0))
                    avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                    avg_r2_mod.append(r2_mod)

                    avg_r2_c1_mod.append(child_1_r2)
                    avg_r2_c2_mod.append(child_2_r2)
                except:
                    try:
                        func_mod = training_procedure(nbr_query, 1, 0, dimension, training_iter, k_vals,
                                                      g_vals, nu_vals, data_name, data_creation_func, eps, h_model,
                                                      multi, seed,
                                                      children=modular_children, noise=noise, visualize=False)[0]
                        name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                        avg_explor_mod.append(np.mean(explor_mod, axis=0))
                        avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                        avg_r2_mod.append(r2_mod)

                        avg_r2_c1_mod.append(child_1_r2)
                        avg_r2_c2_mod.append(child_2_r2)
                    except:
                        try:
                            func_mod = training_procedure(nbr_query, 1, 0, dimension, training_iter, k_vals,
                                                          g_vals, nu_vals, data_name, data_creation_func, eps, h_model,
                                                          multi, seed,
                                                          children=modular_children, noise=noise, visualize=False)[0]
                            name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                            avg_explor_mod.append(np.mean(explor_mod, axis=0))
                            avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                            avg_r2_mod.append(r2_mod)

                            avg_r2_c1_mod.append(child_1_r2)
                            avg_r2_c2_mod.append(child_2_r2)
                        except:
                            try:
                                func_mod = training_procedure(nbr_query, 1, 0, dimension, training_iter, k_vals,
                                                              g_vals, nu_vals, data_name, data_creation_func, eps,
                                                              h_model,
                                                              multi, seed,
                                                              children=modular_children, noise=noise, visualize=False)[
                                    0]
                                name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                                avg_explor_mod.append(np.mean(explor_mod, axis=0))
                                avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                                avg_r2_mod.append(r2_mod)

                                avg_r2_c1_mod.append(child_1_r2)
                                avg_r2_c2_mod.append(child_2_r2)
                            except:
                                continue



    if h_model == hmodel.Efficient_UCB_Hierarchical_GP:
        model_name = "Efficient"

    elif h_model == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
        model_name = "Lossless_Efficient"

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)

    k = str(k_vals[0]).replace('.', ',')
    g = str(g_vals[0]).replace('.', ',')
    n = str(nu_vals[0]).replace('.', ',')
    print("reached the mean section")
    avg_explor_1 = np.mean(avg_explor_1, axis=0)
    avg_exploit_1 = np.mean(avg_exploit_1, axis=0)
    avg_r2_1 = np.mean(avg_r2_1, axis=0)

    avg_explor_2 = np.mean(avg_explor_2, axis=0)
    avg_exploit_2 = np.mean(avg_exploit_2, axis=0)
    avg_r2_2 = np.mean(avg_r2_2, axis=0)

    avg_explor_mod = np.mean(avg_explor_mod, axis=0)
    avg_exploit_mod = np.mean(avg_exploit_mod, axis=0)
    avg_r2_mod = np.mean(avg_r2_mod, axis=0)

    avg_r2_c1_1 = np.mean(avg_r2_c1_1, axis=0)
    avg_r2_c1_2 = np.mean(avg_r2_c1_2, axis=0)
    avg_r2_c2_1 = np.mean(avg_r2_c2_1, axis=0)
    avg_r2_c2_2 = np.mean(avg_r2_c2_2, axis=0)

    avg_r2_c1_mod = np.mean(avg_r2_c1_mod, axis=0)
    avg_r2_c2_mod = np.mean(avg_r2_c2_mod, axis=0)

    plt.plot(np.insert(avg_explor_1, 0, np.zeros(2*nbr_rand_init))[:nbr_query], label='Exploration 1')
    plt.plot(np.insert(avg_explor_2, 0, np.zeros(2*nbr_rand_init))[:nbr_query], label='Exploration 2')
    plt.plot(avg_explor_mod[:nbr_query], label='Exploration Modular')
    plt.ylim(0, 1.1)


    plt.xlabel('Query Number')
    plt.ylabel('Performance')

    plt.legend()
    plt.title(f'2D Modularity 2 Children Experiment Exploration with {nbr_repetition} repetitions')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_2_children_exploration_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_2_children_exploration_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

    plt.plot(avg_r2_1, label='R2 1')
    plt.plot(avg_r2_2, label='R2 2')
    plt.plot(avg_r2_mod, label='R2 Modular')
    plt.ylim(0, 1.1)


    plt.xlabel('Query Number')
    plt.ylabel('Performance')

    plt.legend()
    plt.title(f'2D Modularity 2 Children Experiment R2 with {nbr_repetition} repetitions')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_2_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_2_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

    plt.plot((avg_r2_c1_mod + avg_r2_c2_mod) / 2, label='Modular Avg Child R2')
    plt.plot((avg_r2_c1_1 + avg_r2_c2_1) / 2, label='Parent 1 Avg Child R2')
    plt.plot((avg_r2_c1_2 + avg_r2_c2_2) / 2, label='Parent 2 Avg Child R2')
    plt.ylim(0, 1.1)


    plt.legend()
    plt.title(f'Children R2 Scores with {nbr_repetition} repetitions')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_2_children_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_2_children_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()
    
    noi = str(noise).replace('.', ',')
    # modular
    df = pd.DataFrame([f'mod_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                          avg_explor_mod, avg_exploit_mod, avg_r2_mod, avg_r2_c1_mod, avg_r2_c2_mod])
    df.index = ['name', 'exploration_score', 'exploitation_score', 'parent_r2', 'child1_r2', 'child2_r2']
    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/modular_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}_noise_{noi}.csv')
    
    # parent 1
    df = pd.DataFrame([f'parent1_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                          avg_explor_1, avg_exploit_1, avg_r2_1, avg_r2_c1_1, avg_r2_c2_1])
    df.index = ['name', 'exploration_score', 'exploitation_score', 'parent_r2', 'child1_r2', 'child2_r2']

    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/parent1_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}_noise_{noi}.csv')

    # parent 1
    df = pd.DataFrame([f'parent2_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                          avg_explor_2, avg_exploit_2, avg_r2_2, avg_r2_c1_2, avg_r2_c2_2])
    df.index = ['name', 'exploration_score', 'exploitation_score', 'parent_r2', 'child1_r2', 'child2_r2']

    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/parent2_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}_noise_{noi}.csv')


def two_pretrained_bad(k_vals, g_vals, nu_vals):
    dimension = 15
    nbr_query = 100
    training_iter = 10
    nbr_repetition = 10
    nbr_rand_init = 6
    multi = False
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    seed = [False]*nbr_repetition
    noise = 0.1

    avg_explor_1 = []
    avg_exploit_1 = []
    avg_r2_1 = []
    avg_r2_c1_1 = []
    avg_r2_c1_2 = []

    avg_explor_2 = []
    avg_exploit_2 = []
    avg_r2_2 = []
    avg_r2_c2_1 = []
    avg_r2_c2_2 = []

    avg_explor_mod = []
    avg_exploit_mod = []
    avg_r2_mod = []
    avg_r2_c1_mod = []
    avg_r2_c2_mod = []

    for i in range(nbr_repetition):
        # Setting up for dataset number 1
        data_name, data_creation_func, eps = get_dataset_info(2)
        data_name = 'bad_modular_2D'

        try:
            func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed, noise=noise, visualize=False)[0]
        except:
            try:
                func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed, noise=noise, visualize=False)[0]
            except:
                try:
                    func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                seed, noise=noise, visualize=False)[0]
                except:
                    try:
                        func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed, noise=noise, visualize=False)[0]
                    except:
                        func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed, noise=noise, visualize=False)[0]

        name_1, parent_1, explor_1, exploit_1, r2_1, child_1_r2, child_2_r2 = func_1

        avg_explor_1.append(np.mean(explor_1, axis=0))
        avg_exploit_1.append(np.mean(exploit_1, axis=0))
        avg_r2_1.append(r2_1)
        avg_r2_c1_1.append(child_1_r2)
        avg_r2_c2_1.append(child_2_r2)

        # Setting up for dataset number 2
        data_name, data_creation_func, eps = get_dataset_info(3)
        data_name = 'bad_modular_2D'

        try:
            func_2 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed, noise=noise, visualize=False)[0]
        except:
            try:
                func_2 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed, noise=noise, visualize=False)[0]
            except:
                try:
                    func_2 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                seed, noise=noise, visualize=False)[0]
                except:
                    try:
                        func_2 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed, noise=noise, visualize=False)[0]
                    except:
                        func_2 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed, noise=noise, visualize=False)[0]
        name_2, parent_2, explor_2, exploit_2, r2_2, child_1_r2, child_2_r2 = func_2

        avg_explor_2.append(np.mean(explor_2, axis=0))
        avg_exploit_2.append(np.mean(exploit_2, axis=0))
        avg_r2_2.append(r2_2)

        avg_r2_c1_2.append(child_1_r2)
        avg_r2_c2_2.append(child_2_r2)

        child_11, child_12 = parent_1.sub_models

        child_21, child_22 = parent_2.sub_models

        # =======================================================
        # Modular Section

        modular_children = [child_22, child_12]

        data_name, data_creation_func, eps = get_dataset_info(6)
        data_name = 'bad_modular_2D'

        try:
            func_mod = training_procedure(nbr_query, 1, 0, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                       children=modular_children, noise=noise, visualize=False)[0]
            name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

            avg_explor_mod.append(np.mean(explor_mod, axis=0))
            avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
            avg_r2_mod.append(r2_mod)
            avg_r2_c1_mod.append(child_1_r2)
            avg_r2_c2_mod.append(child_2_r2)
        except:
            try:
                func_mod = training_procedure(nbr_query, 1, 0, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                       children=modular_children, noise=noise, visualize=False)[0]
                name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                avg_explor_mod.append(np.mean(explor_mod, axis=0))
                avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                avg_r2_mod.append(r2_mod)
                avg_r2_c1_mod.append(child_1_r2)
                avg_r2_c2_mod.append(child_2_r2)
            except:
                try:
                    func_mod = training_procedure(nbr_query, 1, 0, dimension, training_iter, k_vals,
                                                  g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                  seed,
                                                  children=modular_children, noise=noise, visualize=False)[0]
                    name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                    avg_explor_mod.append(np.mean(explor_mod, axis=0))
                    avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                    avg_r2_mod.append(r2_mod)
                    avg_r2_c1_mod.append(child_1_r2)
                    avg_r2_c2_mod.append(child_2_r2)
                except:
                    try:
                        func_mod = training_procedure(nbr_query, 1, 0, dimension, training_iter, k_vals,
                                                      g_vals, nu_vals, data_name, data_creation_func, eps, h_model,
                                                      multi, seed,
                                                      children=modular_children, noise=noise, visualize=False)[0]
                        name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                        avg_explor_mod.append(np.mean(explor_mod, axis=0))
                        avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                        avg_r2_mod.append(r2_mod)
                        avg_r2_c1_mod.append(child_1_r2)
                        avg_r2_c2_mod.append(child_2_r2)
                    except:
                        func_mod = training_procedure(nbr_query, 1, 0, dimension, training_iter, k_vals,
                                                      g_vals, nu_vals, data_name, data_creation_func, eps, h_model,
                                                      multi, seed,
                                                      children=modular_children, noise=noise, visualize=False)[0]

                        name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                        avg_explor_mod.append(np.mean(explor_mod, axis=0))
                        avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                        avg_r2_mod.append(r2_mod)
                        avg_r2_c1_mod.append(child_1_r2)
                        avg_r2_c2_mod.append(child_2_r2)

    if h_model == hmodel.Efficient_UCB_Hierarchical_GP:
        model_name = "Efficient"

    elif h_model == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
        model_name = "Lossless_Efficient"

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)

    k = str(k_vals[0]).replace('.', ',')
    g = str(g_vals[0]).replace('.', ',')
    n = str(nu_vals[0]).replace('.', ',')
    noi = str(noise).replace('.', ',')

    avg_explor_1 = np.mean(avg_explor_1, axis=0)
    avg_exploit_1 = np.mean(avg_exploit_1, axis=0)
    avg_r2_1 = np.mean(avg_r2_1, axis=0)

    avg_explor_2 = np.mean(avg_explor_2, axis=0)
    avg_exploit_2 = np.mean(avg_exploit_2, axis=0)
    avg_r2_2 = np.mean(avg_r2_2, axis=0)

    avg_explor_mod = np.mean(avg_explor_mod, axis=0)
    avg_exploit_mod = np.mean(avg_exploit_mod, axis=0)
    avg_r2_mod = np.mean(avg_r2_mod, axis=0)

    avg_r2_c1_1 = np.mean(avg_r2_c1_1, axis=0)
    avg_r2_c1_2 = np.mean(avg_r2_c1_2, axis=0)
    avg_r2_c2_1 = np.mean(avg_r2_c2_1, axis=0)
    avg_r2_c2_2 = np.mean(avg_r2_c2_2, axis=0)
    avg_r2_c1_mod = np.mean(avg_r2_c1_mod, axis=0)
    avg_r2_c2_mod = np.mean(avg_r2_c2_mod, axis=0)

    plt.plot(np.insert(avg_explor_1, 0, np.zeros(2 * nbr_rand_init))[:nbr_query], label='Exploration 1')
    plt.plot(np.insert(avg_explor_2, 0, np.zeros(2 * nbr_rand_init))[:nbr_query], label='Exploration 2')
    plt.plot(avg_explor_mod[:nbr_query], label='Exploration Modular')
    plt.ylim(0, 1.1)
    print(f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/')
    plt.xlabel('Query Number')
    plt.ylabel('Performance')
    plt.legend()
    plt.title(f'2D BAD Modularity 2 Children Experiment Exploration with {nbr_repetition} repetitions')
    plt.ylim(0, 1.1)

    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_2_children_exploration_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_2_children_exploration_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

    plt.plot(avg_r2_1, label='R2 1')
    plt.plot(avg_r2_2, label='R2 2')
    plt.plot(avg_r2_mod, label='R2 Modular')
    plt.ylim(0, 1.1)

    plt.xlabel('Query Number')
    plt.ylabel('Performance')

    plt.legend()
    plt.title(f'2D BAD Modularity 2 Children Experiment R2 with {nbr_repetition} repetitions')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_2_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_2_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

    plt.plot((avg_r2_c1_mod + avg_r2_c2_mod) / 2, label='Modular Avg Child R2')
    plt.plot((avg_r2_c1_1 + avg_r2_c2_1) / 2, label='Parent 1 Avg Child R2')
    plt.plot((avg_r2_c1_2 + avg_r2_c2_2) / 2, label='Parent 2 Avg Child R2')

    plt.ylim(0, 1.1)

    plt.legend()
    plt.title(f'BAD Children R2 Scores with {nbr_repetition} repetitions')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_2_children_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_2_children_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

    # modular
    df = pd.DataFrame([
                          f'mod_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                          avg_explor_mod, avg_exploit_mod, avg_r2_mod, avg_r2_c1_mod, avg_r2_c2_mod])
    df.index = ['name', 'exploration_score', 'exploitation_score', 'parent_r2', 'child1_r2', 'child2_r2']
    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/bad_modular_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}_noise_{noi}.csv')

    # parent 1
    df = pd.DataFrame([
                          f'parent1_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                          avg_explor_1, avg_exploit_1, avg_r2_1, avg_r2_c1_1, avg_r2_c2_1])
    df.index = ['name', 'exploration_score', 'exploitation_score', 'parent_r2', 'child1_r2', 'child2_r2']

    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/parent1_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}_noise_{noi}.csv')

    # parent 1
    df = pd.DataFrame([
                          f'parent2_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                          avg_explor_2, avg_exploit_2, avg_r2_2, avg_r2_c1_2, avg_r2_c2_2])
    df.index = ['name', 'exploration_score', 'exploitation_score', 'parent_r2', 'child1_r2', 'child2_r2']

    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/parent2_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}_noise_{noi}.csv')


def one_pretrained(k_vals, g_vals, nu_vals):
    dimension = 15
    nbr_query = 100
    training_iter = 10
    nbr_repetition = 10
    nbr_rand_init = 6

    multi = False
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    seed = [False] * nbr_repetition
    noise = 0.1

    avg_explor_1 = []
    avg_exploit_1 = []
    avg_r2_1 = []
    avg_r2_c1_1 = []
    avg_r2_c2_1 = []

    avg_explor_mod = []
    avg_exploit_mod = []
    avg_r2_mod = []
    avg_r2_c1_mod = []
    avg_r2_c2_mod = []

    for i in range(nbr_repetition):
        # Setting up for dataset number 1
        data_name, data_creation_func, eps = get_dataset_info(2)
        data_name = "modular_1_child"

        try:
            func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                        g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                        noise=noise, visualize=False)[0]
        except:
            try:
                func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                            g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                            noise=noise, visualize=False)[0]
            except:
                try:
                    func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                seed,
                                                noise=noise, visualize=False)[0]
                except:
                    try:
                        func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed,
                                                    noise=noise, visualize=False)[0]
                    except:
                        func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed,
                                                    noise=noise, visualize=False)[0]

        name_1, parent_1, explor_1, exploit_1, r2_1, child_1_r2, child_2_r2 = func_1

        avg_explor_1.append(np.mean(explor_1, axis=0))
        avg_exploit_1.append(np.mean(exploit_1, axis=0))
        avg_r2_1.append(r2_1)
        avg_r2_c1_1.append(child_1_r2)
        avg_r2_c2_1.append(child_2_r2)

        child_11, child_12 = parent_1.sub_models


        # =======================================================
        # Modular Section

        modular_children = [child_11]

        data_name, data_creation_func, eps = get_dataset_info(6)
        data_name = "modular_1_child"


        try:
            func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                          g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                          children=modular_children, noise=noise, visualize=False)[0]
            name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

            avg_explor_mod.append(np.mean(explor_mod, axis=0))
            avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
            avg_r2_mod.append(r2_mod)

            avg_r2_c1_mod.append(child_1_r2)
            avg_r2_c2_mod.append(child_2_r2)
        except:
            try:
                func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                              g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                              children=modular_children, noise=noise, visualize=False)[0]
                name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                avg_explor_mod.append(np.mean(explor_mod, axis=0))
                avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                avg_r2_mod.append(r2_mod)

                avg_r2_c1_mod.append(child_1_r2)
                avg_r2_c2_mod.append(child_2_r2)
            except:
                try:
                    func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                  g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                  seed,
                                                  children=modular_children, noise=noise, visualize=False)[0]
                    name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                    avg_explor_mod.append(np.mean(explor_mod, axis=0))
                    avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                    avg_r2_mod.append(r2_mod)

                    avg_r2_c1_mod.append(child_1_r2)
                    avg_r2_c2_mod.append(child_2_r2)
                except:
                    try:
                        func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                      g_vals, nu_vals, data_name, data_creation_func, eps, h_model,
                                                      multi, seed,
                                                      children=modular_children, noise=noise, visualize=False)[0]
                        name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                        avg_explor_mod.append(np.mean(explor_mod, axis=0))
                        avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                        avg_r2_mod.append(r2_mod)

                        avg_r2_c1_mod.append(child_1_r2)
                        avg_r2_c2_mod.append(child_2_r2)
                    except:
                        try:
                            func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                          g_vals, nu_vals, data_name, data_creation_func, eps, h_model,
                                                          multi, seed,
                                                          children=modular_children, noise=noise, visualize=False)[0]
                            name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                            avg_explor_mod.append(np.mean(explor_mod, axis=0))
                            avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                            avg_r2_mod.append(r2_mod)

                            avg_r2_c1_mod.append(child_1_r2)
                            avg_r2_c2_mod.append(child_2_r2)
                        except:
                            try:
                                func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                              g_vals, nu_vals, data_name, data_creation_func, eps,
                                                              h_model,
                                                              multi, seed,
                                                              children=modular_children, noise=noise, visualize=False)[
                                    0]
                                name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                                avg_explor_mod.append(np.mean(explor_mod, axis=0))
                                avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                                avg_r2_mod.append(r2_mod)

                                avg_r2_c1_mod.append(child_1_r2)
                                avg_r2_c2_mod.append(child_2_r2)
                            except:
                                continue

    if h_model == hmodel.Efficient_UCB_Hierarchical_GP:
        model_name = "Efficient"

    elif h_model == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
        model_name = "Lossless_Efficient"

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)

    k = str(k_vals[0]).replace('.', ',')
    g = str(g_vals[0]).replace('.', ',')
    n = str(nu_vals[0]).replace('.', ',')
    print("reached the mean section")
    avg_explor_1 = np.mean(avg_explor_1, axis=0)
    avg_exploit_1 = np.mean(avg_exploit_1, axis=0)
    avg_r2_1 = np.mean(avg_r2_1, axis=0)

    avg_explor_mod = np.mean(avg_explor_mod, axis=0)
    avg_exploit_mod = np.mean(avg_exploit_mod, axis=0)
    avg_r2_mod = np.mean(avg_r2_mod, axis=0)

    avg_r2_c1_1 = np.mean(avg_r2_c1_1, axis=0)
    avg_r2_c2_1 = np.mean(avg_r2_c2_1, axis=0)

    avg_r2_c1_mod = np.mean(avg_r2_c1_mod, axis=0)
    avg_r2_c2_mod = np.mean(avg_r2_c2_mod, axis=0)

    plt.plot(np.insert(avg_explor_1, 0, np.zeros(2 * nbr_rand_init))[:nbr_query], label='Exploration 1')
    plt.plot(avg_explor_mod[:nbr_query], label='Exploration Modular')
    plt.ylim(0, 1.1)

    plt.xlabel('Query Number')
    plt.ylabel('Performance')

    plt.legend()
    plt.title(f'2D Modularity 2 Children Experiment Exploration with {nbr_repetition} repetitions')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_1_children_exploration_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_1_children_exploration_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

    plt.plot(avg_r2_1, label='R2 1')
    plt.plot(avg_r2_mod, label='R2 Modular')
    plt.ylim(0, 1.1)

    plt.xlabel('Query Number')
    plt.ylabel('Performance')

    plt.legend()
    plt.title(f'2D Modularity 2 Children Experiment R2 with {nbr_repetition} repetitions')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_1_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_1_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

    plt.plot((avg_r2_c1_mod + avg_r2_c2_mod)/2, label='Modular Avg Child R2')
    plt.plot((avg_r2_c1_1 + avg_r2_c2_1)/2, label='Parent 1 Avg Child R2')

    plt.ylim(0, 1.1)

    plt.legend()
    plt.title(f'Children R2 Scores with {nbr_repetition} repetitions')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_1_children_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_1_children_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

    noi = str(noise).replace('.', ',')
    # modular
    df = pd.DataFrame([
                          f'mod_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                          avg_explor_mod, avg_exploit_mod, avg_r2_mod, avg_r2_c1_mod, avg_r2_c2_mod])
    df.index = ['name', 'exploration_score', 'exploitation_score', 'parent_r2', 'child1_r2', 'child2_r2']
    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/modular_1_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}_noise_{noi}.csv')

    # parent 1
    df = pd.DataFrame([
                          f'parent1_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                          avg_explor_1, avg_exploit_1, avg_r2_1, avg_r2_c1_1, avg_r2_c2_1])
    df.index = ['name', 'exploration_score', 'exploitation_score', 'parent_r2', 'child1_r2', 'child2_r2']

    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/parent1_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}_noise_{noi}.csv')


def one_pretrained_bad(k_vals, g_vals, nu_vals):
    dimension = 15
    nbr_query = 100
    training_iter = 10
    nbr_repetition = 10
    nbr_rand_init = 6
    multi = False
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    seed = [False] * nbr_repetition
    noise = 0.1

    avg_explor_1 = []
    avg_exploit_1 = []
    avg_r2_1 = []
    avg_r2_c1_1 = []
    avg_r2_c2_1 = []

    avg_explor_mod = []
    avg_exploit_mod = []
    avg_r2_mod = []
    avg_r2_c1_mod = []
    avg_r2_c2_mod = []

    for i in range(nbr_repetition):
        # Setting up for dataset number 1
        data_name, data_creation_func, eps = get_dataset_info(2)
        data_name = "bad_modular_1_child"

        try:
            func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                        g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                        noise=noise, visualize=False)[0]
        except:
            try:
                func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                            g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                            noise=noise, visualize=False)[0]
            except:
                try:
                    func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                seed,
                                                noise=noise, visualize=False)[0]
                except:
                    try:
                        func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed,
                                                    noise=noise, visualize=False)[0]
                    except:
                        func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                    g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                    seed,
                                                    noise=noise, visualize=False)[0]

        name_1, parent_1, explor_1, exploit_1, r2_1, child_1_r2, child_2_r2 = func_1

        avg_explor_1.append(np.mean(explor_1, axis=0))
        avg_exploit_1.append(np.mean(exploit_1, axis=0))
        avg_r2_1.append(r2_1)
        avg_r2_c1_1.append(child_1_r2)
        avg_r2_c2_1.append(child_2_r2)

        child_11, child_12 = parent_1.sub_models


        # =======================================================
        # Modular Section

        modular_children = [child_12]

        data_name, data_creation_func, eps = get_dataset_info(6)
        data_name = "bad_modular_1_child"

        try:
            func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                          g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                          children=modular_children, noise=noise, visualize=False)[0]
            name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

            avg_explor_mod.append(np.mean(explor_mod, axis=0))
            avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
            avg_r2_mod.append(r2_mod)

            avg_r2_c1_mod.append(child_1_r2)
            avg_r2_c2_mod.append(child_2_r2)
        except:
            try:
                func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                              g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                              children=modular_children, noise=noise, visualize=False)[0]
                name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                avg_explor_mod.append(np.mean(explor_mod, axis=0))
                avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                avg_r2_mod.append(r2_mod)

                avg_r2_c1_mod.append(child_1_r2)
                avg_r2_c2_mod.append(child_2_r2)
            except:
                try:
                    func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                  g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi,
                                                  seed,
                                                  children=modular_children, noise=noise, visualize=False)[0]
                    name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                    avg_explor_mod.append(np.mean(explor_mod, axis=0))
                    avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                    avg_r2_mod.append(r2_mod)

                    avg_r2_c1_mod.append(child_1_r2)
                    avg_r2_c2_mod.append(child_2_r2)
                except:
                    try:
                        func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                      g_vals, nu_vals, data_name, data_creation_func, eps, h_model,
                                                      multi, seed,
                                                      children=modular_children, noise=noise, visualize=False)[0]
                        name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                        avg_explor_mod.append(np.mean(explor_mod, axis=0))
                        avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                        avg_r2_mod.append(r2_mod)

                        avg_r2_c1_mod.append(child_1_r2)
                        avg_r2_c2_mod.append(child_2_r2)
                    except:
                        try:
                            func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                          g_vals, nu_vals, data_name, data_creation_func, eps, h_model,
                                                          multi, seed,
                                                          children=modular_children, noise=noise, visualize=False)[0]
                            name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                            avg_explor_mod.append(np.mean(explor_mod, axis=0))
                            avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                            avg_r2_mod.append(r2_mod)

                            avg_r2_c1_mod.append(child_1_r2)
                            avg_r2_c2_mod.append(child_2_r2)
                        except:
                            try:
                                func_mod = \
                                training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals,
                                                   g_vals, nu_vals, data_name, data_creation_func, eps,
                                                   h_model,
                                                   multi, seed,
                                                   children=modular_children, noise=noise, visualize=False)[
                                    0]
                                name_mod, parent_mod, explor_mod, exploit_mod, r2_mod, child_1_r2, child_2_r2 = func_mod

                                avg_explor_mod.append(np.mean(explor_mod, axis=0))
                                avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
                                avg_r2_mod.append(r2_mod)

                                avg_r2_c1_mod.append(child_1_r2)
                                avg_r2_c2_mod.append(child_2_r2)
                            except:
                                continue

    if h_model == hmodel.Efficient_UCB_Hierarchical_GP:
        model_name = "Efficient"

    elif h_model == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
        model_name = "Lossless_Efficient"

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)

    k = str(k_vals[0]).replace('.', ',')
    g = str(g_vals[0]).replace('.', ',')
    n = str(nu_vals[0]).replace('.', ',')
    print("reached the mean section")
    avg_explor_1 = np.mean(avg_explor_1, axis=0)
    avg_exploit_1 = np.mean(avg_exploit_1, axis=0)
    avg_r2_1 = np.mean(avg_r2_1, axis=0)

    avg_explor_mod = np.mean(avg_explor_mod, axis=0)
    avg_exploit_mod = np.mean(avg_exploit_mod, axis=0)
    avg_r2_mod = np.mean(avg_r2_mod, axis=0)

    avg_r2_c1_1 = np.mean(avg_r2_c1_1, axis=0)
    avg_r2_c2_1 = np.mean(avg_r2_c2_1, axis=0)

    avg_r2_c1_mod = np.mean(avg_r2_c1_mod, axis=0)
    avg_r2_c2_mod = np.mean(avg_r2_c2_mod, axis=0)

    plt.plot(np.insert(avg_explor_1, 0, np.zeros(2 * nbr_rand_init))[:nbr_query], label='Exploration 1')
    plt.plot(avg_explor_mod[:nbr_query], label='Exploration Modular')
    plt.ylim(0, 1.1)

    plt.xlabel('Query Number')
    plt.ylabel('Performance')

    plt.legend()
    plt.title(f'2D Modularity 2 Children Experiment Exploration with {nbr_repetition} repetitions')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_1_children_exploration_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_1_children_exploration_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

    plt.plot(avg_r2_1, label='R2 1')
    plt.plot(avg_r2_mod, label='R2 Modular')
    plt.ylim(0, 1.1)

    plt.xlabel('Query Number')
    plt.ylabel('Performance')

    plt.legend()
    plt.title(f'2D Modularity 2 Children Experiment R2 with {nbr_repetition} repetitions')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_1_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_1_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

    plt.plot((avg_r2_c1_mod + avg_r2_c2_mod) / 2, label='Modular Avg Child R2')
    plt.plot((avg_r2_c1_1 + avg_r2_c2_1) / 2, label='Parent 1 Avg Child R2')

    plt.ylim(0, 1.1)

    plt.legend()
    plt.title(f'Children R2 Scores with {nbr_repetition} repetitions')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_1_children_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/bad_modular_1_children_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

    noi = str(noise).replace('.', ',')
    # modular
    df = pd.DataFrame([
                          f'mod_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                          avg_explor_mod, avg_exploit_mod, avg_r2_mod, avg_r2_c1_mod, avg_r2_c2_mod])
    df.index = ['name', 'exploration_score', 'exploitation_score', 'parent_r2', 'child1_r2', 'child2_r2']
    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/bad_modular_1_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}_noise_{noi}.csv')

    # parent 1
    df = pd.DataFrame([
                          f'parent1_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}',
                          avg_explor_1, avg_exploit_1, avg_r2_1, avg_r2_c1_1, avg_r2_c2_1])
    df.index = ['name', 'exploration_score', 'exploitation_score', 'parent_r2', 'child1_r2', 'child2_r2']

    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/parent1_kappa_{k}_gamma_{g}_nu_{n}_model_state_{nbr_query}_queries_init_{nbr_rand_init}_train_iter_{training_iter}_repetitions_{nbr_repetition}_noise_{noi}.csv')

if __name__ == '__main__':

    k_vals = [1, 3, 5, 7, 9]
    g_vals = [3]
    nu_vals = [0.5]

    for k_val in k_vals:
        with mp.Pool(processes=4) as pool:
            p1 = pool.apply_async(two_pretrained, args=([k_val], g_vals, nu_vals))
            p2 = pool.apply_async(two_pretrained_bad, args=([k_val], g_vals, nu_vals))
            p3 = pool.apply_async(one_pretrained, args=([k_val], g_vals, nu_vals))
            p4 = pool.apply_async(one_pretrained_bad, args=([k_val], g_vals, nu_vals))

            p1.get()
            p2.get()
            p3.get()
            p4.get()

    """for g_val in g_vals:
        with mp.Pool(processes=4) as pool:
            p1 = pool.apply_async(two_pretrained, args=([7.5], [g_val], nu_vals))
            p2 = pool.apply_async(two_pretrained_bad, args=([7.5], [g_val], nu_vals))
            p3 = pool.apply_async(one_pretrained, args=([7.5], [g_val], nu_vals))
            p4 = pool.apply_async(one_pretrained_bad, args=([7.5], [g_val], nu_vals))

            p1.get()
            p2.get()
            p3.get()
            p4.get()"""