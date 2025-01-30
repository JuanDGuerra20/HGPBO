from efficient_general_2d import *


if __name__ == '__main__':
    dimension = 10
    nbr_query = 80
    training_iter = 5
    nbr_repetition = 5
    nbr_rand_init = 5
    k_vals = [2]
    g_vals = [6]
    nu_vals = [0.5]
    multi = False
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP

    avg_explor_1 = []
    avg_exploit_1 = []
    avg_r2_1 = []
    
    avg_explor_2 = []
    avg_exploit_2 = []
    avg_r2_2 = []

    avg_explor_mod = []
    avg_exploit_mod = []
    avg_r2_mod = []
    
    for i in tqdm(range(nbr_repetition)):
        # Setting up for dataset number 1
        data_name, data_creation_func, eps = get_dataset_info(2)

        func_1 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                                    nu_vals, data_name, data_creation_func, eps, h_model, multi, visualize=True)
        
        name_1, parent_1, explor_1, exploit_1, r2_1 = func_1[0]

        avg_explor_1.append(np.mean(explor_1, axis=0))
        avg_exploit_1.append(np.mean(exploit_1, axis=0))
        avg_r2_1.append(r2_1)

        # Setting up for dataset number 2
        data_name, data_creation_func, eps = get_dataset_info(3)

        func_2 = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                                    nu_vals, data_name, data_creation_func, eps, h_model, multi, visualize=True)

        name_2, parent_2, explor_2, exploit_2, r2_2 = func_2[0]

        avg_explor_2.append(np.mean(explor_2, axis=0))
        avg_exploit_2.append(np.mean(exploit_2, axis=0))
        avg_r2_2.append(r2_2)

        child_11, child_12 = parent_1.sub_models

        child_21, child_22 = parent_2.sub_models


        # =======================================================
        # Modular Section

        modular_children = [child_11, child_22]

        data_name, data_creation_func, eps = get_dataset_info(6)
        func_mod = training_procedure(nbr_query, 1, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                                        nu_vals, data_name, data_creation_func, eps, h_model, multi, children=modular_children, visualize=True)
        
        name_mod, parent_mod, explor_mod, exploit_mod, r2_mod = func_mod[0]

        avg_explor_mod.append(np.mean(explor_mod, axis=0))
        avg_exploit_mod.append(np.mean(exploit_mod, axis=0))
        avg_r2_mod.append(r2_mod)
    
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

    avg_explor_1 = np.mean(avg_explor_1, axis=0)
    avg_exploit_1 = np.mean(avg_exploit_1, axis=0)
    avg_r2_1 = np.mean(avg_r2_1, axis=0)

    avg_explor_2 = np.mean(avg_explor_2, axis=0)
    avg_exploit_2 = np.mean(avg_exploit_2, axis=0)
    avg_r2_2 = np.mean(avg_r2_2, axis=0)

    avg_explor_mod = np.mean(avg_explor_mod, axis=0)
    avg_exploit_mod = np.mean(avg_exploit_mod, axis=0)
    avg_r2_mod = np.mean(avg_r2_mod, axis=0)

    plt.plot(avg_explor_1, label='Exploration 1')
    plt.plot(avg_explor_2, label='Exploration 2')
    plt.plot(avg_explor_mod, label='Exploration Modular')

    plt.legend()
    plt.title(f'2D Modularity Experiment Exploration with {nbr_repetition} repetitions')
    plt.savefig(f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_exploration_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.close()

    plt.plot(avg_exploit_1, label='Exploitation 1')
    plt.plot(avg_exploit_2, label='Exploitation 2')
    plt.plot(avg_exploit_mod, label='Exploitation Modular')

    plt.legend()
    plt.title(f'2D Modularity Experiment Exploitation with {nbr_repetition} repetitions')
    plt.savefig(f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_exploitation_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.close()

    plt.plot(avg_r2_1, label='R2 1')
    plt.plot(avg_r2_2, label='R2 2')
    plt.plot(avg_r2_mod, label='R2 Modular')

    plt.legend()
    plt.title(f'2D Modularity Experiment R2 with {nbr_repetition} repetitions')
    plt.savefig(f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/modular_r2_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.close()