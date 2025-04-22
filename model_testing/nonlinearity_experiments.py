import matplotlib.pyplot as plt
import numpy as np

from efficient_general_2d import *

def mult_factor_experiment(nbr_repetition, nbr_rand_init, nbr_query, alpha_vals):
    dimension = 10
    training_iter = 5
    k_vals = [2]
    g_vals = [6]
    nu_vals = [0.5]
    multi = False
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    seed = [False] * nbr_repetition

    explor = []
    exploit = []
    r2 = []
    child_1_r2 = []
    child_2_r2 = []
    for i, alpha in enumerate(alpha_vals):

        temp_explor = []
        temp_exploit = []
        temp_r2 = []
        temp_child_1_r2_1 = []
        temp_child_2_r2_1 = []
        data_name, data_creation_func, eps = get_dataset_info(8, alpha=alpha)
        try:
            result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                        g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed, noise=0.1)
        except:
            try:
                result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                            g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                            noise=0.1)
            except:
                result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                            g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                            noise=0.1)

        name_1, parent_1, explor_1, exploit_1, r2_1, child_1_r2_1, child_1_r2_2 = result[0]

        temp_explor = np.mean(explor_1, axis=0)
        temp_exploit = np.mean(exploit_1, axis=0)
        """temp_r2 = np.mean(r2_1, axis=0)


        temp_child_1_r2_1 = np.mean(child_1_r2_1 ,axis=0)
        temp_child_2_r2_1 = np.mean(child_1_r2_2, axis=0)"""

        # Here we want to get a single value to plot as alpha increases
        explor.append(temp_explor[-1])
        exploit.append(temp_exploit[-1])
        r2.append(r2_1[-1])
        child_1_r2.append(child_1_r2_1[-1])
        child_2_r2.append(child_1_r2_2[-1])

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

    plt.plot(alpha_vals[:i + 1], explor, label='exploration')
    plt.plot(alpha_vals[:i + 1], exploit, label='exploitation')
    plt.plot(alpha_vals[:i + 1], r2, label='Parent R2')
    plt.plot(alpha_vals[:i + 1], child_1_r2, label='Child 1 R2')
    plt.plot(alpha_vals[:i + 1], child_2_r2, label='Child 2 R2')

    plt.xlabel('Alpha Value (nonlinearity)')
    plt.ylim(0, 1.1)

    plt.ylabel("Performance")
    plt.legend()
    plt.title(f'Mult Factor Nonlinearity Experiment')
    plt.savefig(f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/mult_factor_nonlinearity_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')

    plt.close()

def exponential_experiment(nbr_repetition, nbr_rand_init, nbr_query, alpha_vals, dataset_num):
    dimension = 10
    training_iter = 5
    k_vals = [2]
    g_vals = [6]
    nu_vals = [0.5]
    multi = False
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    seed = [False] * nbr_repetition

    explor = []
    exploit = []
    r2 = []
    child_1_r2 = []
    child_2_r2 = []
    for i, alpha in enumerate(alpha_vals):

        data_name, data_creation_func, eps = get_dataset_info(dataset_num, alpha=alpha)
        try:
            result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                   g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed, noise=0.1)
        except:
            try:
                result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed, noise=0.1)
            except:
                result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                       g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed, noise=0.1)

        name_1, parent_1, explor_1, exploit_1, r2_1, child_1_r2_1, child_1_r2_2 = result[0]

        temp_explor = np.mean(explor_1, axis=0)
        temp_exploit = np.mean(exploit_1, axis=0)
        """temp_r2 = np.mean(r2_1, axis=0)


        temp_child_1_r2_1 = np.mean(child_1_r2_1 ,axis=0)
        temp_child_2_r2_1 = np.mean(child_1_r2_2, axis=0)"""

        # Here we want to get a single value to plot as alpha increases
        explor.append(temp_explor[-1])
        exploit.append(temp_exploit[-1])
        r2.append(r2_1[-1])
        child_1_r2.append(child_1_r2_1[-1])
        child_2_r2.append(child_1_r2_2[-1])

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

    plt.plot(alpha_vals[:i+1], explor, label='exploration')
    plt.plot(alpha_vals[:i+1], exploit, label='exploitation')
    plt.plot(alpha_vals[:i+1], r2, label='Parent R2')
    plt.plot(alpha_vals[:i+1], child_1_r2, label='Child 1 R2')
    plt.plot(alpha_vals[:i+1], child_2_r2, label='Child 2 R2')

    plt.xlabel('Alpha Value (nonlinearity)')
    plt.ylim(0, 1.1)

    plt.ylabel("Performance")
    plt.legend()
    plt.title(f'{data_name} Experiment')
    plt.savefig(f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/{data_name}_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')

    plt.close()

def b_mult_experiment(nbr_repetition, nbr_rand_init, nbr_query, alpha_vals):
    dimension = 10
    training_iter = 5
    k_vals = [2]
    g_vals = [6]
    nu_vals = [0.5]
    multi = False
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP

    heat_explor = []
    heat_exploit = []
    heat_r2 = []
    heat_child_1_r2 = []
    heat_child_2_r2 = []
    seed = [False] * nbr_repetition

    for i in range(len(alpha_vals)):
        explor = []
        exploit = []
        r2 = []
        c1_r2 = []
        c2_r2 = []
        for j in range(len(alpha_vals[i])):
            betas = alpha_vals[i][j]
            data_name, data_creation_func, eps = get_dataset_info(9, alpha=betas)

            temp_explor = []
            temp_exploit = []
            temp_r2 = []
            temp_child_1_r2 = []
            temp_child_2_r2 = []
            try:
                result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                            g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                            noise=0.1)
            except:
                try:
                    result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
                                                k_vals,
                                                g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                                noise=0.1)
                except:
                    result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
                                                k_vals,
                                                g_vals, nu_vals, data_name, data_creation_func, eps, h_model, multi, seed,
                                                noise=0.1)

            name_1, parent_1, explor_1, exploit_1, r2_1, child_1_r2, child_2_r2 = result[0]

            explor.append(np.mean(explor_1, axis=0)[-1])
            exploit.append(np.mean(exploit_1, axis=0)[-1])
            r2.append(r2_1[-1])
            c1_r2.append(child_1_r2[-1])
            c2_r2.append(child_2_r2[-1])

        heat_explor.append(explor)
        heat_exploit.append(exploit)
        heat_r2.append(r2)
        heat_child_1_r2.append(c1_r2)
        heat_child_2_r2.append(c2_r2)


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

    alpha_vals = np.array(alpha_vals)

    plt.imshow(heat_explor)
    plt.xlabel('B1 Value (nonlinearity)')
    plt.ylabel("B2 Value (nonlinearity)")
    plt.xticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.yticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.colorbar()
    plt.title(f'Mult Beta Factor Nonlinearity Exploration Score')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/BETA_explor_mult_factor_nonlinearity_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')

    plt.close()
    plt.imshow(heat_exploit)
    plt.xlabel('B1 Value (nonlinearity)')
    plt.ylabel("B2 Value (nonlinearity)")
    plt.xticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.yticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.colorbar()
    plt.title(f'Mult Beta Factor Nonlinearity Exploitation Score')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/BETA_exploit_mult_factor_nonlinearity_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')

    plt.close()
    plt.imshow(heat_r2)
    plt.xlabel('B1 Value (nonlinearity)')
    plt.ylabel("B2 Value (nonlinearity)")
    plt.xticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.yticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.colorbar()
    plt.title(f'Mult Beta Factor Nonlinearity Parent R2 Score')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/BETA_parent_r2_mult_factor_nonlinearity_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')

    plt.close()
    plt.imshow(heat_child_1_r2)
    plt.xlabel('B1 Value (nonlinearity)')
    plt.ylabel("B2 Value (nonlinearity)")
    plt.xticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.yticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.colorbar()
    plt.title(f'Mult Beta Factor Nonlinearity Parent R2 Score')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/BETA_child_1_r2_mult_factor_nonlinearity_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')

    plt.close()
    plt.imshow(heat_child_2_r2)
    plt.xlabel('B1 Value (nonlinearity)')
    plt.ylabel("B2 Value (nonlinearity)")
    plt.xticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.yticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.colorbar()
    plt.title(f'Mult Beta Factor Nonlinearity Parent R2 Score')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/BETA_child_2_r2_mult_factor_nonlinearity_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')

    plt.close()


if __name__ == "__main__":
    alpha_vals = [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    mult_factor_experiment(20, 10 ,100, alpha_vals)
    exponential_experiment(20, 10 ,100, alpha_vals, 7)

    """ b_list = []
    for i in range(len(alpha_vals)):
        temp_b = []
        for j in range(len(alpha_vals)):
            temp_b.append([alpha_vals[i], alpha_vals[j]])
        b_list.append(temp_b)

    with mp.Pool(processes=3) as pool:

        p1 = pool.apply_async(exponential_experiment, (20, 10, 100, alpha_vals, ))


        p3 = pool.apply_async(generate_mult_factor_nonlinearity_dataset, (20, 10, 100, alpha_vals,))

        p1.get()
        p3.get()"""