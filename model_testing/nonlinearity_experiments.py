import matplotlib.pyplot as plt
import numpy as np

from efficient_general_2d import *

def mult_factor_experiment(nbr_repetition, nbr_rand_init, nbr_query, alpha_vals):
    dimension = 32
    training_iter = 10
    k_vals = [7.5]
    g_vals = [3]
    nu_vals = [0.5]
    multi = True
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    seed = [False] * nbr_repetition

    explor = []
    r2 = []
    child_r2 = []
    auc = []
    for i, alpha in enumerate(alpha_vals):

        data_name, data_creation_func, eps = get_dataset_info(8, alpha=alpha)
        try:
            result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                               nu_vals, data_name, data_creation_func,
                               eps, h_model, multi, seed, noise=0.1, visualize=True, disable_tqdm=False)
        except:
            try:
                result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                               nu_vals, data_name, data_creation_func,
                               eps, h_model, multi, seed, noise=0.1, visualize=True, disable_tqdm=False)
            except:
                try:
                    result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                               nu_vals, data_name, data_creation_func,
                               eps, h_model, multi, seed, noise=0.1, visualize=True, disable_tqdm=False)
                except:
                    try:
                        result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                               nu_vals, data_name, data_creation_func,
                               eps, h_model, multi, seed, noise=0.1, visualize=True, disable_tqdm=False)
                    except:
                        result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                               nu_vals, data_name, data_creation_func,
                               eps, h_model, multi, seed, noise=0.1, visualize=True, disable_tqdm=False)
        name_1, explor_1, r2_1, avg_child_1 = result[0]

        """temp_r2 = np.mean(r2_1, axis=0)


        temp_child_1_r2_1 = np.mean(child_1_r2_1 ,axis=0)
        temp_child_2_r2_1 = np.mean(child_1_r2_2, axis=0)"""

        # Here we want to get a single value to plot as alpha increases
        explor.append(explor_1[-1])
        r2.append(r2_1[-1])
        child_r2.append(avg_child_1[-1])
        auc.append(np.sum(explor_1 + r2_1 + avg_child_1))
    if h_model == hmodel.Efficient_UCB_Hierarchical_GP:
        model_name = "Efficient"

    elif h_model == hmodel.Lossless_Efficient_UCB_Hierarchical_GP:
        model_name = "Lossless_Efficient"
    data_name, data_creation_func, eps = get_dataset_info(8, alpha=alpha)

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)

    k = str(k_vals[0]).replace('.', ',')
    g = str(g_vals[0]).replace('.', ',')
    n = str(nu_vals[0]).replace('.', ',')

    plt.plot(alpha_vals, explor, label='RO')
    plt.plot(alpha_vals, r2, label='Parent R2')
    plt.plot(alpha_vals, child_r2, label='Child R2')

    plt.xlabel('Alpha Value (nonlinearity)')
    plt.ylim(-0.1, 1.1)
    plt.xscale("log", base=2)
    plt.ylabel("Performance")
    plt.legend()
    plt.title(f'Mult Factor Nonlinearity Experiment')
    plt.savefig(f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/mult_factor_nonlinearity_q_{nbr_query}_rep_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/mult_factor_nonlinearity_q_{nbr_query}_rep_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()
    df = pd.DataFrame(
        [alpha_vals, explor, r2, child_r2, auc])
    df.index = ['alpha', 'instantaneous_regret', 'parent_r2', 'avg_child_r2', 'auc']
    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/alpha_values.csv')
def exponential_experiment(nbr_repetition, nbr_rand_init, nbr_query, alpha_vals, dataset_num):
    dimension = 32
    training_iter = 10
    k_vals = [7.5]
    g_vals = [3]
    nu_vals = [0.5]
    multi = True
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    seed = [False] * nbr_repetition

    explor = []
    r2 = []
    auc = []
    child_r2 = []
    for i, alpha in enumerate(alpha_vals):

        data_name, data_creation_func, eps = get_dataset_info(dataset_num, alpha=alpha)
        try:
            result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                        g_vals,
                                        nu_vals, data_name, data_creation_func,
                                        eps, h_model, multi, seed, noise=0.1, visualize=True, disable_tqdm=False)
        except:
            try:
                result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                            g_vals,
                                            nu_vals, data_name, data_creation_func,
                                            eps, h_model, multi, seed, noise=0.1, visualize=True, disable_tqdm=False)
            except:
                try:
                    result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
                                                k_vals, g_vals,
                                                nu_vals, data_name, data_creation_func,
                                                eps, h_model, multi, seed, noise=0.1, visualize=True,
                                                disable_tqdm=False)
                except:
                    try:
                        result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
                                                    k_vals, g_vals,
                                                    nu_vals, data_name, data_creation_func,
                                                    eps, h_model, multi, seed, noise=0.1, visualize=True,
                                                    disable_tqdm=False)
                    except:
                        result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
                                                    k_vals, g_vals,
                                                    nu_vals, data_name, data_creation_func,
                                                    eps, h_model, multi, seed, noise=0.1, visualize=True,
                                                    disable_tqdm=False)
        name_1, explor_1, r2_1, avg_child_1 = result[0]

        # Here we want to get a single value to plot as alpha increases
        explor.append(explor_1[-1])
        r2.append(r2_1[-1])
        child_r2.append(avg_child_1[-1])
        auc.append(np.sum(explor_1 + r2_1 + avg_child_1))
    data_name, data_creation_func, eps = get_dataset_info(dataset_num, alpha=alpha)

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

    plt.plot(alpha_vals, explor, label='exploration')
    plt.plot(alpha_vals, r2, label='Parent R2')
    plt.plot(alpha_vals, child_r2, label='Child R2')

    plt.xlabel('Alpha Value (nonlinearity)')
    plt.ylim(-0.1, 1.1)
    plt.xscale('log', basex=2)
    plt.ylabel("Performance")
    plt.legend()
    plt.title(f'{data_name} Experiment')
    plt.savefig(f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/{data_name}_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/{data_name}_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()
    df = pd.DataFrame(
        [alpha_vals, explor,r2, child_r2, auc])
    df.index = ['alpha', 'instantaneous_regret', 'parent_r2', 'avg_child_r2', 'auc']
    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/alpha_values.csv')

def b_mult_experiment(nbr_repetition, nbr_rand_init, nbr_query, alpha_vals):
    dimension = 32
    training_iter = 10
    k_vals = [7.5]
    g_vals = [3]
    nu_vals = [0.5]
    multi = True
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP
    seed = [False] * nbr_repetition

    heat_explor = []
    heat_r2 = []
    heat_child_r2 = []
    heat_auc = []
    for i in range(len(alpha_vals)):
        explor = []
        r2 = []
        c1_r2 = []
        auc = []
        for j in range(len(alpha_vals[i])):
            betas = alpha_vals[i][j]
            data_name, data_creation_func, eps = get_dataset_info(9, alpha=betas)

            try:
                result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals,
                                            g_vals,
                                            nu_vals, data_name, data_creation_func,
                                            eps, h_model, multi, seed, noise=0.1, visualize=True, disable_tqdm=False)
            except:
                try:
                    result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
                                                k_vals,
                                                g_vals,
                                                nu_vals, data_name, data_creation_func,
                                                eps, h_model, multi, seed, noise=0.1, visualize=True,
                                                disable_tqdm=False)
                except:
                    try:
                        result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
                                                    k_vals, g_vals,
                                                    nu_vals, data_name, data_creation_func,
                                                    eps, h_model, multi, seed, noise=0.1, visualize=True,
                                                    disable_tqdm=False)
                    except:
                        try:
                            result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension,
                                                        training_iter,
                                                        k_vals, g_vals,
                                                        nu_vals, data_name, data_creation_func,
                                                        eps, h_model, multi, seed, noise=0.1, visualize=True,
                                                        disable_tqdm=False)
                        except:
                            result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension,
                                                        training_iter,
                                                        k_vals, g_vals,
                                                        nu_vals, data_name, data_creation_func,
                                                        eps, h_model, multi, seed, noise=0.1, visualize=True,
                                                        disable_tqdm=False)

            name_1, explor_1, r2_1, child_r2_1 = result[0]

            explor.append(explor_1[-1])
            r2.append(r2_1[-1])
            c1_r2.append(child_r2_1[-1])
            auc.append(np.sum(explor_1 + r2_1 + child_r2_1))
        heat_explor.append(explor)
        heat_r2.append(r2)
        heat_child_r2.append(c1_r2)
        heat_auc.append(auc)

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
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/BETA_explor_mult_factor_nonlinearity_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()

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
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/BETA_parent_r2_mult_factor_nonlinearity_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()
    plt.imshow(heat_child_r2)
    plt.xlabel('B1 Value (nonlinearity)')
    plt.ylabel("B2 Value (nonlinearity)")
    plt.xticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.yticks(range(len(alpha_vals)), alpha_vals[:, 0, 0])
    plt.colorbar()
    plt.title(f'Mult Beta Factor Nonlinearity Parent R2 Score')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/BETA_child_1_r2_mult_factor_nonlinearity_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')
    plt.savefig(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/hp_analysis/BETA_child_1_r2_mult_factor_nonlinearity_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}.svg')

    plt.close()
    df = pd.DataFrame(
        [betas, heat_explor,heat_r2, heat_child_r2, heat_auc])
    df.index = ['alpha', 'instantaneous_regret', 'parent_r2', 'avg_child_r2', 'auc']
    df.to_csv(
        f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/beta_values.csv')


if __name__ == "__main__":
    alpha_vals = [0.25, 0.5, 1, 2, 4, 8, 16, 20]
    try:
        mult_factor_experiment(10, 3, 100, alpha_vals)
    except:
        print("mult failed")
    try:
        exponential_experiment(10, 3, 100, alpha_vals, 7)
    except:
        print("exponential failed")
    try:
        exponential_experiment(10, 3, 100, alpha_vals, 7.5)
    except:
        print("exponential 2 failed")
    """
    
    #with mp.Pool(processes=3) as pool:

        p1 = pool.apply_async(exponential_experiment, (20, 10, 100, alpha_vals, 7,))
        p2 = pool.apply_async(exponential_experiment, (20, 10, 100, alpha_vals, 7.5, ))

        p3 = pool.apply_async(mult_factor_experiment, (20, 10, 100, alpha_vals, ))

        p1.get()
        p2.get()
        p3.get()"""

    """alpha_vals = [1,2]
    b_list = []
    for i in range(len(alpha_vals)):
        temp_b = []
        for j in range(len(alpha_vals)):
            temp_b.append([alpha_vals[i], alpha_vals[j]])
        b_list.append(temp_b)
    b_mult_experiment(2, 6, 10, b_list)"""