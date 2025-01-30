from efficient_general_2d import *

def exponential_experiment(nbr_repetition, nbr_rand_init, nbr_query, alpha_vals):
    dimension = 10
    training_iter = 5
    k_vals = [2]
    g_vals = [6]
    nu_vals = [0.5]
    multi = False
    h_model = hmodel.Lossless_Efficient_UCB_Hierarchical_GP

    explor = []
    exploit = []
    r2 = []
    for alpha in alpha_vals:

        temp_explor = []
        temp_exploit = []
        temp_r2 = []
        data_name, data_creation_func, eps = get_dataset_info(7, alpha=alpha)
        result = training_procedure(nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, g_vals,
                                    nu_vals, data_name, data_creation_func, eps, h_model, multi, visualize=False)
        for rep in result:
            name_1, parent_1, explor_1, exploit_1, r2_1 = rep

            temp_explor.append(explor_1)
            temp_exploit.append(exploit_1)
            temp_r2.append(r2_1)

        # Here we want to get a single value to plot as alpha increases
        explor.append(np.mean(temp_explor))
        exploit.append(np.mean(temp_exploit))
        r2.append(np.mean(temp_r2))

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
        plt.plot(alpha_vals, exploit, label='exploitation')
        plt.plot(alpha_vals, r2, label='R2')

        plt.xlabel('Alpha Value (nonlinearity)')

        plt.ylabel("Performance")
        plt.legend()
        plt.title(f'Exponential Nonlinearity Experiment')
        plt.savefig(f'{data_name}/{model_name.lower()}{folder_of_the_day}/differentiable_plots/exponential_nonlinearity_query_{nbr_query}_repetition_{nbr_repetition}_k_{k}_g_{g}_n_{n}')

if __name__ == "__main__":
    alpha_vals = np.arange(1,8)

    exponential_experiment(10, 10, 80, alpha_vals)