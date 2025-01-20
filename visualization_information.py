import os

import gpytorch.settings
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import time
from dataset_actions import NumpyArrayEncoder
import torch
from seaborn import heatmap
from scipy.stats import linregress

def compute_execution_time(executionTime_repetitions, startTime, nbr_repetition, nbr_query, folder_of_the_day,
                           workspace_folder):
    # Compute execution time
    time_per_query = executionTime_repetitions / (nbr_repetition * nbr_query)
    time_per_repetition = executionTime_repetitions / nbr_repetition
    time_per_parameter = executionTime_repetitions
    print('Time per query: ' + str(time_per_query) + ' s')  # time for one query
    print('Time per repetition: ' + str(time_per_repetition) + ' s')  # time for one repetition
    print('Time per parameter: ' + str(time_per_parameter) + ' s')  # time for one parameter evaluation

    # Compute total execution time
    executionTime = (time.time() - startTime)
    print('Query per repetition: ' + str(nbr_query))
    print('Repetition per parameter tested: ' + str(nbr_repetition))
    print('Total execution time in seconds: ' + str(executionTime) + ' s')  # total time to evaluate all parameters
    startTime = time.time()

    time_dict = {'number_of_query': nbr_query,
                 'time_per_query': time_per_query,
                 'number_of_repetition': nbr_repetition,
                 'time_per_repetition': time_per_repetition,
                 'time_per_parameter': time_per_parameter,
                 'total_time': executionTime}

    # Go to the folder of the day and save data
    os.chdir(folder_of_the_day)
    with open(str('time_of_run') + '.json', 'w') as file:
        json.dump(time_dict, file, cls=NumpyArrayEncoder)
    # Go back to workspace
    os.chdir(workspace_folder)


def heatmap_custom(map_name, q, data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    if map_name == 'prior':
        ax.set_title('Prior Map - Query ' + str(q))
    elif map_name == 'observed_pred':
        ax.set_title('Prediction Mean - Query ' + str(q))

    # Create colorbar
    cbar = ax.figure.colorbar(im, fraction=0.046, pad=0.04, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def comparison(list_prior, list_pred, q, Xmean_1D):
    """
    Compares the prior to the observed predictions of the model by plotting heatmaps
    Args:
        list_prior:
        list_pred:
        q:
        Xmean_1D:

    Returns:

    """

    font = {'weight': 'normal',
            'size': 12}
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    if (q % 10) == 0:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

        map_name = 'prior'
        im1, cbar1 = heatmap_custom(map_name, q, list_prior[-1], Xmean_1D, Xmean_1D, ax=ax1,
                                    cmap="Greens", cbarlabel="Normalized value")
        texts = annotate_heatmap(im1, data=list_prior[-1], valfmt="{x:.2f}")

        map_name = 'observed_pred'
        im2, cbar2 = heatmap_custom(map_name, q, list_pred[-1], Xmean_1D, Xmean_1D, ax=ax2,
                                    cmap="Greens", cbarlabel="Normalized value")
        texts = annotate_heatmap(im2, data=list_pred[-1], valfmt="{x:.2f}")

        fig.tight_layout()
        plt.close()
    else:
        pass


def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())
            im.set_clim(changed_image.get_clim())


def prior_map_visualization(data_dict, nbr_query, Xmean_1D, Nrow, Ncol, nbr_repetition):
    font = {'weight': 'normal',
            'size': 12}
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)
    list_of_priors_map = data_dict.get('prior_map')

    fig, axs = plt.subplots(Nrow, Ncol, figsize=(16, 9))
    # fig.tight_layout(rect=[0.1, 0.1, 0.1, 0.1])
    fig.suptitle('Prior Map at several queries')

    data = []
    for q in range(0, len(list_of_priors_map), nbr_repetition):
        data.append(list_of_priors_map[q])

    q_graph = 0
    map_indice = 0
    images = []
    for i in range(Nrow):
        for j in range(Ncol):
            images.append(axs[i, j].imshow(data[map_indice]))
            axs[i, j].label_outer()
            axs[i, j].set_title('Query ' + str(q_graph))
            axs[i, j].xaxis.set_ticks_position('top')
            axs[i, j].set_yticks(np.arange(len(Xmean_1D)), labels=Xmean_1D)
            axs[i, j].set_xticks(np.arange(len(Xmean_1D)), labels=Xmean_1D)
            plt.setp(axs[i, j].get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
            map_indice = map_indice + 1
            q_graph += nbr_repetition
    map_indice = 0
    q_graph = 0

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.05, label='Normalized, 1 = strong prior')

    for im in images:
        im.callbacks.connect('changed', update)

    fig.savefig('HGP-BO_2D-prior_maps.pdf')
    plt.close()


def heatmap_querried(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, fraction=0.046, pad=0.04, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def ch_querried_visualization(list_prior, Xmean_1D):
    font = {'weight': 'normal',
            'size': 12}
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))

    im1, cbar1 = heatmap_querried(list_prior[-1], Xmean_1D, Xmean_1D, ax=ax1,
                                  cmap="Greens", cbarlabel="Normalized value")
    texts = annotate_heatmap(im1, data=list_prior[-1], valfmt="{x:.2f}")

    fig.tight_layout()
    plt.show()
    plt.close()


def visualization_function(files, data_dict):
    font = {'weight': 'noral',
            'size': 12}
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    nbr_pts_ini = data_dict.get('pts_of_ini')
    nbr_query = data_dict.get('nbr_of_query_per_rep')
    x_axis = np.arange(nbr_query)
    x_axis = x_axis + nbr_pts_ini

    # Get a fake point to link the plot to y = 0
    first_data_point = np.array(data_dict.get('exploitation_score_mean_repetition'))[0]
    fake_data_explt = np.zeros(2)
    fake_data_explt[1] = first_data_point
    fake_abs_explt = np.zeros(2)
    fake_abs_explt[0] = fake_abs_explt[1] = nbr_pts_ini

    first_data_point = np.array(data_dict.get('exploration_score_mean_repetition'))[0]
    fake_data_explr = np.zeros(2)
    fake_data_explr[1] = first_data_point
    fake_abs_explr = np.zeros(2)
    fake_abs_explr[0] = fake_abs_explr[1] = nbr_pts_ini

    # Exploration and exploitation score/query
    with torch.no_grad():
        fig, ax = plt.subplots()
        plt.figure(figsize=(8, 5))

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # ax.scatter(for_plotting_query_number, data_dict.get('exploitation_score_mean_repetition'), c='red', marker ='x')
        ax.plot(fake_abs_explt, fake_data_explt, "r-", alpha=0.25)
        ax.plot(x_axis, data_dict.get('exploitation_score_mean_repetition'), "r-", label='Exploitation score')
        # ax.fill_between(x_axis, data_dict.get('confidenceLow_exploitation'), data_dict.get('confidenceUp_exploitation'), alpha=0.15, color='orange')

        # ax.scatter(for_plotting_query_number, data_dict.get('exploration_score_mean_repetition'), c='blue', marker ='x')
        ax.plot(fake_abs_explr, fake_data_explr, "b-", alpha=0.25)
        ax.plot(x_axis, data_dict.get('exploration_score_mean_repetition'), "b-", label='Exploration score')
        # ax.fill_between(x_axis, data_dict.get('confidenceLow_exploration'), data_dict.get('confidenceUp_exploration'), alpha=0.15, color='blue')

        ax.set_ylim([0, 1.2])
        ax.set_xlim([-5, nbr_query + nbr_pts_ini])

        ax.set_ylabel('\\textbf{Score}', fontsize=12, fontweight='bold')
        ax.set_xlabel('\\textbf{Number of query}', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle="--", color='grey')
        ax.legend(loc="lower right", fontsize='large')

        ax.set_title('HGP-BO_2D ' + str(data_dict.get('nbr_of_repetition')) + ' repetitions of ' + str(
            data_dict.get('nbr_of_query_per_rep')) + ' query ' + 'kappa ' + str(data_dict.get('kappa_value')))
        fig.savefig('HGP-BO_2D ' + str(data_dict.get('nbr_of_repetition')) + ' repetitions of ' + str(
            data_dict.get('nbr_of_query_per_rep')) + ' query ' + 'kappa ' + str(data_dict.get('kappa_value')) + '.pdf')

        if 'HGP-BO' in files:
            ax.set_title('HGP-BO_2D ' + str(data_dict.get('nbr_of_repetition')) + ' repetitions of ' + str(
                data_dict.get('nbr_of_query_per_rep')) + ' query ' + 'kappa ' + str(data_dict.get('kappa_value')))
            fig.savefig('HGP-BO_2D ' + str(data_dict.get('nbr_of_repetition')) + ' repetitions of ' + str(
                data_dict.get('nbr_of_query_per_rep')) + ' query ' + 'kappa ' + str(
                data_dict.get('kappa_value')) + '.pdf')
        elif 'GPBO_1' in files:
            ax.set_title('GPBO_1 ' + str(data_dict.get('nbr_of_repetition')) + ' repetitions of ' + str(
                data_dict.get('nbr_of_query_per_rep')) + ' query ' + 'kappa ' + str(data_dict.get('kappa_value')))
            fig.savefig('GPBO_1 ' + str(data_dict.get('nbr_of_repetition')) + ' repetitions of ' + str(
                data_dict.get('nbr_of_query_per_rep')) + ' query ' + 'kappa ' + str(
                data_dict.get('kappa_value')) + '.pdf')
        elif 'GPBO_2' in files:
            ax.set_title('GPBO_2 ' + str(data_dict.get('nbr_of_repetition')) + ' repetitions of ' + str(
                data_dict.get('nbr_of_query_per_rep')) + ' query ' + 'kappa ' + str(data_dict.get('kappa_value')))
            fig.savefig('GPBO_2 ' + str(data_dict.get('nbr_of_repetition')) + ' repetitions of ' + str(
                data_dict.get('nbr_of_query_per_rep')) + ' query ' + 'kappa ' + str(
                data_dict.get('kappa_value')) + '.pdf')


def exploration_visualization_function(folder_data_path, nbr_query, nbr_pts_ini, nbr_repetition):
    fig, ax = plt.subplots()
    plt.figure(figsize=(8, 5))

    for files in os.listdir(folder_data_path):
        if files != ("time_of_run.json"):
            if files.endswith(".json"):
                with open(files, 'r') as dict_json_file:
                    data_dict = json.load(dict_json_file)

                # Plot all exploration score for corresponding repetitions
                # Get a fake point to link the plot to y = 0
                print(files)
                nbr_pts_ini = data_dict.get('pts_of_ini')
                nbr_query = data_dict.get('nbr_of_query_per_rep')
                x_axis = np.arange(nbr_query)
                x_axis = x_axis + nbr_pts_ini

                first_data_point = np.array(data_dict.get('exploration_score_mean_repetition'))[0]
                fake_data_explr = np.zeros(2)
                fake_data_explr[1] = first_data_point
                fake_abs_explr = np.zeros(2)
                fake_abs_explr[0] = fake_abs_explr[1] = nbr_pts_ini

                # ax.scatter(for_plotting_query_number, data_dict.get('exploration_score_mean_repetition'), marker ='x')
                ax.plot(fake_abs_explr, fake_data_explr, "b-", alpha=0.25)
                ax.plot(x_axis, data_dict.get('exploration_score_mean_repetition'),
                        label='k = ' + str(data_dict.get('kappa_value')))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_ylabel('\\textbf{Exploration Score}', fontsize=12, fontweight='bold')
    ax.set_xlabel('\\textbf{Number of query}', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle="--", color='grey')
    ax.legend(loc="lower right", fontsize='large')
    ax.set_title(
        'GP-BO_2D ' + str(nbr_repetition) + ' repetitions of ' + str(nbr_query) + ' query ' + 'kappa variation')

    ax.set_ylim([0, 1.2])
    ax.set_xlim([-5, nbr_query + nbr_pts_ini])

    font = {'weight': 'normal',
            'size': 12}
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    fig.savefig(
        'GP-BO_2D' + str(data_dict.get('nbr_of_query_per_rep')) + '_query ' + '_kappa_variation_ExplorationScore.pdf')
    plt.close()


def exploitation_visualization_function(folder_data_path, nbr_repetition):
    fig, ax = plt.subplots()
    plt.figure(figsize=(8, 5))

    for files in os.listdir(folder_data_path):
        if files != ("time_of_run.json"):
            if files.endswith(".json"):
                with open(files, 'r') as dict_json_file:
                    data_dict = json.load(dict_json_file)

                print(files)
                # Plot all exploitation score for corresponding repetitions
                # Get a fake point to link the plot to y = 0
                nbr_pts_ini = data_dict.get('pts_of_ini')
                nbr_query = data_dict.get('nbr_of_query_per_rep')
                x_axis = np.arange(nbr_query)
                x_axis = x_axis + nbr_pts_ini
                first_data_point = np.array(data_dict.get('exploitation_score_mean_repetition'))[0]
                fake_data_explt = np.zeros(2)
                fake_data_explt[1] = first_data_point
                fake_abs_explt = np.zeros(2)
                fake_abs_explt[0] = fake_abs_explt[1] = nbr_pts_ini
                # ax.scatter(for_plotting_query_number, data_dict.get('exploitation_score_mean_repetition'), marker ='x')
                ax.plot(fake_abs_explt, fake_data_explt, "r-", alpha=0.25)
                ax.plot(x_axis, data_dict.get('exploitation_score_mean_repetition'),
                        label='k = ' + str(data_dict.get('kappa_value')))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_ylabel('\\textbf{Exploitation Score}', fontsize=12, fontweight='bold')
    ax.set_xlabel('\\textbf{Number of query}', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle="--", color='grey')
    ax.legend(loc="lower right", fontsize='large')
    ax.set_title('GP-BO ' + str(nbr_repetition) + ' repetitions of ' + str(nbr_query) + ' query ' + 'kappa variation')

    ax.set_ylim([0, 1.2])
    ax.set_xlim([-5, nbr_query + nbr_pts_ini])

    font = {'weight': 'normal',
            'size': 12}
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('text', usetex=True)

    fig.savefig('GP-BO_' + str(nbr_query) + '_query ' + '_kappa_variation_ExploitationScore.pdf')
    plt.close()


def model_heatmap(data, input, z, file_name, model_type, folder_of_the_day, data_name, neural=False):
    data = np.mean(data, axis=0)
    if neural:
        re_output = np.reshape(data, (10, 10))
        input = np.reshape(input, (10, 10))
    else:
        re_output = np.reshape(data, z.shape)

    fig, axs = plt.subplots(1, 2)
    ax = heatmap(re_output, cmap="viridis", xticklabels=np.round(input[0, :, 1].numpy(), 3),
                 yticklabels=np.round(input[0, :, 1].numpy(), 3), ax=axs[0])
    axs[0].set_title(f'Model Prediction of State Space')
    axs[0].set_yticklabels(np.round(input[0, :, 1].numpy(), 3), rotation=45)

    ax = heatmap(z, cmap="viridis", xticklabels=False, yticklabels=False, ax=axs[1])
    axs[1].set_title(f'True State Space')
    fig.suptitle(f"{model_type} Model Heatmap vs True State Space Heatmap")

    if neural:
        plt.savefig(f'{model_type}/{folder_of_the_day}/contour/{file_name}')
    else:
        plt.savefig(f'{data_name}/{model_type}/{folder_of_the_day}/differentiable_plots/{file_name}')
    plt.close()

def heatmap_r_score(data, z):
    data = np.mean(data, axis=0)

    r_scores = []

    z = z.reshape(data[0].shape)

    for q in range(len(data)):
        r_scores.append(linregress(z, data[q]).rvalue)

    return r_scores

def contour_plot_1D(sub_models, test_x, true_y, file_name, model_type, folder_of_the_day, data_name, neural=False, save=True):
    for i, model in enumerate(sub_models):
        model.eval()

        likelihood = model.likelihood
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
        
        with torch.no_grad():
            f, ax = plt.subplots(1, 1)

            mean = observed_pred.mean.numpy()
            mean = mean/np.max(mean)
            std = observed_pred.stddev.numpy() / np.sqrt(model.query_counter.numpy())
            train_x = model.train_inputs[0]
            train_y = model.train_targets

            div1 = torch.clone(train_y)
            div1[model.env_ind] = div1[model.env_ind]/model.env_max_seen
            div1[model.bif_ind] = div1[model.bif_ind]/model.bif_max_seen

            if neural:
                temp_x = list(range(len(test_x)))
                new_train = map_neural_to_list(train_x.numpy())
                #ax.plot(new_train, train_y.numpy(), 'k*')
                ax.plot(temp_x, mean, 'b')
                #ax.fill_between(temp_x, mean-std, mean+std, alpha=0.5)

                ax.plot(temp_x, true_y[i].numpy(), 'r')
            else:
                #ax.plot(train_x.numpy(), div1.numpy(), 'k*')
                ax.plot(test_x.numpy(), mean, 'b')
                ax.fill_between(test_x.numpy(), mean-std, mean+std, alpha=0.5)
                ax.plot(test_x.numpy(), true_y[i].numpy(), 'r')

            ax.legend(['Observed Data', 'Mean', 'Confidence', 'True'])
        plt.title(f"{model_type} SubModel {i} Contour Map for Respective Data")
        plt.tight_layout()
        if save:
            if neural:
                plt.savefig(f'{model_type}/{folder_of_the_day}/{file_name}_submodel_{i}')

            else:
                plt.savefig(f'{data_name}/{model_type}/{folder_of_the_day}/{file_name}_submodel_{i}')
        else:
            plt.show()
        plt.close()

def map_neural_to_list(train):
    new_train = []

    for x in train:
        a = abs(x[0] - 1)
        b = x[1]

        new_train.append(int(a * 5 + b))

    return new_train

def heatmap_3d(y, z, file_name, model_type, folder_of_the_day, data_name):
    y = np.mean(y, axis=0)

    re_y = np.reshape(y, z.shape)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(re_y, cmap="viridis")
    ax.set_title(f'Model Prediction of State Space')
    #axs[0].set_yticklabels(np.round(input[0, :, 1].numpy(), 3), rotation=45)

    ax2 = fig.add_subplot(111, projection='3d')
    img = ax2.scatter(z[0], z[1], z[2], cmap="viridis", xticklabels=False, yticklabels=False)
    ax2.set_title(f'True State Space')
    fig.suptitle(f"{model_type} Model Heatmap vs True State Space Heatmap")



    # creating the heatmap
    img = ax.scatter(y1, y2, y3, marker='s',
                     s=200, color='green')
    plt.colorbar()

    # adding title and labels
    ax.set_title("3D Heatmap")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.savefig(f'{data_name}/{model_type}/{folder_of_the_day}/{file_name}')