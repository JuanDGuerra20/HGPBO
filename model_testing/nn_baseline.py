import matplotlib.pyplot as plt
import torch
from nn_models import *
import warnings

def training_procedure(model, optimizer, loss_fn, X, y, training_iter):
    model.train()
    if model.device != torch.device('cpu'):
        torch.cuda.synchronize()
    X = X.to(torch.float)
    y = y.to(torch.float)
    for i in range(training_iter):
        pred = model(X)
        train_loss = loss_fn(y/torch.max(y), pred/torch.max(pred))
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def run_repetition(model, optimizer, loss_fn, nbr_query, training_iter, rand_init, data_creation_func, dimension, eps, model_name, folder_of_the_day, data_name, seed=True, noise=0.1):
    device = model.device
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)
    heatmap_rep = []
    better_exploration_score = []
    test_x_hier = test_x_hier.to(torch.float)
    x_hier = x_hier.to(torch.float)
    test_x_hier = test_x_hier.to(device)
    x_hier = x_hier.to(device)
    y_hier = y_hier.to(device)

    list_acquisitions = []

    for q in tqdm(range(nbr_query)):

        if q == 0:
            train_x_hier, train_y_hier = hierarchical_select_random_queries(rand_init, x_hier, y_hier, seed=seed, noise=noise)

        acquisition_map = model.get_acquisition_map(test_x_hier)
        list_acquisitions.append(acquisition_map)

        next_query_pins = torch.tensor(get_next_query_pins(acquisition_map, test_x_hier))


        next_query_value_random, next_query_value_mean = get_next_query_value(next_query_pins,
                                                                                     test_x_hier,
                                                                                     y_hier, noise=noise)
        response = torch.tensor(next_query_value_random, device=device)


        train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins,
                                                          response)
        training_procedure(model, optimizer, loss_fn, train_x_hier, train_y_hier, training_iter)

        hierar_y_mu = model(test_x_hier)
        ground_truth_max_hier = torch.argmax(hierar_y_mu)

        exploration_score_2D, next_query_pins_exploration_2D = get_exploration_score(hierar_y_mu,
                                                                                    ground_truth_max_hier,
                                                                                    test_x_hier,
                                                                                    x_hier, y_hier)
        better_exploration_score.append(exploration_score_2D)

        #plotting the heatmaps
        """heatmap = hierar_y_mu.reshape(10,10)
        plt.imshow(heatmap.detach().numpy())
        plt.title(f"NN Heatmap for query {q} of {nbr_query}")
        plt.colorbar()
        plt.savefig(f"{data_name}/{model_name}/{folder_of_the_day}/png/heatmap_q_{q}_{nbr_query}.png")
        plt.close()"""
        heatmap_rep.append(hierar_y_mu)

    return master, better_exploration_score, heatmap_rep


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    dimension = 10
    training_iter = 10
    nbr_query = 1000
    rand_init = 10
    lr = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    over_explor = []
    over_r2 = []

    data_name, data_creation_func, eps = get_dataset_info(3)
    model_name = "nn_baseline"

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)

    if os.path.exists(workspace + folder_of_the_day):
        print('Data folder is ready')
    else:
        os.mkdir(workspace + folder_of_the_day)
        print('Data folder created')
        os.mkdir(workspace + folder_of_the_day + '/contour')
        print("Contour folder created")
        os.mkdir(workspace + folder_of_the_day + '/differentiable_plots')
        print("CSV folder created")
        os.mkdir(workspace + folder_of_the_day + '/csv')
        print("HP folder created")
        os.mkdir(workspace + folder_of_the_day + '/hp_analysis')
        print("Model folder created")
        os.mkdir(workspace + folder_of_the_day + '/models')
        print("PNG folder created")
        os.mkdir(workspace + folder_of_the_day + '/png')
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    x_hier = x_hier.to(torch.float)
    test_x_hier = test_x_hier.to(torch.float)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    test_x_hier = test_x_hier.to(device)

    for alpha in lr:
        master = NN_baseline([dimension**2, dimension**2], test_x_hier, device=device).to(device)

        optimizer = torch.optim.Adam(master.parameters(), lr=alpha)

        loss_fn = nn.MSELoss()


        master, exploration_score, heatmap = run_repetition(master, optimizer, loss_fn, nbr_query, training_iter, rand_init, data_creation_func, dimension, eps, model_name, folder_of_the_day, data_name, seed=False, noise=0.1)

        print(f"exploration: {exploration_score}")

        plt.plot(list(range(len(exploration_score))), exploration_score)
        plt.ylim(-0.1, 1.1)
        plt.xlabel(f"Query Number")
        plt.ylabel('Exploration Score')
        plt.savefig(f"{data_name}/{model_name}/{folder_of_the_day}/differentiable_plots/exploration_score_over_training_lr_{alpha}.png")
        plt.close()
        over_explor.append(exploration_score)
        over_r2.append(heatmap)