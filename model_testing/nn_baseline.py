import torch
from nn_models import *

def training_procedure(model, optimizer, loss_fn, X, y, training_iter):
    model.train()
    torch.cuda.synchronize()

    for i in training_iter:
        pred = model(X)
        train_loss = loss_fn(y, pred)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def run_repetition(model, optimizer, loss_fn, nbr_query, training_iter, data_creation_func, dimension, eps, model_name, folder_of_the_day, data_name, seed=True, noise=0.1):
    device = model.device
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    test_x_hier = test_x_hier.to(device)
    x_hier = x_hier.to(device)
    y_hier = y_hier.to(device)

    for q in tqdm(range(nbr_query)):

        if q == 0:
            train_x_hier, train_y_hier = hierarchical_select_random_queries(1, x_hier, y_hier, seed=seed, noise=noise)

        acquisition_map = model.get_acquisition_map(test_x_hier)

        next_query_pins = torch.tensor(get_next_query_pins(acquisition_map, test_x_hier))


        next_query_value_random, next_query_value_mean = get_next_query_value(next_query_pins,
                                                                                     test_x_hier,
                                                                                     y_hier, noise=noise)
        response = torch.tensor(next_query_value_random)


        train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins,
                                                          response)

        training_procedure(model, optimizer, loss_fn, train_x_hier, train_y_hier, training_iter)


if __name__ == "__main__":
    dimension = 10
    training_iter = 10
    nbr_query = 100

    data_name, data_creation_func, eps = get_dataset_info(3)
    model_name = "nn_baseline"

    current_datetime = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_x_hier = test_x_hier.to(device)
    master = NN_baseline([dimension**2, dimension**2], test_x_hier, device=device)

    optimizer = torch.optim.Adam(master.parameters(), lr=1e-7)

    loss_fn = nn.MSELoss()


    run_repetition(master, optimizer, loss_fn, nbr_query, training_iter, data_creation_func, dimension, eps, model_name, folder_of_the_day, data_name, seed=False, noise=0.1)