from torch.xpu import device

from nn_models import *

def run_repetition(model, optimizer, data_creation_func, dimension, eps, model_name, folder_of_the_day, data_name, seed=True, noise=0.1):
    device = model.device
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    x_sub1 = x_sub1.to_device(device)
    y_sub1 = y_sub1.to_device(device)
    x_hier = x_hier.to_device(device)
    y_hier = y_hier.to_device(device)


if __name__ == "__main__":
    dimension = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = NN_baseline([dimension**2, dimension**2], device=device)

    optimizer = torch.optim.Adam(master.parameters(), lr=1e-7)