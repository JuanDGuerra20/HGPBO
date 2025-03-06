
from dataset_actions import *


def nn_pretraining(train_x, train_y, test_x, test_y, model, loss_fn, optimizer):

    size = len(train_x)

    model.train()

    for batch, (X,y) in enumerate(zip(train_x, train_y)):

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def create_parent_distr_from_child(y1, y2):
    y_1 = torch.reshape(y1, (-1,1))
    y_2 = torch.reshape(y2, (1,-1))

    numerator = torch.add(y_1, y_2)

    coord1 = torch.arange(1, y_1.shape[0] + 1).reshape(-1,1)
    coord2 = torch.arange(1, y_2.shape[1] + 1).reshape(1,-1)
    denominator = torch.sub(coord1, coord2) + 0.000002
    denominator = torch.pow(denominator, 2)
    return torch.div(numerator, denominator)


if __name__ == '__main__':
    dimension = 15
    nbr_query = 80
    training_iter = 5
    nbr_repetition = 15
    k_vals = [2]
    g_vals = [6]
    nu_vals = [0.5]
    multi = True

    # must pretrain the model before running a repetition, consider saving it to huggingface
    #master = hmodel.NN_Hierarchical_Comb(input_dim=dimension*dimension, hidden_dim=dimension*dimension, output_dim=3)

    a = torch.rand((dimension))
    b = torch.rand((dimension))

    c = create_parent_distr_from_child(a, b)
    print(c.shape)
