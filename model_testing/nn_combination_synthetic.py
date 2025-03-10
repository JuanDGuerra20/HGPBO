import matplotlib.pyplot as plt
import numpy as np

import hmodel_synthetic as hmodel
from dataset_actions import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def nn_pretraining(train_x, train_y, val_x, val_y, model, loss_fn, optimizer, num_epochs, min_delta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    queue = []

    torch.cuda.synchronize()

    model.train()
    val_losses = []
    train_losses = []
    x_vals = []

    epoch_val = []
    t = tqdm(range(num_epochs))
    tolerance = 0
    for epoch in t:
        batch_val = []
        for batch, (X,y) in enumerate(zip(train_x, train_y)):
            pred = model(X)
            train_loss = loss_fn(y, pred)
            if batch % 100 == 0:
                train_losses.append(train_loss.item())
                val_loss = loss_fn(val_y, model(val_x))
                val_losses.append(val_loss.item())

                batch_val.append(val_loss.item())

                x_vals.append(epoch*len(X) + batch)
                plt.plot(train_losses, label='train')
                plt.plot(val_losses, label='val')
                plt.legend()
                plt.ylabel("Loss")
                plt.xlabel("Batch")
                plt.savefig(f"nn_combination/Loss_Graph")
                plt.close()

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        batch_val = np.mean(batch_val)
        epoch_val.append(batch_val)
        t.postfix = f"Train Loss: {train_loss.item():.2f}, Val Loss: {val_loss.item():.2f}"
        queue.append(model.state_dict())
        if len(queue) > 5:
            queue.pop()

        if batch_val - np.min(epoch_val) > min_delta:
            tolerance += 1
        elif tolerance < 0 and batch_val - np.min(epoch_val) < min_delta:
            tolerance -= 1

        if tolerance >= 5:
            print(f"Early stopping detected, validation loss increasing over past 5 epochs")
            return queue[-1]
    return model

def create_parent_distr_from_child(y1, y2):
    y_1 = torch.reshape(y1, (-1,1))
    y_2 = torch.reshape(y2, (1,-1))

    numerator = torch.add(y_1, y_2)

    coord1 = torch.arange(1, y_1.shape[0] + 1).reshape(-1,1)
    coord2 = torch.arange(1, y_2.shape[1] + 1).reshape(1,-1)
    denominator = torch.abs(torch.sub(coord1, coord2)/torch.max(coord2))
    denominator = torch.sqrt(denominator) + 1
    xh = torch.div(denominator, numerator)
    y_mag = torch.max(xh)
    y_loc = torch.argmax(xh)
    return torch.reshape(xh, (-1,)), torch.tensor([y_mag, y_loc])

def create_data(num_data_points):
    y1_train = []
    y2_train = []
    xh_train = []
    yh_train = []
    for i in range(num_data_points):
        y1 = torch.randint(1, 100, (dimension,))
        y2 = torch.randint(1, 100, (dimension,))

        xh, yh = create_parent_distr_from_child(y1, y2)

        y1_train.append(y1)
        y2_train.append(y2)
        xh_train.append(xh)
        yh_train.append(yh)

    y1_train = torch.stack(y1_train)
    y2_train = torch.stack(y2_train)
    xh_train = torch.stack(xh_train)
    yh_train = torch.stack(yh_train)

    return y1_train, y2_train, xh_train, yh_train

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dimension = 15
    nbr_query = 80
    training_iter = 5
    nbr_repetition = 15
    k_vals = [2]
    g_vals = [6]
    nu_vals = [0.5]
    multi = True
    batch_size = 100
    num_batches = 400
    y1_train, y2_train, xh_train, yh_train = create_data(num_batches*batch_size)


    y1_train = torch.reshape(y1_train, (-1, batch_size, 15))
    y2_train = torch.reshape(y2_train, (-1, batch_size, 15))
    xh_train = torch.reshape(xh_train, (-1, batch_size, 15*15))
    yh_train = torch.reshape(yh_train, (-1, batch_size, 2))

    y1_val, y2_val, xh_val, yh_val = create_data(100)

    hidden_dims = [dimension*dimension, dimension*dimension, dimension*dimension, dimension*dimension, dimension*dimension, 2*dimension, dimension]
    # must pretrain the model before running a repetition, consider saving it to huggingface
    master = hmodel.NN_Hierarchical_Comb(input_dim=dimension*dimension, hidden_dims=hidden_dims, output_dim=2)

    optimizer = torch.optim.Adam(master.parameters(), lr=1e-6)

    num_epochs = 500
    master = nn_pretraining(xh_train, yh_train, xh_val, yh_val, master, master.loss_fn, optimizer, num_epochs, min_delta=0.5)

    master.eval()

    xh_val = xh_val.to(device)
    master = master.to(device)
    yh_val = yh_val.to(device)
    pred = master(xh_val)
    loss = master.loss_fn(yh_val, pred)

    print(f"Validation Loss after {num_epochs}: {loss}")