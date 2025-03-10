import hmodel_synthetic as hmodel
from dataset_actions import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def nn_pretraining(train_x, train_y, val_x, val_y, model, loss_fn, optimizer, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    model = model.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    torch.cuda.synchronize()

    model.train()

    for epoch in range(num_epochs):
        t = tqdm(enumerate(zip(train_x, train_y)), postfix=[0, 0])
        for batch, (X,y) in t:
            pred = model(X)
            train_loss = loss_fn(y, pred)
            if batch % 100 == 0:
                val_pred = model(val_x)
                val_loss = loss_fn(val_y, val_pred)
                t.postfix[0] = round(train_loss.item(), 4)
                t.postfix[1] = round(val_loss.item(), 4)
                t.update()

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch}/{num_epochs} complete")

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
    dimension = 15
    nbr_query = 80
    training_iter = 5
    nbr_repetition = 15
    k_vals = [2]
    g_vals = [6]
    nu_vals = [0.5]
    multi = True

    y1_train, y2_train, xh_train, yh_train = create_data(200000)

    y1_train = torch.reshape(y1_train, (-1, 200, 15))
    y2_train = torch.reshape(y2_train, (-1, 200, 15))
    xh_train = torch.reshape(xh_train, (-1, 200, 15*15))
    yh_train = torch.reshape(yh_train, (-1, 200, 2))

    y1_val, y2_val, xh_val, yh_val = create_data(10000)

    hidden_dims = [dimension*dimension, dimension*dimension, dimension]
    # must pretrain the model before running a repetition, consider saving it to huggingface
    master = hmodel.NN_Hierarchical_Comb(input_dim=dimension*dimension, hidden_dims=hidden_dims, output_dim=2)

    optimizer = torch.optim.Adam(master.parameters(), lr=0.001)

    num_epochs = 100
    master = nn_pretraining(xh_train, yh_train, xh_val, yh_val, master, master.loss_fn, optimizer, num_epochs)

    loss = master.loss_fn(yh_train, yh_train)

    print(loss)