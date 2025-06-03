import matplotlib.pyplot as plt
import torch
from nn_models import *
import warnings

def training_procedure(model, optimizer, scheduler, loss_fn, X, y, y_hier, num_epochs, tolerance=10):
    model.train()
    if model.device != torch.device('cpu'):
        torch.cuda.synchronize()

    X = X.to(model.device)
    y = y.to(model.device)
    X = X.to(torch.float)
    y = y.to(torch.float)
    #scheduler.to(model.device)
    y_hier = y_hier.to(model.device)
    losses = []
    exploration_score_tracker = []
    heatmaps = []

    pbar = tqdm(range(num_epochs))
    early_stopping = False
    for epoch in pbar:
        pred = model(X)
        train_loss = loss_fn((pred).view(-1), y)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(train_loss)

        losses.append(train_loss.item())
        pred = model(test_x_hier)
        heatmaps.append(pred.detach().cpu().numpy())
        ground_truth_max_hier = torch.argmax(pred)

        exploration_score_2D = get_exploration_score(pred, ground_truth_max_hier, y_hier)
        exploration_score_tracker.append(exploration_score_2D.cpu())
        if epoch % 10 == 0:
            pbar.set_postfix({'loss': losses[-1], 'exploration': exploration_score_2D.item()})
            #print(f"Loss after {epoch} epochs: {losses[-1]:.4f}\t Exploration score: {exploration_score_2D:.4f}")
        if epoch > tolerance:
            for i in range(tolerance):
                if losses[-i] < losses[-i-1]:
                    early_stopping = False
                    break
                else:
                    early_stopping = True

            if early_stopping:
                break

    return losses, exploration_score_tracker, heatmaps

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

    for q in tqdm(range(nbr_query)):

        if q == 0:
            train_x_hier, train_y_hier = hierarchical_select_random_queries(rand_init, x_hier, y_hier, seed=seed, noise=noise)
            for i in range(len(train_x_hier)):
                query_counter = model.increment_q_n(train_x_hier[i], x_hier)
        acquisition_map = model.get_acquisition_map(test_x_hier)

        next_query_pins = torch.tensor(get_next_query_pins(acquisition_map, test_x_hier))


        next_query_value_random, next_query_value_mean = get_next_query_value(next_query_pins,
                                                                                     test_x_hier,
                                                                                     y_hier, noise=noise)
        response = torch.tensor(next_query_value_random, device=device)


        train_x_hier, train_y_hier = update_training_data(train_x_hier, train_y_hier, next_query_pins,
                                                          response)
        query_counter = model.increment_q_n(train_x_hier[-1], x_hier)

        training_procedure(model, optimizer, loss_fn, train_x_hier, train_y_hier, training_iter)

        hierar_y_mu = model(test_x_hier)
        ground_truth_max_hier = torch.argmax(hierar_y_mu)

        exploration_score_2D = get_exploration_score(hierar_y_mu, ground_truth_max_hier, y_hier)

        better_exploration_score.append(exploration_score_2D)

        """#plotting the heatmaps
        heatmap = hierar_y_mu.reshape(10,10)
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
    training_iter = 1000
    nbr_query = 1
    rand_init = 1000
    lr = [0.2]
    l2 = [0]
    over_explor = []
    over_r2 = []

    data_name, data_creation_func, eps = get_dataset_info(2)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    print(f"Using Device {device}")
    test_x_hier = test_x_hier.to(device)
    query_counter = torch.ones(dimension**2)
    seed = False
    noise = 0
    train_x_hier, train_y_hier = hierarchical_select_random_queries(rand_init, x_hier, y_hier, seed=seed,
                                                                    noise=noise)

    for alpha in lr:
        for beta in l2:
            master = NN_baseline([dimension**2], test_x_hier, query_counter, device=device).to(device)

            optimizer = torch.optim.Adam(master.parameters(), lr=alpha, weight_decay=beta)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
            loss_fn = nn.MSELoss()

            loss, explor, heatmaps = training_procedure(master, optimizer, scheduler, loss_fn, x_hier.reshape((100, 2)), y_hier.reshape((100,)), y_hier, training_iter, tolerance=1000)

            plt.plot(loss, label=f"lr_{alpha}_beta_{beta}")
            plt.title(f"Training Loss for different LR and Weight Decay")
            plt.legend()
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.savefig(f"{data_name}/{model_name}/{folder_of_the_day}/png/training_loss_lr_{alpha}_weight_decay_{beta}.png")
            plt.close()

            plt.plot(list(range(len(explor))), explor)
            plt.ylim(-0.1, 1.1)
            plt.xlabel(f"Query Number")
            plt.ylabel('Exploration Score')
            plt.savefig(
                f"{data_name}/{model_name}/{folder_of_the_day}/differentiable_plots/exploration_score_over_lr_{alpha}_training_penalty_{beta}.png")
            plt.close()
            over_explor.append(explor)
            over_r2.append(heatmap)

            final_heatmap = heatmaps[-1].reshape(10, 10)
            plt.imshow(final_heatmap)
            plt.title(f"NN Heatmap for lr {alpha} and beta {beta}")
            plt.colorbar()
            plt.savefig(
                f"{data_name}/{model_name}/{folder_of_the_day}/png/final_heatmap_lr_{alpha}_weight_decay_{beta}.png")
            plt.close()

    """plt.plot(explor, label="Training Exploration")
    plt.title(f"Training Loss for LR {alpha} Weight Decay {beta}")
    plt.ylabel("Exploration_Score")
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Epoch")
    plt.savefig(
        f"{data_name}/{model_name}/{folder_of_the_day}/png/Exploration_Score_lr_{alpha}_weight_decay_{beta}.png")
    plt.close()"""
    """master, exploration_score, heatmap = run_repetition(master, optimizer, loss_fn, nbr_query, training_iter, rand_init, data_creation_func, dimension, eps, model_name, folder_of_the_day, data_name, seed=False, noise=0.1)

    print(f"exploration: {exploration_score}")

    plt.plot(list(range(len(exploration_score))), exploration_score)
    plt.ylim(-0.1, 1.1)
    plt.xlabel(f"Query Number")
    plt.ylabel('Exploration Score')
    plt.savefig(f"{data_name}/{model_name}/{folder_of_the_day}/differentiable_plots/exploration_score_over_lr_{alpha}__training_penalty_{beta}.png")
    plt.close()
    over_explor.append(exploration_score)
    over_r2.append(heatmap)

    final_heatmap = heatmap[-1].reshape(10, 10)
    plt.imshow(final_heatmap.detach().numpy())
    plt.title(f"NN Heatmap for l2 penalty {beta}")
    plt.colorbar()
    plt.savefig(f"{data_name}/{model_name}/{folder_of_the_day}/png/final_heatmap_lr_{alpha}_weight_decay_{beta}.png")
    plt.close()"""
