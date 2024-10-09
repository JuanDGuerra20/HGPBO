"""
Essentially this file builds on the standard models file but implements the hierarchical process
Easier to separate these files because they have similar uses

This file contains the hierarchical specific things, anything that is both will be in the models file
"""

from models import *


class PriorMean(gpytorch.means.Mean):  # PMF
    def __init__(self, prior_map):
        super().__init__()
        self.register_parameter('map', torch.nn.Parameter(prior_map, requires_grad=False))

    def forward(self, input):
        Xmean_1D = [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4]]
        if input.shape[0] == 1:
            first_indice = input[0, 0:2]
            for indices, pins in enumerate(Xmean_1D):
                if (pins[0] == first_indice[0] and pins[1] == first_indice[1]):
                    indice_1 = indices
            second_indice = input[0, 2:4]
            for indices, pins in enumerate(Xmean_1D):
                if (pins[0] == second_indice[0] and pins[1] == second_indice[1]):
                    indice_2 = indices

            new_prior = self.map[indice_1, indice_2].reshape(1, )

        else:
            new_prior = torch.zeros(input.shape[0])
            input = input.long()

            for i in range(input.shape[0]):
                first_indice = input[i, 0:2]
                for indices, pins in enumerate(Xmean_1D):
                    if (pins[0] == first_indice[0] and pins[1] == first_indice[1]):
                        indice_1 = indices
                second_indice = input[i, 2:4]
                for indices, pins in enumerate(Xmean_1D):
                    if (pins[0] == second_indice[0] and pins[1] == second_indice[1]):
                        indice_2 = indices

                new_prior[i] = self.map[indice_1, indice_2]

        return new_prior


class Hierarchical_GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, hierarchical_kernel, prior_map):
        super(Hierarchical_GP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = PriorMean(prior_map)
        self.mean_module.requires_grad = False
        self.covar_module = hierarchical_kernel

    def forward(self, x):
        # print('x',x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def Hoptimize(model, likelihood, training_iter, train_x, train_y, verbose=True):
    """
    Optimize the GP model.

    Parameters:
    - model: GP model.
    - likelihood: Likelihood function.
    - training_iter (int): Number of optimization iterations.
    - train_x (torch.Tensor): Training input data. . If 1D [x, y], if 2D [x1, y1, x2, y2]
    - train_y (torch.Tensor): Training output data: EMG values
    - verbose (bool): Whether to print optimization progress.

    Returns:
    - model: Optimized GP model.
    - likelihood: Optimized likelihood function.
    """

    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        """
        I believe this is where we want to optimize and pass information to the submodules
        """
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.sum().backward(retain_graph=True)
        if verbose:
            print('Iter %d/%d - Loss: %.3f   lengthscale_1: [%.3f , %.3f]   lengthscale_2: [%.3f , %.3f]   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.kernels[0].lengthscale[0][0].item(),
                model.covar_module.kernels[0].lengthscale[0][1].item(),
                model.covar_module.kernels[1].lengthscale[0][0].item(),
                model.covar_module.kernels[1].lengthscale[0][1].item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
    return model, likelihood


def random_initialization(random_sample, emg, trainsC, max_seen_resp, dt):
    """
    Perform random initialization for training points.

    Parameters:
    - random_sample (int): Number of random samples.
    - emg: EMG data.
    - trainsC: Training configuration.

    Returns:
    - X: List of input data.
    - Y: List of output data.
    """
    # Initial random training points
    X = []
    Y = []
    for _ in range(random_sample):
        ch1 = random.choice(CHS)
        ch2 = random.choice(CHS)
        X.append(ch2xy[ch1] + ch2xy[ch2])
        reponse = random.choice(trainsC.get_resp(emg, dt, ch1, ch2).max(axis=1))

        # Add of if statement and max seen resp in the case of several initialization points
        if (reponse > max_seen_resp) or (max_seen_resp == 0):
            max_seen_resp = reponse

        Y.append(reponse / max_seen_resp)  # need for normalization between 0 and 1

    return X, Y


def get_y_mu_point_value(next_query_pins, y_mu, X):
    a = 0
    for indices, pins in enumerate(X):
        if pins[0] == next_query_pins[0] and pins[1] == next_query_pins[1]:  # find pins of (x, y) coordinates in X
            ind = indices
    a = y_mu[ind]
    return a


def make_Hierarchique_prediction(model, x, likelihood):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x))
        return observed_pred


def hierarchical_kernel(kernel_type, model1, model2):
    """
    Creates hierarchical kernel based on input kernel_type. Right now available only for additive kernels

    Notes additive kernels will make the process of
    :param kernel_type: how the hierarchical kernel will implement the sub kernels
    :param model1: first child model
    :param model2: second child model
    :return:
    """
    if kernel_type == 'add_kernel':
        kernel1 = gpytorch.kernels.MaternKernel(nu=2.5, has_lengthscale=True, ard_num_dims=2, input_dim=4,
                                                active_dims=[0, 1])
        kernel1.outputscale = model1.covar_module.outputscale
        kernel1.lengthscale = model1.covar_module.base_kernel.lengthscale

        kernel2 = gpytorch.kernels.MaternKernel(nu=2.5, has_lengthscale=True, ard_num_dims=2, input_dim=4,
                                                active_dims=[2, 3])
        kernel2.outputscale = model2.covar_module.outputscale
        kernel2.lengthscale = model2.covar_module.base_kernel.lengthscale

        hierarchical_kernel = kernel1 + kernel2
    return hierarchical_kernel


def update_kernel_parameters(model, model1, model2):
    """
    Updates the outputscale and lengthscale of ther parent kernel based on the revised child processes
    updates the kernel for the respective children within the hierarchical instance of the children
    :param model: parent model
    :param model1: child model
    :param model2: child model 2
    :return:
    """
    model.covar_module.kernels[0].outputscale = model1.covar_module.outputscale
    model.covar_module.kernels[0].lengthscale = model1.covar_module.base_kernel.lengthscale

    model.covar_module.kernels[1].outputscale = model2.covar_module.outputscale
    model.covar_module.kernels[1].lengthscale = model2.covar_module.base_kernel.lengthscale
    return model


def prediction_model1_1D(model, likelihood, test_x, Xmean, coord_hierarchique, model_number):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    # Make a prediction, observed_pred = likelihood, prediction_mean = y_mu
    observed_pred = make_prediction(model, test_x, likelihood)

    y_mu_matrix = observed_pred.mean

    if model_number == '1':
        coord = coord_hierarchique[0:2]
    elif model_number == '2':
        coord = coord_hierarchique[2:4]

    for indices, pins in enumerate(Xmean):
        if pins[0] == coord[0] and pins[0] == coord[0]:
            indice_value = indices
            y_mu = y_mu_matrix[indice_value]

    return y_mu


def compute_responses(y_mu1, y_mu2, response_hierarchique):
    y_mu1 = torch.as_tensor(y_mu1)
    y_mu2 = torch.as_tensor(y_mu2)
    response_1 = (response_hierarchique - y_mu1) / y_mu2
    response_2 = (response_hierarchique - y_mu2) / y_mu1
    return response_1, response_2


def update_model1_1D(model, likelihood, train_x, train_y, next_query_pins, response1, response2, training_iter=10):
    """
    Used to update the submodel within the hierarchical model
    :param model: the child model that will be updated
    :param likelihood: the likelihood associated to param model
    :param train_x: x input
    :param train_y: y output
    :param next_query_pins: which pins will be queried next
    :param response1: EMG response of first target
    :param response2: EMG response of second target
    :param training_iter: number of training iterations
    :return:
    """

    # Update training data by adding next_query_value to the train dataset
    train_x, train_y = update_training_data(train_x, train_y, next_query_pins[0:2], response1)  # has to be 1D dataset
    train_x, train_y = update_training_data(train_x, train_y, next_query_pins[2:4], response2)  # has to be 1D dataset
    # Update the model with the new training data
    model.set_train_data(train_x, train_y, strict=False)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    model, likelihood = optimize(model, likelihood, training_iter, train_x, train_y, verbose=False)

    return model, likelihood, train_x, train_y
