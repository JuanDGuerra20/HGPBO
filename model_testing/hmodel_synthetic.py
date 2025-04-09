"""
Essentially this file builds on the standard models file but implements the hierarchical process
Easier to separate these files because they have similar uses

This file contains the hierarchical specific things, anything that is both will be in the models file
"""
import gpytorch.mlls
import torch
from numpy import dtype
from torch.cuda import device

from synthetic_models import *
from scipy.special import jn
from scipy.optimize import root_scalar
from torch import nn

class HashTable:

    # Create empty bucket list of given size
    def __init__(self, size):
        self.size = size
        self.hash_table = self.create_buckets()

    def create_buckets(self):
        return [[] for _ in range(self.size)]

    # Insert values into hash map
    def set_val(self, key, val):

        # Get the index from the key
        # using hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key to be inserted
            if record_key[0] == key[0] and record_key[1] == key[1] and record_key[2] == key[2]:
                found_key = True
                break

        # If the bucket has same key as the key to be inserted,
        # Update the key value
        # Otherwise append the new key-value pair to the bucket
        if found_key:
            bucket[index] = (key, val)
        else:
            bucket.append((key, val))

    # Return searched value with specific key
    def get_val(self, key):

        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key being searched
            if record_key[0] == key[0] and record_key[1] == key[1]:
                found_key = True
                break

        # If the bucket has same key as the key being searched,
        # Return the value found
        # Otherwise indicate there was no record found
        if found_key:
            return record_val
        else:
            return "No record found"

    # Remove a value with specific key
    def delete_val(self, key):

        # Get the index from the key using
        # hash function
        hashed_key = hash(key) % self.size

        # Get the bucket corresponding to index
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record

            # check if the bucket has same key as
            # the key to be deleted
            if record_key == key:
                found_key = True
                break
        if found_key:
            bucket.pop(index)
        return

    def __str__(self):
        return "".join(str(item) for item in self.hash_table)

class PriorMean(gpytorch.means.Mean):  # PMF
    """
    This class works as basically a look-up function from the prior mean map that is passed from the submodules
    to the hierarchical one. Saves a mean map and then when doing inference, the new mean is just looked up for that
    point by looking up the prior map passed from the submodules
    """

    def __init__(self, prior_map, test_x, device="cpu"):
        super().__init__()
        self.register_parameter('map', torch.nn.Parameter(prior_map.to(device), requires_grad=False))
        self.hash_map = HashTable(np.prod(test_x.shape[:-1]))
        self.device = device

        for i in range(len(test_x)):
            for j in range(len(test_x)):
                self.hash_map.set_val(str(test_x[i][j]), [i, j])

    """def forward(self, input):

        Xmean_1D = torch.linspace(0.0, 2.0, self.map.shape[0]).double()
        Xmean_2D = torch.zeros((self.map.shape[0], self.map.shape[1], 2)).double()

        for i in range(len(Xmean_1D)):
            for j in range(len(Xmean_1D)):
                Xmean_2D[i, j, 0] = Xmean_1D[i]
                Xmean_2D[i, j, 1] = Xmean_1D[j]

        new_prior = torch.zeros(input.shape[0])

        for i in range(input.shape[0]):
            x = 0
            flag = True
            while flag:
                y = 0
                while flag and y < Xmean_2D.shape[1]:
                    if Xmean_2D[x][y][0] == input[i][0] and Xmean_2D[x][y][1] == input[i][1]:
                        indice_1 = x
                        indice_2 = y
                        flag = False
                    y += 1
                x += 1

            new_prior[i] = self.map[indice_1, indice_2]

        return new_prior #.to(self.device)"""
    def forward(self, input):
        new_prior = torch.zeros(input.shape[0], device=self.device)
        for i in range(input.shape[0]):
            look_up = self.hash_map.get_val(str(input[i]))
            new_prior[i] = self.map[look_up[0], look_up[1]]

        return new_prior


class Hierarchical_GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, hierarchical_kernel, prior_map, kernel_op, sub_models, kappa):
        super(Hierarchical_GP, self).__init__(train_x, train_y, likelihood)

        self.sub_models = sub_models  # This will be useful for creating the training procedure
        self.kernel_op = kernel_op
        self.mean_module = PriorMean(prior_map)
        self.mean_module.requires_grad = False
        self.covar_module = hierarchical_kernel
        self.kappa = kappa

    def forward(self, x):
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

        sub_optimizers = []
        sub_losses = []
        for sub in model.sub_models:
            sub.train()
            sub_optimizers.append(torch.optim.Adam(sub.parameters(), lr=0.01))
            sub_losses.append(gpytorch.mlls.ExactMarginalLogLikelihood(sub.likelihood, sub))
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
            # decompose_hkernel_loss(model, loss, train_x)
            loss.sum().backward(retain_graph=True)
            if verbose:
                print(
                    'Iter %d/%d - Loss: %.3f   lengthscale_1: [%.3f , %.3f]   lengthscale_2: [%.3f , %.3f]   noise: %.3f' % (
                        i + 1, training_iter, loss.item(),
                        model.covar_module.kernels[0].lengthscale[0][0].item(),
                        model.covar_module.kernels[0].lengthscale[0][1].item(),
                        model.covar_module.kernels[1].lengthscale[0][0].item(),
                        model.covar_module.kernels[1].lengthscale[0][1].item(),
                        model.likelihood.noise.item()
                    ))
            optimizer.step()

        return model, likelihood

    def mult_penalized_subkernel_loss(self, loss, train_x, sub_optimizers, sub_losses):
        contributions = []

        for i in range(self.sub_models):
            model = self.sub_models[i]

            output = model(train_x[:, i * 2: (i + 1) * 2])

            mean = output.mean
            var = output.var

            cont = mean + self.kappa * var  # Doing something similar to the UCB for the multiplier

            contributions.append(cont)

        if self.kernel_op == 'add_kernel':
            sum = np.sum(contributions)
            contributions[0] = 2 * contributions[0] / sum
            contributions[1] = 2 * contributions[1] / sum

        elif self.kernel_op == 'mult':
            contributions[0] = contributions[0] / contributions[1]
            contributions[1] = contributions[1] / contributions[0]

        for i in range(self.sub_models):
            opt = sub_optimizers[i]
            opt.zero_grad()

            loss1 = loss


class Efficient_UCB_Hierarchical_GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, test_x, likelihood, hierarchical_kernel, prior_map, kernel_op, sub_models, kappa, query_counter=None, device=device):
        super(Efficient_UCB_Hierarchical_GP, self).__init__(train_x, train_y, likelihood)

        self.sub_models = sub_models  # This will be useful for creating the training procedure
        self.kernel_op = kernel_op
        self.mean_module = PriorMean(prior_map, test_x, device=device).to(device)
        self.mean_module.requires_grad = False
        self.covar_module = hierarchical_kernel
        self.kappa = kappa
        self.query_counter = query_counter

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def Hoptimize(self, likelihood, training_iter, train_x, train_y, verbose=True):
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

        self.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
        sub_optimizers = []
        sub_loss_functions = []
        for sub in self.sub_models:
            sub.train()
            sub_optimizers.append(torch.optim.Adam(sub.parameters(), lr=0.01))
            sub_loss_functions.append(gpytorch.mlls.ExactMarginalLogLikelihood(sub.likelihood, sub))

        """
        Don't need this since every hierarchical training step is linked to submodel training
        
        for i, sub in enumerate(model.sub_models):
            sub.train()
            sub_train_x = list(sub.train_inputs)
            sub_train_y = list(sub.train_targets)

            sub_train_x.append(torch.reshape(train_x[:, i], (-1, 1)))
            sub_train_y.append(train_y)
            sub_train_x = torch.stack(sub_train_x)
            sub_train_y = torch.as_tensor(sub_train_y)
            sub.set_train_data(sub_train_x, sub_train_y, strict=False)"""

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            for opt in sub_optimizers:
                # I am guessing, but I guess this is the best spot to zero the gradients
                opt.zero_grad()

            torch.autograd.set_detect_anomaly(True)
            # Output from model
            output = self(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            # decompose_hkernel_loss(model, loss, train_x)
            loss.sum().backward(retain_graph=True)
            if verbose:
                print(
                    'Iter %d/%d - Loss: %.3f   lengthscale_1: [%.3f , %.3f]   lengthscale_2: [%.3f , %.3f]   noise: %.3f' % (
                        i + 1, training_iter, loss.item(),
                        self.covar_module.kernels[0].lengthscale[0][0].item(),
                        self.covar_module.kernels[0].lengthscale[0][1].item(),
                        self.covar_module.kernels[1].lengthscale[0][0].item(),
                        self.covar_module.kernels[1].lengthscale[0][1].item(),
                        self.likelihood.noise.item()
                    ))
            optimizer.step()

            self.simplified_subkernel_loss_sanity(loss, train_x, train_y, sub_optimizers, sub_loss_functions)
            # self.mult_penalized_subkernel_loss(loss, train_x, train_y, sub_optimizers, sub_loss_functions)

        return self, likelihood

    def simplified_subkernel_loss_sanity(self, loss, train_x, train_y, sub_optimizers, sub_loss_functions):
        """
        This is the kernel loss but using the full training data that does not changing
        :param loss:
        :param train_x:
        :param train_y:
        :param sub_optimizers:
        :param sub_loss_functions:
        :return:
        """
        target_loss = loss.detach().numpy()

        for i, sub in enumerate(self.sub_models):
            contributions = torch.zeros((len(self.sub_models),))
            for j, sub_sub in enumerate(self.sub_models):
                # Calculating the contributions based on the UCB assumption
                train_out = sub_sub(sub_sub.train_inputs[0])

                tense = sub_sub.likelihood(train_out).mean
                std = sub_sub.likelihood(train_out).stddev * self.kappa
                tense = tense + std
                contributions[j] = torch.mean(tense)

            norm = contributions.sum()



            sub.train()
            sub_out = sub(sub.train_inputs[0])  # TODO: currently a duct tape fix

            loss_func = sub_loss_functions[i]
            sub_opt = sub_optimizers[i]

            sub_loss = -loss_func(sub_out, sub.train_targets)   # TODO: Currently a duct tape fix
            scaling = sub_loss.detach() / target_loss
            multiplier = contributions[i].divide(norm).detach()

            sub_loss = sub_loss.divide(scaling)
            sub_loss = sub_loss.multiply(multiplier)

            sub_loss.backward(retain_graph=True)
            sub_opt.step()
            sub_opt.zero_grad()

    def increment_q_n(self, query_c, query, domain):
        for x in range(len(domain)):
            for y in range(len(domain[x])):
                if domain[x][y][0] == query[0] and domain[x][y][1] == query[1]:
                    query_c[x*(len(domain[x])) + y] += 1
                    break
        self.query_counter = query_c
        return query_c

class Lossless_Efficient_UCB_Hierarchical_GP(Efficient_UCB_Hierarchical_GP):
    def __init__(self, train_x, train_y, test_x, likelihood, hierarchical_kernel, prior_map, kernel_op, sub_models, kappa, query_counter=None, device="cpu"):
        super().__init__(train_x, train_y, test_x, likelihood, hierarchical_kernel, prior_map, kernel_op, sub_models, kappa, query_counter=query_counter, device=device)

    def Hoptimize(self, likelihood, training_iter, train_x, train_y, verbose=True):
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

        self.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
        sub_optimizers = []
        sub_loss_functions = []
        for sub in self.sub_models:
            sub.train()
            sub_optimizers.append(torch.optim.Adam(sub.parameters(), lr=0.01))
            sub_loss_functions.append(gpytorch.mlls.ExactMarginalLogLikelihood(sub.likelihood, sub))

        """
        Don't need this since every hierarchical training step is linked to submodel training

        for i, sub in enumerate(model.sub_models):
            sub.train()
            sub_train_x = list(sub.train_inputs)
            sub_train_y = list(sub.train_targets)

            sub_train_x.append(torch.reshape(train_x[:, i], (-1, 1)))
            sub_train_y.append(train_y)
            sub_train_x = torch.stack(sub_train_x)
            sub_train_y = torch.as_tensor(sub_train_y)
            sub.set_train_data(sub_train_x, sub_train_y, strict=False)"""

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            for opt in sub_optimizers:
                # I am guessing, but I guess this is the best spot to zero the gradients
                opt.zero_grad()

            torch.autograd.set_detect_anomaly(True)
            # Output from model
            output = self(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            # decompose_hkernel_loss(model, loss, train_x)
            loss.sum().backward(retain_graph=True)
            if verbose:
                print(
                    'Iter %d/%d - Loss: %.3f   lengthscale_1: [%.3f , %.3f]   lengthscale_2: [%.3f , %.3f]   noise: %.3f' % (
                        i + 1, training_iter, loss.item(),
                        self.covar_module.kernels[0].lengthscale[0][0].item(),
                        self.covar_module.kernels[0].lengthscale[0][1].item(),
                        self.covar_module.kernels[1].lengthscale[0][0].item(),
                        self.covar_module.kernels[1].lengthscale[0][1].item(),
                        self.likelihood.noise.item()
                    ))
            optimizer.step()
        return self, likelihood

class Changing_Data_UCB_Hierarchical_GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, hierarchical_kernel, prior_map, kernel_op, sub_models, kappa):
        super(Changing_Data_UCB_Hierarchical_GP, self).__init__(train_x, train_y, likelihood)

        self.sub_models = sub_models  # This will be useful for creating the training procedure
        self.kernel_op = kernel_op
        self.mean_module = PriorMean(prior_map)
        self.mean_module.requires_grad = False
        self.covar_module = hierarchical_kernel
        self.kappa = kappa

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def Hoptimize(self, likelihood, training_iter, train_x, train_y, verbose=True):
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

        self.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
        sub_optimizers = []
        sub_loss_functions = []
        for sub in self.sub_models:
            sub.train()
            sub_optimizers.append(torch.optim.Adam(sub.parameters(), lr=0.01))
            sub_loss_functions.append(gpytorch.mlls.ExactMarginalLogLikelihood(sub.likelihood, sub))

        """
        Don't need this since every hierarchical training step is linked to submodel training

        for i, sub in enumerate(model.sub_models):
            sub.train()
            sub_train_x = list(sub.train_inputs)
            sub_train_y = list(sub.train_targets)

            sub_train_x.append(torch.reshape(train_x[:, i], (-1, 1)))
            sub_train_y.append(train_y)
            sub_train_x = torch.stack(sub_train_x)
            sub_train_y = torch.as_tensor(sub_train_y)
            sub.set_train_data(sub_train_x, sub_train_y, strict=False)"""

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            for opt in sub_optimizers:
                # I am guessing, but I guess this is the best spot to zero the gradients
                opt.zero_grad()

            torch.autograd.set_detect_anomaly(True)
            # Output from model
            output = self(train_x)# .to(device)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            # decompose_hkernel_loss(model, loss, train_x)
            loss.sum().backward(retain_graph=True)
            if verbose:
                print(
                    'Iter %d/%d - Loss: %.3f   lengthscale_1: [%.3f , %.3f]   lengthscale_2: [%.3f , %.3f]   noise: %.3f' % (
                        i + 1, training_iter, loss.item(),
                        self.covar_module.kernels[0].lengthscale[0][0].item(),
                        self.covar_module.kernels[0].lengthscale[0][1].item(),
                        self.covar_module.kernels[1].lengthscale[0][0].item(),
                        self.covar_module.kernels[1].lengthscale[0][1].item(),
                        self.likelihood.noise.item()
                    ))
            optimizer.step()

            self.simplified_subkernel_loss_sanity(loss, train_x, train_y, sub_optimizers, sub_loss_functions)
            # self.mult_penalized_subkernel_loss(loss, train_x, train_y, sub_optimizers, sub_loss_functions)

        return self, likelihood

    def simplified_subkernel_loss_sanity(self, loss, train_x, train_y, sub_optimizers, sub_loss_functions):
        """
        This loss propagation changes the data
        :param loss:
        :param train_x:
        :param train_y:
        :param sub_optimizers:
        :param sub_loss_functions:
        :return:
        """

        target_loss = loss.detach()

        for i, sub in enumerate(self.sub_models):
            contributions = torch.zeros((len(self.sub_models),))
            for j, sub_sub in enumerate(self.sub_models):
                # Calculating the contributions based on the UCB assumption
                sub_sub.set_train_data(train_x[:, j], train_y)

                train_out = sub_sub(sub_sub.train_inputs[0])

                tense = sub_sub.likelihood(train_out).mean
                std = sub_sub.likelihood(train_out).stddev * self.kappa
                tense = tense + std
                contributions[j] = torch.mean(tense)

            norm = contributions.sum()

            sub.train()
            sub_out = sub(sub.train_inputs[0])  # TODO: currently a duct tape fix

            loss_func = sub_loss_functions[i]
            sub_opt = sub_optimizers[i]

            sub_loss = -loss_func(sub_out, sub.train_targets)  # TODO: Currently a duct tape fix

            scaling = sub_loss.detach() / target_loss
            multiplier = contributions[i].divide(norm).detach()

            sub_loss = sub_loss.divide(scaling)
            sub_loss = sub_loss.multiply(multiplier)

            sub_loss.backward(retain_graph=True)
            sub_opt.step()
            sub_opt.zero_grad()

class NN_Hierarchical_Comb(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self.output_dim = output_dim

        self.mag_loss = nn.MSELoss()

        print(f"using device: {self.device}")

        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
        )
        self.linear_stack.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        self.linear_stack.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            self.linear_stack.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            self.linear_stack.append(nn.ReLU())
            self.linear_stack.append(nn.Dropout(0.4))
        self.linear_stack.append((nn.Linear(self.hidden_dims[-1], self.output_dim)))

    def euclid_derivative(self, y_true, y_pred):
        x_true_locs, y_true_locs = self.get_x_y_loc(y_true)
        x_pred_locs, y_pred_locs = self.get_x_y_loc(y_pred)
        euclid_loss = torch.sub((x_pred_locs + y_pred_locs), (x_true_locs + y_true_locs))
        euclid_loss = torch.div(euclid_loss,
                                torch.sqrt((x_true_locs - x_pred_locs) ** 2 + (y_true_locs - y_pred_locs) ** 2))
        return torch.where(euclid_loss == torch.nan, euclid_loss, 0)

    def euclid_loss(self, y_true, y_pred):
        x_true_locs, y_true_locs = self.get_x_y_loc(y_true)
        x_pred_locs, y_pred_locs = self.get_x_y_loc(y_pred)

        x_diff = torch.pow(torch.sub(x_true_locs, x_pred_locs), 2)
        y_diff = torch.pow(torch.sub(y_true_locs, y_pred_locs), 2)

        loss = torch.sqrt(torch.add(x_diff, y_diff))

        return loss

    def loss_fn(self, y_true, y_pred):

        euclid_loss = self.euclid_loss(y_true, y_pred)
        y_true_mag = y_true[:,0]
        y_pred_mag = y_pred[:,0]

        mag = self.mag_loss(y_true_mag, y_pred_mag)

        loss = torch.add(euclid_loss, mag)
        return torch.mean(loss)

    def get_x_y_loc(self, y):
        ind_dim = np.sqrt(self.input_dim)
        y_true_spat = y[:, 1]

        y_true_locs = torch.remainder(y_true_spat, ind_dim)
        x_true_locs = torch.div(y_true_spat - y_true_locs, ind_dim)

        return x_true_locs, y_true_locs

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

class NN_Hierarchical_Comb(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self.output_dim = output_dim

        self.mag_loss = nn.MSELoss()

        print(f"using device: {self.device}")

        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
        )
        self.linear_stack.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        self.linear_stack.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            self.linear_stack.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            self.linear_stack.append(nn.ReLU())
            self.linear_stack.append(nn.Dropout(0.4))
        self.linear_stack.append((nn.Linear(self.hidden_dims[-1], self.output_dim)))

    def euclid_derivative(self, y_true, y_pred):
        x_true_locs, y_true_locs = self.get_x_y_loc(y_true)
        x_pred_locs, y_pred_locs = self.get_x_y_loc(y_pred)
        euclid_loss = torch.sub((x_pred_locs + y_pred_locs), (x_true_locs + y_true_locs))
        euclid_loss = torch.div(euclid_loss,
                                torch.sqrt((x_true_locs - x_pred_locs) ** 2 + (y_true_locs - y_pred_locs) ** 2))
        return torch.where(euclid_loss == torch.nan, euclid_loss, 0)

    def euclid_loss(self, y_true, y_pred):
        x_true_locs, y_true_locs = self.get_x_y_loc(y_true)
        x_pred_locs, y_pred_locs = self.get_x_y_loc(y_pred)

        x_diff = torch.pow(torch.sub(x_true_locs, x_pred_locs), 2)
        y_diff = torch.pow(torch.sub(y_true_locs, y_pred_locs), 2)

        loss = torch.sqrt(torch.add(x_diff, y_diff))

        return loss

    def loss_fn(self, y_true, y_pred):

        euclid_loss = self.euclid_loss(y_true, y_pred)
        y_true_mag = y_true[:,0]
        y_pred_mag = y_pred[:,0]

        mag = self.mag_loss(y_true_mag, y_pred_mag)

        loss = torch.add(euclid_loss, mag)
        return torch.mean(loss)

    def get_x_y_loc(self, y):
        ind_dim = np.sqrt(self.input_dim)
        y_true_spat = y[:, 1]

        y_true_locs = torch.remainder(y_true_spat, ind_dim)
        x_true_locs = torch.div(y_true_spat - y_true_locs, ind_dim)

        return x_true_locs, y_true_locs

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

class NN_Hierarchical_Comb_NoMag(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        print(f"using device: {self.device}")

        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
        )
        self.linear_stack.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        self.linear_stack.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            self.linear_stack.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            self.linear_stack.append(nn.ReLU())
            self.linear_stack.append(nn.Dropout(0.6))
        self.linear_stack.append(nn.Linear(self.hidden_dims[-1], int((self.input_dim/2)**2)))

    def euclid_derivative(self, y_true, y_pred):
        x_true_locs, y_true_locs = self.get_x_y_loc(y_true)
        x_pred_locs, y_pred_locs = self.get_x_y_loc(y_pred)
        euclid_loss = torch.sub((x_pred_locs + y_pred_locs), (x_true_locs + y_true_locs))
        euclid_loss = torch.div(euclid_loss,
                                torch.sqrt((x_true_locs - x_pred_locs) ** 2 + (y_true_locs - y_pred_locs) ** 2))
        return torch.where(euclid_loss == torch.nan, euclid_loss, 0)

    def euclid_loss(self, y_true, y_pred):
        x_true_locs, y_true_locs = self.get_x_y_loc(y_true)
        x_pred_locs, y_pred_locs = self.get_x_y_loc(y_pred)

        x_diff = torch.pow(torch.sub(x_true_locs, x_pred_locs), 2)
        y_diff = torch.pow(torch.sub(y_true_locs, y_pred_locs), 2)

        loss = torch.sqrt(torch.add(x_diff, y_diff))

        return loss

    def loss_fn(self, y_true, y_pred):

        euclid_loss = self.euclid_loss(y_true, y_pred)

        return torch.mean(euclid_loss)

    def get_x_y_loc(self, y):

        loc = torch.argmax(y, dim=1)
        ind_dim = self.input_dim/2

        y_locs = torch.remainder(loc, ind_dim).clone().detach().requires_grad_(True)
        x_locs = torch.div(loc - y_locs, ind_dim).clone().detach().requires_grad_(True)

        return x_locs, y_locs

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


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
        if pins == next_query_pins:  # find pins of (x, y) coordinates in X
            ind = indices
            break
    a = y_mu[ind]
    return a


def make_Hierarchique_prediction(model, x, likelihood):
    model.eval()
    likelihood.eval()
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

    kernel1 = gpytorch.kernels.MaternKernel(nu=2.5, has_lengthscale=True, ard_num_dims=1, input_dim=2,
                                            active_dims=[0])
    kernel1.outputscale = model1.covar_module.outputscale
    kernel1.lengthscale = model1.covar_module.base_kernel.lengthscale

    kernel2 = gpytorch.kernels.MaternKernel(nu=2.5, has_lengthscale=True, ard_num_dims=1, input_dim=2,
                                            active_dims=[1])
    kernel2.outputscale = model2.covar_module.outputscale
    kernel2.lengthscale = model2.covar_module.base_kernel.lengthscale

    if kernel_type == 'add_kernel':
        hierarchical_kernel = kernel1 + kernel2

    elif kernel_type == 'mult':
        hierarchical_kernel = kernel1 * kernel2

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


def update_model1_1D(model, likelihood, train_x, train_y, next_query_pin, response1, training_iter=10):
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
    train_x, train_y = update_training_data(train_x, train_y, next_query_pin, response1)  # has to be 1D dataset
    # Update the model with the new training data
    model.set_train_data(train_x, train_y, strict=False)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    model, likelihood = optimize(model, likelihood, training_iter, train_x, train_y, verbose=False)

    return model, likelihood, train_x, train_y

def update_model1_1D_max_seen(model, likelihood, train_x, train_y, next_query_pin, response1, env, training_iter=10):
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
    train_x, train_y = model.update_training_data(train_x, train_y, next_query_pin, response1, env)  # has to be 1D dataset
    # Update the model with the new training data

    div_y = train_y.clone()
    if model.env_max_seen != 0 and model.bif_max_seen != 0:
        div_y[model.env_ind] = div_y[model.env_ind]/abs(model.env_max_seen)
        div_y[model.bif_ind] = div_y[model.bif_ind]/abs(model.bif_max_seen)
    else:
        div_y[model.env_ind] = div_y[model.env_ind]/1e-5
        div_y[model.bif_ind] = div_y[model.bif_ind]/1e-5

    model.set_train_data(train_x, div_y, strict=False)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    model, likelihood = optimize(model, likelihood, training_iter, train_x, div_y, verbose=False)

    return model, likelihood, train_x, train_y


def create_prior_map(G, model1, model2, test_x):
    prior_map = torch.zeros(10, 10)
    # Make a prediction, observed_pred = likelihood
    with gpytorch.settings.lazily_evaluate_kernels(state=False):
        # Prediction of Hierarchical Model
        observed_pred1 = make_prediction(model1, test_x, model1.likelihood)
        observed_pred2 = make_prediction(model2, test_x, model2.likelihood)

    y_mu1 = observed_pred1.mean
    y_mu2 = observed_pred2.mean

    for i in range(len(prior_map)):
        for j in range(len(prior_map)):
            if G[0] == '+':
                prior_map[i][j] = y_mu1[i] + y_mu2[j]
            elif G[0] == '*':
                prior_map[i][j] = y_mu1[i] * y_mu2[j]

    prior_map_max = torch.max(prior_map)
    prior_map = prior_map / prior_map_max

    return prior_map


def create_nested_kern(G, train_x, train_y, test_x, trainsC):
    """
    This function allows for the creation of a more complex hierarchical structure using recursion
    Graph G will get decomposed into a hierarchy of kernels/models
    Currently only supports one hierarchy due to data creation
    Puts all of the models into Eval mode

    :param G: A graph in made through array structure  of form [+, [*, [+], [+]], [+]]
    :param train_x: x data for the definition of the model
    :param train_y: y data for the definition of the model
    :param test_x: x test data for the definition of the hierarchical model
    :return: model hierarchy where each kernel is accessible through the covar_module
    """

    if len(G) == 1:
        # Diverges from Thomas by creating two different models in each hierarchy
        # should it be one base model for all of them???
        likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
        model1 = ExactGPModel(train_x, train_y, likelihood1)  # we must guarantee that train_x and train_y are correct

        model1.eval()
        likelihood1.eval()

        h_kernel = hierarchical_kernel(G[0], model1, model1)

        prior_map = create_prior_map(G, model1, model1, test_x)

        return h_kernel, [model1, model1], prior_map

    else:

        k1, sub_models1, prior1 = create_nested_kern(G[1], train_x[1], train_y[1], test_x[1], test_y[1], trainsC)
        train_x_2D, train_y_2D = random_initialization(nbr_rdm_points_ini, EMG, trainsC,
                                                       0, DT)
        likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
        m1 = Hierarchical_GP(train_x_2D, train_y_2D, likelihood1, k1, prior1)

        k2, sub_models2, prior2 = create_nested_kern(G[2], train_x[2], train_y[2], test_x[2], test_y[2], trainsC)

        train_x_2D, train_y_2D = random_initialization(nbr_rdm_points_ini, EMG, trainsC,
                                                       0, DT)
        likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
        m2 = Hierarchical_GP(train_x_2D, train_y_2D, likelihood2, k2, prior2)

        h_kernel = hierarchical_kernel(G[0], m1, m2)

        prior_map = create_prior_map(G, m1, m2, test_x)

        return h_kernel, [[m1, sub_models1], [m2, sub_models2]], prior_map


def inverse_bessel(n, y, x0=1.0):
    """
    Find x such that J_n(x) = y for Bessel function of the first kind of order n.
    Uses the Newton Raphson model of approximation with derivatives to find the "inverse" of the Bessel function
    Note that this is simply an approximation of the inverse and may not actually find the function

    Parameters:
    n (int): Order of the Bessel function.
    y (float): Value for which to find the inverse (the output of the rest of the decomposition equation)
    x0 (float): Initial guess for the root finding algorithm.

    Returns:
    float: Value of x such that J_n(x) = y.
    """
    # Define the function for which we are finding the root
    func = lambda x: jn(n, x) - y

    # Use a root-finding algorithm to find the root
    result = root_scalar(func, x0=0, method='newton')

    if result.converged:
        return result.root
    else:
        raise ValueError("Root finding did not converge")


def decompose_hkernel_loss(model, loss, train_x):
    """
    Here we are trying to separate the signal for the loss and break that down to be passed into the submodules within
    one time step

    based on the conversion of the matern matrix
    :param model: the hierarchical model that contains different kernels
    :param loss: the MLL loss from training that we want to break down
    :return: a tuple containing the recalibrated loss for the two children
    """

    c1 = model.covar_module.kernels[0]
    c2 = model.covar_module.kernels[1]

    kv1_values = []
    kv2_values = []

    for i in range(len(train_x)):
        train = train_x[i:i + 2]
        g = math.gamma(c1.nu)
        k1 = c1(train)  # automatically separates into the subdimensions
        p1 = k1 * g / (np.exp2(1 - c1.nu) * (c1.outputscale ** 2))
        p2 = (math.sqrt(2 * c1.outputscale) * math.dist(train[0, :2], train[1, :2]) / c1.lengthscale) ** (-c1.nu)
        kv1 = p1 * p2
        kv1 = kv1.tensor.detach().numpy()
        kv1 = inverse_bessel(2, kv1[0][0])
        kv1_values.append(kv1)

        # Use numerical computing approximation to find the roots of the method (this is approximately the inverse)

        g = math.gamma(c2.nu)
        k2 = c2(train)  # automatically separates into the subdimensions
        p1 = k2 * g / (np.exp2(1 - c2.nu) * (c2.outputscale ** 2))
        p2 = (math.sqrt(2 * c2.outputscale) * math.sqrt(math.dist(train[0, 2:], train[1, 2:])) / c2.lengthscale) ** (
            -c2.nu)
        kv2 = p1 * p2
        kv2_values.append(kv2)  # NEED TO FURTHER BREAK THIS DOWN

        # Use numerical computing approximation to find the roots of the method (this is approximately the inverse)

    # note total_cover = covar_1 + covar_2 as defined
