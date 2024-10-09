from hmodel_synthetic import *


class PriorMeanGPU(gpytorch.means.Mean):  # PMF
    """
    This class works as basically a look-up function from the prior mean map that is passed from the submodules
    to the hierarchical one. Saves a mean map and then when doing inference, the new mean is just looked up for that
    point by looking up the prior map passed from the submodules
    """

    def __init__(self, prior_map, device):
        super().__init__()
        self.register_parameter('map', torch.nn.Parameter(prior_map, requires_grad=False))
        self.device = device

    def forward(self, input):
        Xmean_1D = torch.linspace(0.0, 2.0, self.map.shape[0], device=self.device).double()
        Xmean_2D = torch.zeros((self.map.shape[0], self.map.shape[1], 2), device=self.device).double()

        for i in range(len(Xmean_1D)):
            for j in range(len(Xmean_1D)):
                Xmean_2D[i, j, 0] = Xmean_1D[i]
                Xmean_2D[i, j, 1] = Xmean_1D[j]

        new_prior = torch.zeros(input.shape[0], device=self.device).double()

        for i in range(input.shape[0]):
            x = 0
            flag = True
            while flag:
                y = 0
                while flag and y < Xmean_2D.shape[1]:
                    if Xmean_2D[x][y][0] - input[i][0] < 1e-6 and Xmean_2D[x][y][1] - input[i][1] < 1e-6:
                        indice_1 = x
                        indice_2 = y
                        flag = False
                    y += 1
                x += 1

            new_prior[i] = self.map[indice_1, indice_2]

        return new_prior #.to(self.device)


class GPU_Hierarchical_GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, hierarchical_kernel, prior_map, kernel_op, sub_models, kappa, device=torch.device('cpu')):
        super(GPU_Hierarchical_GP, self).__init__(train_x, train_y, likelihood)

        self.sub_models = sub_models  # This will be useful for creating the training procedure
        self.kernel_op = kernel_op
        self.mean_module = PriorMeanGPU(prior_map, device=device)
        self.mean_module.requires_grad = False
        self.covar_module = hierarchical_kernel
        self.kappa = kappa
        self.device = device

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

class GPU_Changing_Data_UCB_Hierarchical_GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, hierarchical_kernel, prior_map, kernel_op, sub_models, kappa, device=torch.device('cpu')):
        super(GPU_Changing_Data_UCB_Hierarchical_GP, self).__init__(train_x, train_y, likelihood)

        self.sub_models = sub_models  # This will be useful for creating the training procedure
        self.kernel_op = kernel_op
        self.mean_module = PriorMeanGPU(prior_map, device)
        self.mean_module.requires_grad = False
        self.covar_module = hierarchical_kernel
        self.kappa = kappa
        self.device = device

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def Hoptimize(self, likelihood, training_iter, train_x, train_y, device, verbose=False,):
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


class GPU_Efficient_UCB_Hierarchical_GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, hierarchical_kernel, prior_map, kernel_op, sub_models, kappa, device):
        super(GPU_Efficient_UCB_Hierarchical_GP, self).__init__(train_x, train_y, likelihood)

        self.sub_models = sub_models  # This will be useful for creating the training procedure
        self.kernel_op = kernel_op
        self.mean_module = PriorMeanGPU(prior_map, device)
        self.mean_module.requires_grad = False
        self.covar_module = hierarchical_kernel
        self.kappa = kappa

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def Hoptimize(self, likelihood, training_iter, train_x, train_y, verbose=False):
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
        target_loss = loss.detach()

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

            sub_loss = -loss_func(sub_out, sub.train_targets)  # TODO: Currently a duct tape fix
            scaling = sub_loss.detach() / target_loss
            multiplier = contributions[i].divide(norm).detach()

            sub_loss = sub_loss.divide(scaling)
            sub_loss = sub_loss.multiply(multiplier)

            sub_loss.backward(retain_graph=True)
            sub_opt.step()
            sub_opt.zero_grad()