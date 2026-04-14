"""
Hierarchical GP model adapted for the GAN experiment.

The GAN experiment has 2D children (generator: lr_gen x z_dim, discriminator: lr_disc x dropout)
and a 4D parent (lr_gen, z_dim, lr_disc, dropout).

Adapted from hmodel_synthetic.py with the following key changes:
- HashTable uses string-based key comparison (generic, works for any dimension)
- PriorMean indexes a (49, 49) prior map via flat test_x_hier (2401, 4)
- hierarchical_kernel builds additive kernel from two 2D child kernels
  with active_dims [0,1] and [2,3]
- Lossless_Efficient_UCB_Hierarchical_GP is the main BIF parent model
- Helper functions adapted for 2D children instead of 1D
"""

import torch
import numpy as np
import gpytorch
import gpytorch.mlls


# ---------------------------------------------------------------------------
# HashTable — string-based key comparison (works for any dimension)
# ---------------------------------------------------------------------------

class HashTable:

    def __init__(self, size):
        self.size = size
        self.hash_table = self.create_buckets()

    def create_buckets(self):
        return [[] for _ in range(self.size)]

    def set_val(self, key, val):
        hashed_key = hash(key) % self.size
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
            if record_key == key:  # string comparison — works for any dimension
                found_key = True
                break

        if found_key:
            bucket[index] = (key, val)
        else:
            bucket.append((key, val))

    def get_val(self, key):
        hashed_key = hash(key) % self.size
        bucket = self.hash_table[hashed_key]

        for index, record in enumerate(bucket):
            record_key, record_val = record
            if record_key == key:
                return record_val

        return "No record found"

    def delete_val(self, key):
        hashed_key = hash(key) % self.size
        bucket = self.hash_table[hashed_key]

        found_key = False
        for index, record in enumerate(bucket):
            record_key, record_val = record
            if record_key == key:
                found_key = True
                break
        if found_key:
            bucket.pop(index)
        return

    def __str__(self):
        return "".join(str(item) for item in self.hash_table)


# ---------------------------------------------------------------------------
# PriorMean — look-up from the (49, 49) prior map via 4D test points
# ---------------------------------------------------------------------------

class PriorMean(gpytorch.means.Mean):
    """
    Look-up mean function for the hierarchical GP.

    Given a prior_map of shape (49, 49) and a flat test_x_hier of shape (2401, 4),
    each input point is hashed to its (i, j) index in the prior map:
        idx in test_x_hier -> i = idx // 49 (child1 config), j = idx % 49 (child2 config)
    """

    def __init__(self, prior_map, test_x_hier, device="cpu"):
        super().__init__()
        self.register_parameter('map', torch.nn.Parameter(prior_map.to(device), requires_grad=False))
        self.hash_map = HashTable(len(test_x_hier))
        self.device = device

        # Build hash map: each 4D test point -> (i, j) indices in the prior_map
        # test_x_hier is (2401, 4), prior_map is (49, 49)
        # test_x_hier[idx] corresponds to i = idx // 49, j = idx % 49
        for idx in range(len(test_x_hier)):
            i = idx // 49  # child1 config index
            j = idx % 49   # child2 config index
            key = str(test_x_hier[idx])
            self.hash_map.set_val(key, [i, j])

    def forward(self, input):
        new_prior = torch.zeros(input.shape[0], device=self.device)
        for i in range(input.shape[0]):
            look_up = self.hash_map.get_val(str(input[i]))
            new_prior[i] = self.map[look_up[0], look_up[1]]
        return new_prior


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

def hierarchical_kernel(kernel_type, model1, model2):
    """
    Create additive (or multiplicative) kernel from two 2D child models.

    Child 1 kernel operates on dims [0, 1] (gen LR, z_dim).
    Child 2 kernel operates on dims [2, 3] (disc LR, dropout).

    Parameters:
        kernel_type: 'add_kernel' or 'mult'
        model1: child GP model for generator (2D)
        model2: child GP model for discriminator (2D)

    Returns:
        Combined kernel (AdditiveKernel or ProductKernel)
    """
    # Child 1 kernel: operates on dims [0, 1] (gen LR, z_dim)
    kernel1 = gpytorch.kernels.MaternKernel(nu=2.5, has_lengthscale=True,
                                             ard_num_dims=2, active_dims=[0, 1])
    kernel1.lengthscale = model1.covar_module.base_kernel.lengthscale

    # Child 2 kernel: operates on dims [2, 3] (disc LR, dropout)
    kernel2 = gpytorch.kernels.MaternKernel(nu=2.5, has_lengthscale=True,
                                             ard_num_dims=2, active_dims=[2, 3])
    kernel2.lengthscale = model2.covar_module.base_kernel.lengthscale

    if kernel_type == 'add_kernel':
        return kernel1 + kernel2
    elif kernel_type == 'mult':
        return kernel1 * kernel2


def update_kernel_parameters(model, model1, model2):
    """
    Update the parent kernel lengthscales from the revised child processes.

    Parameters:
        model: parent hierarchical GP model
        model1: child 1 (generator) GP model
        model2: child 2 (discriminator) GP model

    Returns:
        model: updated parent model
    """
    model.covar_module.kernels[0].lengthscale = model1.covar_module.base_kernel.lengthscale
    model.covar_module.kernels[1].lengthscale = model2.covar_module.base_kernel.lengthscale
    return model


# ---------------------------------------------------------------------------
# Lossless Efficient UCB Hierarchical GP — main BIF parent model
# ---------------------------------------------------------------------------

class Lossless_Efficient_UCB_Hierarchical_GP(gpytorch.models.ExactGP):
    """
    Hierarchical GP for the GAN experiment (4D parent, 2D children).

    This is the "lossless" variant: during Hoptimize only the parent MLL is
    optimized (no subkernel loss propagation to children).
    """

    def __init__(self, train_x, train_y, test_x_hier, likelihood, hierarchical_kernel,
                 prior_map, kernel_op, sub_models, kappa, query_counter=None, device="cpu"):
        super().__init__(train_x, train_y, likelihood)
        self.sub_models = sub_models
        self.kernel_op = kernel_op
        self.mean_module = PriorMean(prior_map, test_x_hier, device=device)
        self.mean_module.requires_grad = False
        self.covar_module = hierarchical_kernel
        self.kappa = kappa
        self.query_counter = query_counter

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def increment_q_n(self, query_c, query, test_x_hier):
        """
        Increment the query counter for the given 4D query point.

        Searches the flat test_x_hier (2401, 4) for the query and increments
        the corresponding counter.

        Parameters:
            query_c: (2401,) tensor of query counts
            query: (4,) tensor — the queried point
            test_x_hier: (2401, 4) tensor — all test points

        Returns:
            query_c: updated query counter
        """
        for i in range(len(test_x_hier)):
            if torch.allclose(test_x_hier[i], query, atol=1e-6):
                query_c[i] += 1
                break
        self.query_counter = query_c
        return query_c

    def Hoptimize(self, likelihood, training_iter, train_x, train_y, verbose=False):
        """
        Optimize the hierarchical GP model (lossless variant).

        Only optimizes the parent MLL — child subkernel losses are not
        propagated during this step.

        Parameters:
            likelihood: Likelihood function
            training_iter: Number of optimization iterations
            train_x: (N, 4) training inputs
            train_y: (N,) training outputs
            verbose: Whether to print optimization progress

        Returns:
            self: Optimized model
            likelihood: Optimized likelihood
        """
        self.train()
        likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        sub_optimizers = []
        sub_loss_functions = []
        for sub in self.sub_models:
            sub.train()
            sub_optimizers.append(torch.optim.Adam(sub.parameters(), lr=0.01))
            sub_loss_functions.append(gpytorch.mlls.ExactMarginalLogLikelihood(sub.likelihood, sub))

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)

        for i in range(training_iter):
            optimizer.zero_grad()
            for opt in sub_optimizers:
                opt.zero_grad()

            output = self(train_x)
            loss = -mll(output, train_y)
            loss.sum().backward(retain_graph=True)

            if verbose:
                print(
                    'Iter %d/%d - Loss: %.3f   ls1: [%.3f, %.3f]   ls2: [%.3f, %.3f]   noise: %.3f' % (
                        i + 1, training_iter, loss.item(),
                        self.covar_module.kernels[0].lengthscale[0][0].item(),
                        self.covar_module.kernels[0].lengthscale[0][1].item(),
                        self.covar_module.kernels[1].lengthscale[0][0].item(),
                        self.covar_module.kernels[1].lengthscale[0][1].item(),
                        self.likelihood.noise.item()
                    ))
            optimizer.step()

        return self, likelihood


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def make_prediction(model, test_x, likelihood):
    """
    Make a GP prediction (child or standalone model).

    Parameters:
        model: GP model
        test_x: (N, D) test inputs
        likelihood: Likelihood function

    Returns:
        observed_pred: MultivariateNormal predictive distribution
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return likelihood(model(test_x))


def make_Hierarchique_prediction(model, test_x, likelihood):
    """
    Make a hierarchical GP prediction (parent model).

    Parameters:
        model: Hierarchical GP model
        test_x: (N, 4) test inputs
        likelihood: Likelihood function

    Returns:
        observed_pred: MultivariateNormal predictive distribution
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return likelihood(model(test_x))


# ---------------------------------------------------------------------------
# Value look-up helpers
# ---------------------------------------------------------------------------

def get_y_mu_point_value_2d(query_2d, values, domain):
    """
    Look up a value in a 2D domain by finding the matching point.

    Parameters:
        query_2d: (2,) tensor — the query coordinates
        values: (N,) tensor — the function values
        domain: (N, 2) tensor — all domain points

    Returns:
        scalar tensor — the value at the matched point
    """
    for i in range(len(domain)):
        if torch.allclose(domain[i], query_2d, atol=1e-6):
            return values[i]
    raise ValueError(f"Query {query_2d} not found in domain")


# ---------------------------------------------------------------------------
# Prior map construction
# ---------------------------------------------------------------------------

def create_prior_map(G, model1, model2, test_x):
    """
    Create the (49, 49) prior mean map from child predictions.

    For each combination of child1 config i and child2 config j,
    the prior map is computed as:
        '+': y_mu1[i] + y_mu2[j]
        '*': y_mu1[i] * y_mu2[j]

    Parameters:
        G: tuple/list where G[0] is '+' or '*'
        model1: child 1 GP model (generator)
        model2: child 2 GP model (discriminator)
        test_x: (49, 2) child test domain

    Returns:
        prior_map: (49, 49) tensor
    """
    prior_map = torch.zeros(49, 49).double()

    with gpytorch.settings.lazily_evaluate_kernels(state=False):
        observed_pred1 = make_prediction(model1, test_x, model1.likelihood)
        observed_pred2 = make_prediction(model2, test_x, model2.likelihood)

    y_mu1 = observed_pred1.mean
    y_mu2 = observed_pred2.mean

    for i in range(49):
        for j in range(49):
            if G[0] == '+':
                prior_map[i][j] = y_mu1[i] + y_mu2[j]
            elif G[0] == '*':
                prior_map[i][j] = y_mu1[i] * y_mu2[j]

    return prior_map


# ---------------------------------------------------------------------------
# Child model update (2D)
# ---------------------------------------------------------------------------

def update_model_2d_max_seen(model, likelihood, train_x, train_y, next_query_pin,
                              response, env, training_iter=10):
    """
    Update a 2D child model with a new data point.

    Adapted from hmodel_synthetic.py update_model1_1D_max_seen for 2D children.
    Handles adaptive rescaling of env vs bif observations.

    Parameters:
        model: child ExactGPModel (2D)
        likelihood: child likelihood
        train_x: (N, 2) current training inputs
        train_y: (N,) current training outputs
        next_query_pin: (2,) tensor — the child coordinate to add
        response: scalar tensor — the contribution value
        env: bool — True if this is an environment query, False if BIF
        training_iter: number of optimization iterations

    Returns:
        model: updated child model
        likelihood: updated likelihood
        train_x: (N+1, 2) updated training inputs
        train_y: (N+1,) updated training outputs
    """
    from gan_models import optimize

    # Update training data
    train_x, train_y = model.update_training_data(train_x, train_y, next_query_pin, response, env)

    # Adaptive rescaling (env vs bif indices)
    div_y = train_y.clone()
    if torch.max(div_y[model.env_ind]) - torch.min(div_y[model.env_ind]) == 0:
        div_y[model.env_ind] = div_y[model.env_ind] / torch.max(div_y[model.env_ind])
    else:
        div_y[model.env_ind] = (div_y[model.env_ind] - torch.min(div_y[model.env_ind])) / \
                                (torch.max(div_y[model.env_ind]) - torch.min(div_y[model.env_ind]))
    if len(model.bif_ind) > 1:
        div_y[model.bif_ind] = (div_y[model.bif_ind] - torch.min(div_y[model.bif_ind])) / \
                                (torch.max(div_y[model.bif_ind]) - torch.min(div_y[model.bif_ind]))

    model.set_train_data(train_x, div_y, strict=False)
    model.train()
    likelihood.train()

    model, likelihood = optimize(model, likelihood, training_iter, train_x, div_y, verbose=False)

    return model, likelihood, train_x, train_y
