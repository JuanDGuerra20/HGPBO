"""
GP model definitions and utility functions for the GAN experiment.
Adapted from synthetic_2d/synthetic_models.py for 2D children.
"""
import torch
import numpy as np
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Standard GP model. Works for any input dimension.
    Tracks env_ind (environment/init data) and bif_ind (BIF-contributed data)
    for adaptive rescaling.
    """
    def __init__(self, train_x, train_y, likelihood, nu=2.5, query_counter=None):
        # Min-max normalize train_y to [0, 1]
        if len(train_y) == 1:
            y_max = torch.max(torch.abs(train_y))
            scaled_y = train_y / y_max if y_max > 0 else train_y
        else:
            y_range = torch.max(train_y) - torch.min(train_y)
            if y_range > 0:
                scaled_y = (train_y - torch.min(train_y)) / y_range
            else:
                scaled_y = torch.zeros_like(train_y)

        super().__init__(train_x, scaled_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu))
        self.query_counter = query_counter
        self.env_max_seen = torch.max(train_y)
        self.env_ind = list(range(0, len(train_x)))
        self.bif_max_seen = torch.tensor(-9999999, dtype=torch.float64)
        self.bif_ind = []
        # Store raw (unscaled) training y for correct transfer across tasks
        self.raw_train_y = train_y.clone()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def increment_q_n(self, query_c, query, domain):
        """
        Increment query counter for a 2D child domain.
        query: (2,) tensor
        domain: (N, 2) tensor
        query_c: (N,) tensor
        """
        for x in range(len(domain)):
            if torch.allclose(domain[x], query, atol=1e-6):
                query_c[x] += 1
                break
        self.query_counter = query_c
        return query_c

    def update_max_seen_response_no_norm(self, next_query_value_random, max_seen_resp, env=False):
        if env:
            if next_query_value_random > self.env_max_seen:
                self.env_max_seen = next_query_value_random
        else:
            if next_query_value_random > self.bif_max_seen:
                self.bif_max_seen = next_query_value_random
        return next_query_value_random

    def update_training_data(self, train_x, train_y, next_query_pins, next_query_value, env=False):
        """
        Append a new data point to training data.
        Handles both 1D and 2D inputs.
        next_query_pins: (D,) tensor where D is input dim
        next_query_value: scalar tensor
        """
        if env:
            self.env_ind.append(len(train_x))
        else:
            self.bif_ind.append(len(train_x))

        # Ensure next_query_pins has right shape for concatenation
        if next_query_pins.dim() == 1:
            new_x = next_query_pins.unsqueeze(0)  # (1, D)
        else:
            new_x = next_query_pins

        train_x = torch.cat([train_x, new_x.to(train_x.dtype)], dim=0)

        new_y = torch.tensor([next_query_value], dtype=train_y.dtype)
        train_y = torch.cat([train_y, new_y], dim=0)

        # Keep raw_train_y in sync
        self.raw_train_y = torch.cat([self.raw_train_y, new_y], dim=0)

        return train_x, train_y


def optimize(model, likelihood, training_iter, train_x, train_y, verbose=False):
    """Optimize GP hyperparameters via MLL."""
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if verbose:
            print(f'Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}')
        optimizer.step()

    return model, likelihood


def make_prediction(model, test_x, likelihood):
    """Make GP prediction at test points."""
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return likelihood(model(test_x))


def compute_acq_value(mu, sigma, query_count, kappa, acq_func, best_f):
    """
    Compute acquisition value for UCB, EI, or PI.

    Parameters:
        mu: predictive mean
        sigma: predictive standard deviation
        query_count: per-point query counter tensor (used by UCB)
        kappa: exploration weight (UCB trade-off; EI/PI jitter xi)
        acq_func: 'ucb', 'ei', or 'pi'
        best_f: current best observed value (required for 'ei' and 'pi')
    """
    if acq_func == 'ucb':
        return mu + kappa * torch.nan_to_num(sigma / torch.sqrt(query_count))
    normal = torch.distributions.Normal(0, 1)
    safe_sigma = torch.clamp(sigma, min=1e-9)
    Z = (mu - best_f - kappa) / safe_sigma
    if acq_func == 'ei':
        ei = safe_sigma * (Z * normal.cdf(Z) + torch.exp(normal.log_prob(Z)))
        return torch.clamp(ei, min=0.0)
    if acq_func == 'pi':
        return normal.cdf(Z)
    raise ValueError(f"Unknown acq_func '{acq_func}'. Choose 'ucb', 'ei', or 'pi'.")


def get_acquisition_map(kappa, observed_pred, query_counter, acq_func='ucb', best_f=None):
    """
    Compute acquisition map.
    Returns (acquisition_values, posterior_mean).
    """
    y_mu = observed_pred.mean
    y_conf = observed_pred.stddev
    acquisition_map = compute_acq_value(y_mu, y_conf, query_counter, kappa, acq_func, best_f)
    return acquisition_map, y_mu


def get_next_query_pins(acquisition_map, test_x):
    """Select the test point with highest acquisition value."""
    best_idx = torch.argmax(acquisition_map)
    return test_x[best_idx]


def get_instantaneous_regret(y_mu, ground_truth_max, test_x_hier, y_hier):
    """
    Compute instantaneous regret (Relative Optimum).
    y_mu: posterior mean predictions
    ground_truth_max: best value in ground truth
    """
    predicted_best_idx = torch.argmax(y_mu)
    predicted_best_point = test_x_hier[predicted_best_idx]

    # Find the true value at the predicted best point
    # test_x_hier is (2401, 4), y_hier is (49, 49)
    i = predicted_best_idx.item() // 49
    j = predicted_best_idx.item() % 49

    if y_hier.dim() == 2:
        true_value = y_hier[i, j]
    else:
        true_value = y_hier.flatten()[predicted_best_idx]

    ground_truth_min = torch.min(y_hier)

    # Relative Optimum = (y_hat + y_bar) / (y_star + y_bar)
    # where y_hat = true value at predicted best, y_star = true best, y_bar = true worst (negated)
    relative_optimum = (true_value + torch.abs(ground_truth_min)) / \
                       (ground_truth_max + torch.abs(ground_truth_min))

    return relative_optimum.item(), predicted_best_point


def get_exploitation_score(next_query_value_mean, ground_truth_max):
    """Exploitation score = query_value / max_possible."""
    if ground_truth_max == 0:
        return torch.tensor(0.0)
    return next_query_value_mean / ground_truth_max


def update_max_seen_response_no_norm(next_query_value_random, max_seen_resp):
    """Update and return max seen response."""
    if next_query_value_random > max_seen_resp:
        max_seen_resp = next_query_value_random
    return next_query_value_random, max_seen_resp


def update_training_data(train_x, train_y, next_query_pins, response):
    """
    Append a new data point to hierarchical training data.
    next_query_pins: (4,) tensor
    response: scalar tensor
    """
    new_x = next_query_pins.unsqueeze(0)  # (1, 4)
    train_x = torch.cat([train_x, new_x.to(train_x.dtype)], dim=0)

    new_y = torch.tensor([response], dtype=train_y.dtype)
    train_y = torch.cat([train_y, new_y], dim=0)

    return train_x, train_y
