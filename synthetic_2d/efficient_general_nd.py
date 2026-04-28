"""
N-child BIF experiment runner.

Generalises efficient_general_2d.py to an arbitrary number of children.
Set n_children in the __main__ block (or pass it through training_procedure_nd).

Run from synthetic_2d/:
    python efficient_general_nd.py
"""

import math
import os
import warnings
from datetime import datetime
from functools import partial

import gpytorch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import synthetic_models as models
import hmodel_synthetic as hmodel
import visualization_information as vi
from dataset_actions import select_random_queries, update_training_data


# ---------------------------------------------------------------------------
# N-D Prior Mean
# ---------------------------------------------------------------------------

class PriorMeanND(gpytorch.means.Mean):
    """
    Lookup-based prior mean for an N-dimensional parent GP.

    Unlike PriorMean (2D only), this stores a flat index map and supports
    any number of children.
    """

    def __init__(self, prior_map, test_x_grid, n_children, device="cpu"):
        """
        Parameters
        ----------
        prior_map   : torch.Tensor of shape [dim]*n_children
        test_x_grid : torch.Tensor of shape [dim]*n_children + [n_children]
                      (the N-D grid of parent input points)
        n_children  : int
        device      : str
        """
        super().__init__()
        self.register_parameter(
            'map', torch.nn.Parameter(prior_map.to(device), requires_grad=False)
        )
        self.n_children = n_children
        self.device = device
        # Build flat lookup: str(coord_tensor) -> flat index
        flat_grid = test_x_grid.reshape(-1, n_children)
        self.lookup = {}
        for i, pt in enumerate(flat_grid):
            self.lookup[str(pt)] = i

    def forward(self, input):
        map_flat = self.map.reshape(-1)
        new_prior = torch.zeros(
            input.shape[0], device=self.device, dtype=input.dtype
        )
        for i in range(input.shape[0]):
            idx = self.lookup[str(input[i])]
            new_prior[i] = map_flat[idx]
        return new_prior


# ---------------------------------------------------------------------------
# N-D Hierarchical GP (wraps the existing Lossless_Efficient class)
# ---------------------------------------------------------------------------

class Lossless_Efficient_UCB_Hierarchical_GP_ND(hmodel.Lossless_Efficient_UCB_Hierarchical_GP):
    """
    Subclass of Lossless_Efficient_UCB_Hierarchical_GP with:
      - PriorMeanND instead of PriorMean (N-D lookup)
      - N-D increment_q_n
      - Hoptimize without hardcoded 2-kernel verbose print
    """

    def __init__(self, train_x, train_y, test_x_grid, likelihood, hier_kernel,
                 prior_map, kernel_op, sub_models, kappa, n_children,
                 query_counter=None, device="cpu"):
        # Bypass parent __init__ (which hard-wires PriorMean) and set up directly.
        gpytorch.models.ExactGP.__init__(self, train_x, train_y, likelihood)
        self.sub_models = sub_models
        self.kernel_op = kernel_op
        self.mean_module = PriorMeanND(
            prior_map, test_x_grid, n_children, device=device
        )
        self.mean_module.requires_grad = False
        self.covar_module = hier_kernel
        self.kappa = kappa
        self.query_counter = query_counter

    def increment_q_n(self, query_c, query, domain_grid):
        """Increment query counter for an N-D parent grid."""
        flat_domain = domain_grid.reshape(-1, domain_grid.shape[-1])
        matches = torch.all(flat_domain == query.unsqueeze(0), dim=1)
        flat_idx = matches.nonzero(as_tuple=True)[0][0].item()
        query_c[flat_idx] += 1
        self.query_counter = query_c
        return query_c

    def Hoptimize(self, likelihood, training_iter, train_x, train_y, verbose=False):
        self.train()
        likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        sub_optimizers = []
        for sub in self.sub_models:
            sub.train()
            sub_optimizers.append(torch.optim.Adam(sub.parameters(), lr=0.01))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)
        for _ in range(training_iter):
            optimizer.zero_grad()
            for opt in sub_optimizers:
                opt.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            output = self(train_x)
            loss = -mll(output, train_y)
            loss.sum().backward(retain_graph=True)
            optimizer.step()
        return self, likelihood


# ---------------------------------------------------------------------------
# N-D kernel helpers
# ---------------------------------------------------------------------------

def hierarchical_kernel_nd(child_models):
    """
    Build an additive ScaleKernel sum over N children.

    Each sub-kernel operates on a single active dimension (child index).
    """
    n = len(child_models)
    scaled_kernels = []
    for i, m in enumerate(child_models):
        k = gpytorch.kernels.MaternKernel(
            nu=2.5, has_lengthscale=True, ard_num_dims=1,
            input_dim=n, active_dims=[i]
        )
        k.outputscale = m.covar_module.outputscale
        k.lengthscale = m.covar_module.base_kernel.lengthscale
        scaled_kernels.append(gpytorch.kernels.ScaleKernel(k))
    result = scaled_kernels[0]
    for sk in scaled_kernels[1:]:
        result = result + sk
    return result


def update_kernel_parameters_nd(model, child_models):
    """Sync parent kernel parameters from the updated child models."""
    for i, child in enumerate(child_models):
        model.covar_module.kernels[i].base_kernel.outputscale = \
            child.covar_module.outputscale
        model.covar_module.kernels[i].base_kernel.lengthscale = \
            child.covar_module.base_kernel.lengthscale
    return model


# ---------------------------------------------------------------------------
# N-D dataset generators
# ---------------------------------------------------------------------------

def _make_test_sub(n_test, x_sub):
    """Thin version of make_test_sub for child subspaces."""
    idx = torch.linspace(0, len(x_sub) - 1, n_test).long()
    return x_sub[idx]


def generate_michalewicz_nd(n_children, dimension, eps):
    """
    N-dimensional Michalewicz benchmark.

    f(x) = sum_k sin(x_k) * (sin((k+1)*x_k^2/pi))^20
    y_hier[idx] = sum_k y_sub_k[idx_k]  +  eps * prod_k y_sub_k[idx_k]
    """
    x_subs = [torch.linspace(0, math.pi, dimension).double() for _ in range(n_children)]
    y_subs = [
        torch.sin(x_subs[k]) * (torch.sin((k + 1) * x_subs[k] ** 2 / math.pi) ** 20)
        for k in range(n_children)
    ]

    coord_grids = torch.meshgrid(*x_subs, indexing='ij')
    x_hier = torch.stack(coord_grids, dim=-1)
    y_grids = torch.meshgrid(*y_subs, indexing='ij')
    y_hier = sum(y_grids) + eps * torch.prod(torch.stack(list(y_grids)), dim=0)

    test_x = _make_test_sub(5, x_subs[0])
    test_x_hier = x_hier.reshape(-1, n_children)
    return x_subs, y_subs, x_hier, y_hier, test_x, test_x_hier


def generate_rastrigin_nd(n_children, dimension, eps):
    """
    N-dimensional Rastrigin benchmark.

    f_k(x) = x^2 - 10*cos(2*pi*x)
    y_hier[idx] = 10*n_children + sum_k y_sub_k[idx_k]
                + eps * prod_k x_sub_k[idx_k]
    """
    x_subs = [torch.linspace(-2.0, 2.0, dimension).double() for _ in range(n_children)]
    y_subs = [
        x_subs[k] ** 2 - 10 * torch.cos(2 * math.pi * x_subs[k])
        for k in range(n_children)
    ]

    coord_grids = torch.meshgrid(*x_subs, indexing='ij')
    x_hier = torch.stack(coord_grids, dim=-1)
    y_grids = torch.meshgrid(*y_subs, indexing='ij')
    y_hier = 10.0 * n_children + sum(y_grids) + eps * torch.prod(torch.stack(list(coord_grids)), dim=0)

    test_x = _make_test_sub(5, x_subs[0])
    test_x_hier = x_hier.reshape(-1, n_children)
    return x_subs, y_subs, x_hier, y_hier, test_x, test_x_hier


def get_dataset_info_nd(dataset_num, n_children, alpha=1):
    """
    Returns (data_name, data_creation_func, eps) for the N-child case.

    dataset_num == 1 : N-D Michalewicz
    dataset_num == 2 : N-D Rastrigin

    data_creation_func(dimension, eps) follows the ND signature, returning
    (x_subs, y_subs, x_hier, y_hier, test_x, test_x_hier).
    Uses functools.partial so the returned function is picklable (multiprocessing-safe).
    """
    if dataset_num == 1:
        data_name = f'michalewicz_{n_children}d'
        eps = 0.25
        data_creation_func = partial(generate_michalewicz_nd, n_children)
    elif dataset_num == 2:
        data_name = f'rastrigin_{n_children}d'
        eps = 0.1
        data_creation_func = partial(generate_rastrigin_nd, n_children)
    else:
        raise AssertionError(
            f"dataset_num {dataset_num} not supported for ND runner. "
            "Use 1 (Michalewicz) or 2 (Rastrigin)."
        )

    return data_name, data_creation_func, eps


# ---------------------------------------------------------------------------
# N-D query helpers (replace hardcoded 2D versions in synthetic_models)
# ---------------------------------------------------------------------------

def get_next_query_value_nd(next_query_pins, test_x_hier, y_hier, noise=0):
    """Look up the ground-truth value at next_query_pins from the N-D grid."""
    y_flat = y_hier.reshape(-1)
    matches = torch.all(test_x_hier == next_query_pins.unsqueeze(0), dim=1)
    flat_idx = matches.nonzero(as_tuple=True)[0][0].item()
    value = y_flat[flat_idx].item()
    y_range = (y_flat.max() - y_flat.min()).item()
    random_val = value + np.random.normal(0, noise) * y_range
    mean_val = value + np.random.normal(0, noise) * y_range
    return random_val, mean_val


def get_instantaneous_regret_nd(y_mu, ground_truth_max, test_x_hier, y_hier):
    """Compute instantaneous regret for the N-D parent model."""
    y_flat = y_hier.reshape(-1)
    mu_flat = y_mu.reshape(-1)
    # Pick best predicted point (random tie-break)
    best_val = torch.max(mu_flat)
    candidates = torch.where(mu_flat == best_val)[0]
    best_idx = candidates[np.random.randint(len(candidates))].item()
    next_query_pins = test_x_hier[best_idx]

    mean_value = y_flat[best_idx].item()
    y_min = y_flat.min().item()
    y_max = ground_truth_max.item()
    regret = (mean_value - y_min) / (y_max - y_min) if y_max != y_min else 0.0
    return regret, next_query_pins


# ---------------------------------------------------------------------------
# Main repetition loop
# ---------------------------------------------------------------------------

def run_repetition_nd(kappa, gamma, nu, nbr_query, nbr_rand_init, dimension,
                      training_iter, data_creation_func, eps,
                      model_name, folder_of_the_day, data_name, n_children,
                      final=False, seed=True, noise=0.1,
                      disable_tqdm=False, acq_func='ucb'):
    warnings.filterwarnings('ignore')

    x_subs, y_subs, x_hier, y_hier, test_x, test_x_hier = data_creation_func(dimension, eps)

    prior_map = torch.zeros([dimension] * n_children, dtype=torch.float64)

    ground_truth_max_hier = torch.max(y_hier)

    max_seen_resp_2D = 0
    max_seen_k = [0] * n_children  # per-child max-seen responses

    sub_qcs = [torch.ones(x.shape, dtype=torch.float64) for x in x_subs]
    hier_qc = torch.ones(dimension ** n_children, dtype=torch.float64)

    better_exploitation_score = []
    better_exploration_score = []
    child_r2s = [[] for _ in range(n_children)]
    heatmap_rep = []

    # ------------------------------------------------------------------
    # q == 0 : initialise children and parent
    # ------------------------------------------------------------------
    for q in tqdm(range(nbr_query), disable=disable_tqdm):
        if q == 0:
            sub_models = []
            sub_likes = []
            train_x_subs = []
            train_y_subs = []

            for k in range(n_children):
                tx, ty = select_random_queries(
                    nbr_rand_init, x_subs[k], y_subs[k], seed=seed, noise=noise
                )
                like = gpytorch.likelihoods.GaussianLikelihood()
                m = models.ExactGPModel(tx, ty, like, query_counter=sub_qcs[k], nu=nu)
                for pt in tx:
                    sub_qcs[k] = m.increment_q_n(sub_qcs[k], pt, x_subs[k])
                sub_models.append(m)
                sub_likes.append(like)
                train_x_subs.append(tx)
                train_y_subs.append(ty)
                max_seen_k[k] = torch.max(ty)

            # --- get predictions from all children ---
            y_mus, y_confs, acq_vals = [], [], []
            for k in range(n_children):
                sub_models[k].eval()
                sub_likes[k].eval()
                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    pred = models.make_prediction(sub_models[k], x_subs[k], sub_likes[k])
                y_mus.append(pred.mean)
                y_confs.append(pred.stddev)
                acq_vals.append(
                    pred.mean + gamma * pred.stddev / torch.sqrt(sub_qcs[k])
                )
                sub_models[k].train()
                sub_likes[k].train()

            # --- build N-D prior map from child acquisition values ---
            grids = torch.meshgrid(*acq_vals, indexing='ij')
            prior_map = sum(grids) / n_children

            prior_map_max = torch.max(prior_map)

            # --- pick first query from prior map ---
            next_query_pins = models.get_next_query_pins(
                torch.flatten(prior_map), test_x_hier
            )
            next_query_value_random, next_query_value_mean = get_next_query_value_nd(
                next_query_pins, test_x_hier, y_hier, noise=noise
            )
            response = torch.tensor(next_query_value_random, dtype=torch.float64)
            train_x_hier = torch.reshape(next_query_pins, (1, n_children))
            train_y_hier = torch.reshape(response, (1,))
            max_seen_resp_2D = torch.max(train_y_hier)

            # --- initialise parent GP ---
            hier_kernel = hierarchical_kernel_nd(sub_models)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            master = Lossless_Efficient_UCB_Hierarchical_GP_ND(
                train_x_hier,
                train_y_hier - torch.mean(train_y_hier),
                x_hier,
                likelihood,
                hier_kernel,
                prior_map / prior_map_max,
                kernel_op='add_kernel',
                sub_models=sub_models,
                kappa=kappa,
                n_children=n_children,
                query_counter=hier_qc,
            )
            for i in range(len(train_x_hier)):
                hier_qc = master.increment_q_n(hier_qc, train_x_hier[i], x_hier)

        # ------------------------------------------------------------------
        # Evaluate parent and score
        # ------------------------------------------------------------------
        master.eval()
        likelihood.eval()
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            child_r2_step = vi.child_contour_r2(master.sub_models, x_subs, y_subs)
        for k in range(n_children):
            child_r2s[k].append(child_r2_step[k])

        best_f_hier = torch.max(master.train_targets)
        acquisition_map, hierar_y_mu = models.get_acquisition_map(
            kappa, observed_pred, hier_qc, acq_func=acq_func, best_f=best_f_hier
        )

        next_query_pins = models.get_next_query_pins(acquisition_map, test_x_hier)
        next_query_value_random, next_query_value_mean = get_next_query_value_nd(
            next_query_pins, test_x_hier, y_hier, noise=noise
        )

        # --- per-child point values for decomposition ---
        y_mus = []
        y_confs = []
        for k in range(n_children):
            sub_models[k].eval()
            sub_likes[k].eval()
            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                pred = models.make_prediction(sub_models[k], x_subs[k], sub_likes[k])
            y_mus.append(pred.mean)
            y_confs.append(pred.stddev)

        mu_pts, conf_pts, qc_pts = [], [], []
        for k in range(n_children):
            mu_pts.append(hmodel.get_y_mu_point_value(next_query_pins[k], y_mus[k], x_subs[k]))
            conf_pts.append(hmodel.get_y_mu_point_value(next_query_pins[k], y_confs[k], x_subs[k]))
            qc_pts.append(hmodel.get_y_mu_point_value(next_query_pins[k], sub_qcs[k], x_subs[k]))

        next_query_value_random, max_seen_resp_2D = models.update_max_seen_response_no_norm(
            next_query_value_random, max_seen_resp_2D
        )
        response = torch.tensor(next_query_value_random, dtype=torch.float64)

        # --- softmax decomposition over N children ---
        conts_scaled = []
        for k in range(n_children):
            norm_k = torch.max(
                models.compute_acq_value(y_mus[k], y_confs[k], sub_qcs[k], gamma, acq_func, max_seen_k[k])
            )
            cont_k = models.compute_acq_value(
                mu_pts[k], conf_pts[k], qc_pts[k], gamma, acq_func, max_seen_k[k]
            )
            conts_scaled.append(torch.nan_to_num(cont_k / norm_k))

        exp_conts = [torch.exp(c) for c in conts_scaled]
        div = sum(exp_conts)
        contributions = [torch.nan_to_num(response * e / div) for e in exp_conts]

        # side-effect: update model's internal bif_max_seen
        for k in range(n_children):
            sub_models[k].update_max_seen_response_no_norm(contributions[k], max_seen_k[k])

        # --- increment query counters ---
        for k in range(n_children):
            sub_qcs[k] = sub_models[k].increment_q_n(sub_qcs[k], next_query_pins[k], x_subs[k])
        hier_qc = master.increment_q_n(hier_qc, next_query_pins, x_hier)

        # --- find N-D grid index for next_query_pins ---
        matches = torch.all(test_x_hier == next_query_pins.unsqueeze(0), dim=1)
        flat_idx = matches.nonzero(as_tuple=True)[0][0].item()
        next_query_indices = list(np.unravel_index(flat_idx, [dimension] * n_children))

        # --- update child models ---
        for k in range(n_children):
            sub_models[k], sub_likes[k], train_x_subs[k], train_y_subs[k] = \
                hmodel.update_model1_1D_max_seen(
                    sub_models[k], sub_likes[k],
                    train_x_subs[k], train_y_subs[k],
                    x_subs[k][next_query_indices[k]],
                    contributions[k],
                    False,
                    training_iter=training_iter,
                )

        # --- re-predict from updated children ---
        new_acq_vals = []
        for k in range(n_children):
            sub_models[k].eval()
            sub_likes[k].eval()
            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                pred = models.make_prediction(sub_models[k], x_subs[k], sub_likes[k])
            y_mus[k] = pred.mean
            y_confs[k] = pred.stddev
            new_acq_vals.append(
                pred.mean + gamma * pred.stddev / torch.sqrt(sub_qcs[k])
            )

        # --- update prior map ---
        grids = torch.meshgrid(*new_acq_vals, indexing='ij')
        prior_map = sum(grids) / n_children
        prior_map_max = torch.max(prior_map)
        master.mean_module.map = torch.nn.Parameter(prior_map / prior_map_max)
        master = update_kernel_parameters_nd(master, sub_models)

        # --- update parent training data and optimise ---
        train_x_hier, train_y_hier = update_training_data(
            train_x_hier, train_y_hier, next_query_pins, response
        )
        master.set_train_data(
            train_x_hier,
            (train_y_hier - torch.mean(train_y_hier)) / torch.std(train_y_hier),
            strict=False,
        )
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            master.train()
            likelihood.train()
            master, likelihood = master.Hoptimize(
                likelihood, training_iter, train_x_hier,
                (train_y_hier - torch.mean(train_y_hier)) / torch.std(train_y_hier),
                verbose=False,
            )
            master.eval()
            likelihood.eval()
            for k in range(n_children):
                sub_models[k].eval()
                sub_likes[k].eval()
            observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

        # --- scores ---
        instantaneous_regret, _ = get_instantaneous_regret_nd(
            observed_pred.mean, ground_truth_max_hier, test_x_hier, y_hier
        )
        exploitation_score = models.get_exploitation_score(next_query_value_mean, ground_truth_max_hier)

        better_exploration_score.append(instantaneous_regret)
        better_exploitation_score.append(exploitation_score)
        heatmap_rep.append(observed_pred.mean.detach().cpu().numpy())

    if final:
        return (master, sub_models, better_exploration_score,
                better_exploitation_score, heatmap_rep, child_r2s)
    else:
        return better_exploration_score, better_exploitation_score, heatmap_rep, child_r2s


# ---------------------------------------------------------------------------
# Module-level worker (must be at module level for multiprocessing on Windows)
# ---------------------------------------------------------------------------

def _run_rep_worker_nd(args):
    (kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter,
     data_creation_func, eps, model_name, folder_of_the_day, data_name,
     n_children, seed, noise, disable_tqdm, acq_func) = args
    return run_repetition_nd(
        kappa, gamma, nu, nbr_query, nbr_rand_init, dimension, training_iter,
        data_creation_func, eps, model_name, folder_of_the_day, data_name,
        n_children, final=False, seed=seed, noise=noise,
        disable_tqdm=disable_tqdm, acq_func=acq_func,
    )


# ---------------------------------------------------------------------------
# Training procedure (orchestrates repetitions)
# ---------------------------------------------------------------------------

def training_procedure_nd(nbr_query, nbr_repetition, nbr_rand_init, dimension,
                          training_iter, k_vals, g_vals, nu_vals,
                          data_name, data_creation_func, eps,
                          n_children, multi, seed, noise=0.1,
                          disable_tqdm=False, acq_func='ucb'):
    model_name = "Lossless_Efficient"

    current_dateday = datetime.now().strftime("%Y-%m-%d")
    workspace = f"{data_name}/{model_name.lower()}"
    folder_of_the_day = '/data-' + str(current_dateday)
    if not os.path.exists(f"{data_name}"):
        os.mkdir(f"{data_name}")
        os.mkdir(workspace)
    if os.path.exists(workspace + folder_of_the_day):
        print('Data folder is ready')
    else:
        os.mkdir(workspace + folder_of_the_day)
        os.mkdir(workspace + folder_of_the_day + '/contour')
        os.mkdir(workspace + folder_of_the_day + '/differentiable_plots')
        os.mkdir(workspace + folder_of_the_day + '/csv')
        os.mkdir(workspace + folder_of_the_day + '/hp_analysis')
        os.mkdir(workspace + folder_of_the_day + '/models')
        os.mkdir(workspace + folder_of_the_day + '/png')
        print('Data folders created')

    list_models = []

    # How many "free" initialization steps to prepend to metric arrays
    init_steps = n_children * nbr_rand_init

    for kappa in k_vals:
        for gamma in g_vals:
            for nu in nu_vals:
                better_exploration_score = []
                better_exploitation_score = []
                heatmap_data = []
                all_child_r2_data = []   # list[rep] of list[child][query]

                if multi:
                    worker_args = [
                        (kappa, gamma, nu, nbr_query, nbr_rand_init, dimension,
                         training_iter, data_creation_func, eps, model_name,
                         folder_of_the_day, data_name, n_children,
                         seed[i], noise, disable_tqdm, acq_func)
                        for i in range(nbr_repetition - 1)
                    ]
                    with mp.Pool(processes=nbr_repetition - 1) as pool:
                        processes = [pool.apply_async(_run_rep_worker_nd, (a,))
                                     for a in worker_args]

                        # Run the final repetition synchronously to retrieve models
                        try:
                            master, sub_models, rep_explor, rep_exploit, hm, cr2 = \
                                run_repetition_nd(
                                    kappa, gamma, nu, nbr_query, nbr_rand_init,
                                    dimension, training_iter, data_creation_func,
                                    eps, model_name, folder_of_the_day, data_name,
                                    n_children, final=True, seed=seed[-1],
                                    noise=noise, disable_tqdm=disable_tqdm,
                                    acq_func=acq_func,
                                )
                        except Exception:
                            try:
                                master, sub_models, rep_explor, rep_exploit, hm, cr2 = \
                                    run_repetition_nd(
                                        kappa, gamma, nu, nbr_query, nbr_rand_init,
                                        dimension, training_iter, data_creation_func,
                                        eps, model_name, folder_of_the_day, data_name,
                                        n_children, final=True, seed=seed[-1],
                                        noise=noise, disable_tqdm=disable_tqdm,
                                        acq_func=acq_func,
                                    )
                            except Exception:
                                master = sub_models = None
                                rep_explor = rep_exploit = hm = cr2 = None

                        if rep_explor is not None:
                            better_exploration_score.append(rep_explor)
                            better_exploitation_score.append(rep_exploit)
                            heatmap_data.append(hm)
                            all_child_r2_data.append(cr2)

                        for proc in processes:
                            try:
                                rep_explor, rep_exploit, hm, cr2 = proc.get()
                                better_exploration_score.append(rep_explor)
                                better_exploitation_score.append(rep_exploit)
                                heatmap_data.append(hm)
                                all_child_r2_data.append(cr2)
                            except Exception:
                                continue
                else:
                    for i in range(nbr_repetition):
                        master, sub_models, rep_explor, rep_exploit, hm, cr2 = \
                            run_repetition_nd(
                                kappa, gamma, nu, nbr_query, nbr_rand_init,
                                dimension, training_iter, data_creation_func,
                                eps, model_name, folder_of_the_day, data_name,
                                n_children, final=True, seed=seed[i],
                                noise=noise, disable_tqdm=disable_tqdm,
                                acq_func=acq_func,
                            )
                        better_exploration_score.append(rep_explor)
                        better_exploitation_score.append(rep_exploit)
                        heatmap_data.append(hm)
                        all_child_r2_data.append(cr2)

                # ----------------------------------------------------------
                # Aggregate and save
                # ----------------------------------------------------------
                k_str = str(kappa).replace('.', ',')
                g_str = str(gamma).replace('.', ',')
                n_str = str(nu).replace('.', '_')
                e_str = str(eps).replace('.', '')

                heatmap_data = np.array(heatmap_data)

                # Save parent model state
                if master is not None:
                    torch.save(
                        master.state_dict(),
                        f'{data_name}/{model_name.lower()}{folder_of_the_day}/models/'
                        f'{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}'
                        f'_eps_{e_str}_k_{k_str}_g_{g_str}_nu_{n_str}_noise_{str(noise).replace(".", ",")}.pth',
                    )

                """# Instantaneous regret curve
                n_reps = len(better_exploration_score)
                y_raw = np.mean(better_exploration_score, axis=0)
                y = np.concatenate([np.zeros(init_steps), y_raw])[:nbr_query]
                std_raw = np.std(better_exploration_score, axis=0) / np.sqrt(n_reps)
                std = np.concatenate([np.zeros(init_steps), std_raw])[:nbr_query]

                plt.plot(y, label='Instantaneous Regret')
                plt.fill_between(range(len(y)), y - std, y + std, alpha=0.4)

                # Parent R2
                _, _, _, y_hier_gt, _, _ = data_creation_func(dimension, eps)
                r2 = np.array(vi.heatmap_r_score(heatmap_data, y_hier_gt))
                # r2 shape: (n_reps, n_queries_actual); prepend init_steps zeros
                r2_padded = np.concatenate(
                    [np.zeros((n_reps, init_steps)), r2], axis=1
                )[:, :nbr_query]
                r2_avg = np.mean(r2_padded, axis=0)
                r2_std = np.std(r2_padded, axis=0) / np.sqrt(n_reps)
                plt.plot(r2_avg, label='Parent R2')
                plt.fill_between(range(len(r2_avg)), r2_avg - r2_std, r2_avg + r2_std, alpha=0.4)

                # Per-child R2 (averaged across children and repetitions)
                # all_child_r2_data[rep][child][query]
                per_child_rep = []
                for k in range(n_children):
                    child_k = np.array([all_child_r2_data[rep][k]
                                        for rep in range(len(all_child_r2_data))])
                    per_child_rep.append(child_k)
                # all_children shape: (n_children * n_reps, n_queries_actual)
                all_children_raw = np.concatenate(per_child_rep, axis=0)
                all_children = np.concatenate(
                    [np.zeros((len(all_children_raw), init_steps)), all_children_raw], axis=1
                )[:, :nbr_query]
                avg_child = np.mean(all_children, axis=0)
                std_child = np.std(all_children, axis=0) / np.sqrt(len(all_children))

                plt.plot(avg_child, label='Child Avg R2')
                plt.fill_between(range(len(avg_child)), avg_child - std_child,
                                 avg_child + std_child, alpha=0.4)

                plt.legend()
                plt.ylim(-0.1, 1.1)
                plt.ylabel('Performance')
                plt.xlabel('Query Number')
                plt.title(
                    f'{model_name} {n_children}-Child HGPBO {nbr_repetition} reps '
                    f'kappa {k_str} Gamma {g_str} Nu {n_str}'
                )
                svg_base = (
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/'
                    f'{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}'
                    f'_eps_{e_str}_k_{k_str}_g_{g_str}_nu_{n_str}'
                )
                plt.savefig(svg_base + '_differentiable_plots.svg')
                plt.savefig(svg_base + '_png.png')
                plt.close()

                # CSV summary
                auc = np.sum(y + avg_child + r2_avg)
                print(f"explor {y[-1] * 100:.2f} ± {std[-1] * 100:.2f}")
                print(f"r2 avg {r2_avg[-1] * 100:.2f} ± {r2_std[-1] * 100:.2f}")
                print(f"child r2 {avg_child[-1] * 100:.2f} ± {std_child[-1] * 100:.2f}")
                print(f"AUC {np.sum(auc) * 100:.2f}")
                print(f'\n{data_name} {model_name} {n_children}D kappa {k_str} gamma {g_str} nu {n_str} complete!\n')

                df = pd.DataFrame(
                    [f'kappa_{k_str}_gamma_{g_str}_nu_{n_str}_{n_children}children',
                     y[-1], r2_avg[-1], avg_child[-1], auc]
                )
                df.index = ['name', 'instantaneous_regret', 'parent_r2', 'avg_child_r2', 'auc']
                df.to_csv(
                    f'{data_name}/{model_name.lower()}{folder_of_the_day}/csv/'
                    f'{nbr_repetition}_rep_init_{nbr_rand_init}_train_iter_{training_iter}'
                    f'_eps_{e_str}_k_{k_str}_g_{g_str}_nu_{n_str}.csv'
                )

                list_models.append(
                    [f'kappa_{k_str}_gamma_{g_str}_nu_{n_str}',
                     master, y, r2_avg, avg_child]
                )"""

    return list_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # ---- configuration ----
    n_children = 3       # change to any N >= 2
    dataset = 1          # 1 = Michalewicz-ND, 2 = Rastrigin-ND

    dimension = 20
    nbr_query = 100
    training_iter = 10
    nbr_repetition = 1
    k_vals = [7.5]
    g_vals = [2.0]
    nu_vals = [0.5]
    multi = False        # set True to use multiprocessing
    acq = 'ucb'
    nbr_rand_init = 2
    seed = [False] * nbr_repetition
    noise = 0.1

    data_name, data_creation_func, eps = get_dataset_info_nd(dataset, n_children)

    training_procedure_nd(
        nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter,
        k_vals, g_vals, nu_vals,
        data_name, data_creation_func, eps,
        n_children, multi, seed, noise=noise,
        disable_tqdm=False, acq_func=acq,
    )
