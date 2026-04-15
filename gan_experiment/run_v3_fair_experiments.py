"""
v3 fair comparison experiments: v3_fair_main + v3_fair_modularity.

Changes from v2:
  - k=20, nu=1.5, gamma=4, ls_prior=1 (LogNormalPrior centered at 1)
  - BIF starts without child queries (random parent queries decomposed to children)
  - Laferriere pays for child queries (counted in budget), performance reflects
    model's best at each query (not zero during init)
  - Laferriere queries a randomly fixed value for the non-queried child
    (not the marginal)
  - Modularity: BIF carries gen child, Laferriere carries gen child,
    Vanilla starts from scratch
  - 30 seeds
  - 3-panel plots: RO (model best), Parent R², Child R²
  - Main RO y-axis: 80%–100%. Modularity: auto.
  - Methods: BIF, Laferriere, Vanilla (no Random, no cold-start)
"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import warnings
import torch
import gpytorch
import matplotlib.pyplot as plt
from scipy.stats import linregress

warnings.filterwarnings('ignore')

import gan_models as models
import gan_hmodel as hmodel
from gan_surrogate import load_gan_surrogate, lookup_surrogate
from bif_gan import heatmap_r_score, compute_child_r2, run_repetition as _bif_run
from deep_gp_gan import run_repetition as _dgp_run_repetition

SEEDS = np.array([9049607, 2402697, 6510749, 758529, 3523986, 3224638, 9729091,
   5830471, 5343420, 2417321, 9891788, 9314146, 9488226, 2697408,
   5135059, 6813578, 430826, 6192331, 8026546, 6735254, 1112898,
   5609958, 4736968, 617977, 8500888, 4205117, 756214, 4283694,
   7449696, 9848369])

NQ = 20; NR = 30; NOISE = 0.1
KAPPA = 20; GAMMA = 4; NU = 1.5; LS_PRIOR = 1.0; INIT = 3; TITER = 15

base_dir = os.path.dirname(__file__)
C_BIF = '#1f77b4'; C_LAF = '#ff7f0e'; C_VAN = '#d62728'; C_DGP = '#9467bd'
C_ADD = '#8c564b'   # brown — 6th matplotlib default colour

path_a = os.path.join(base_dir, 'gan_surrogate.pt')
path_b = os.path.join(base_dir, 'gan_surrogate_specnorm.pt')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def patch_gp_lengthscale_prior(model, ls_prior_mean):
    """Add LogNormal prior on lengthscale and set initial value."""
    prior = gpytorch.priors.LogNormalPrior(np.log(ls_prior_mean), 0.5)
    model.covar_module.base_kernel.register_prior(
        'lengthscale_prior', prior, 'lengthscale')
    model.covar_module.base_kernel.lengthscale = ls_prior_mean
    return model


def compute_ro(true_val, gt_min, gt_max):
    return ((true_val - gt_min) / (gt_max - gt_min)).item()


def true_val_at_idx(idx, y_hier):
    return y_hier[idx // 49, idx % 49]


def find_flat_idx(query, test_x_hier):
    for j in range(len(test_x_hier)):
        if torch.allclose(test_x_hier[j], query, atol=1e-6):
            return j
    raise ValueError(f"Query {query} not found")


def stats(arr):
    a = np.array(arr)
    return a.mean(0), a.std(0) / np.sqrt(len(a))


# ---------------------------------------------------------------------------
# Vanilla BO (flat 4D GP, k=20, nu=1.5, ls_prior=1)
# Returns: (ro_model, heatmap_data)
# ---------------------------------------------------------------------------

def run_vanilla(surrogate_path, nq, n_init, titer, kappa, nu, seed, noise):
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, _, test_x_hier = \
        load_gan_surrogate(surrogate_path)
    y_flat = y_hier.flatten()
    n_parent = len(test_x_hier)
    gt_max = torch.max(y_flat); gt_min = torch.min(y_flat)
    hier_qc = torch.ones(n_parent).double()

    np.random.seed(seed)
    ro_model_traj = []
    heatmap_data = []

    for q in range(nq):
        if q < n_init:
            idx = np.random.randint(0, n_parent)
            query = test_x_hier[idx]
            true_val = y_flat[idx].item()
            noisy_val = true_val + np.random.normal(0, noise * (gt_max - gt_min).item())

            if q == 0:
                train_x = query.unsqueeze(0)
                train_y = torch.tensor([noisy_val]).double()
            else:
                train_x = torch.cat([train_x, query.unsqueeze(0)])
                train_y = torch.cat([train_y, torch.tensor([noisy_val]).double()])

            # Model-best: best observed point's true value
            best_obs_idx = train_y.argmax().item()
            flat_idx = find_flat_idx(train_x[best_obs_idx], test_x_hier)
            ro_model_traj.append(compute_ro(true_val_at_idx(flat_idx, y_hier), gt_min, gt_max))
            heatmap_data.append(np.zeros(n_parent))
            continue

        if q == n_init:
            norm_y = (train_y - train_y.mean())
            if train_y.std() > 0: norm_y = norm_y / train_y.std()
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gp = models.ExactGPModel(train_x, norm_y, likelihood, nu=nu)
            gp = patch_gp_lengthscale_prior(gp, LS_PRIOR)
            for i in range(len(train_x)):
                for j in range(n_parent):
                    if torch.allclose(test_x_hier[j], train_x[i], atol=1e-6):
                        hier_qc[j] += 1; break
            gp.eval(); likelihood.eval()
            with gpytorch.settings.lazily_evaluate_kernels(state=False):
                pred = models.make_prediction(gp, test_x_hier, likelihood)

        acq, y_mu = models.get_acquisition_map(kappa, pred, hier_qc)
        query = models.get_next_query_pins(acq, test_x_hier)
        val_r, _ = lookup_surrogate(query, test_x_hier, y_hier, noise=noise)
        response = torch.tensor(val_r).double()

        train_x = torch.cat([train_x, query.unsqueeze(0)])
        train_y = torch.cat([train_y, response.unsqueeze(0)])

        for j in range(n_parent):
            if torch.allclose(test_x_hier[j], query, atol=1e-6):
                hier_qc[j] += 1; break

        norm_y = (train_y - train_y.mean())
        if train_y.std() > 0: norm_y = norm_y / train_y.std()
        gp.set_train_data(train_x, norm_y, strict=False)

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            gp.train(); likelihood.train()
            gp, likelihood = models.optimize(gp, likelihood, titer, train_x, norm_y, verbose=False)
            gp.eval(); likelihood.eval()
            pred = models.make_prediction(gp, test_x_hier, likelihood)

        # Model-best RO
        pred_best_idx = torch.argmax(y_mu).item()
        ro_model_traj.append(compute_ro(true_val_at_idx(pred_best_idx, y_hier), gt_min, gt_max))
        heatmap_data.append(pred.mean.detach().cpu().numpy())

    return ro_model_traj, heatmap_data


# ---------------------------------------------------------------------------
# BIF — delegates to bif_gan.run_repetition() which supports ucb/ei/pi.
# Returns: (ro_model, heatmap_data, c1_r2, c2_r2, sub1, sub2)
# ---------------------------------------------------------------------------

def run_bif(surrogate_path, nq, n_init, titer, kappa, gamma, nu, seed, noise,
            pretrained_gen=None, acq_func='ucb'):
    children = [pretrained_gen] if pretrained_gen is not None else []
    master, sub1, sub2, ro, _, heatmap, c1_r2, c2_r2, _ = _bif_run(
        kappa=kappa, gamma=gamma, nu=nu,
        nbr_query=nq, nbr_rand_init=n_init,
        training_iter=titer, acq_func=acq_func,
        seed=seed, noise=noise,
        disable_tqdm=True, children=children,
        final=True, surrogate_path=surrogate_path,
    )
    return list(ro), heatmap, list(c1_r2), list(c2_r2), sub1, sub2


# ---------------------------------------------------------------------------
# Laferriere (corrected: fixed random value for non-queried child,
#             child queries cost parent budget, RO reflects model's best)
# Returns: (ro_model, heatmap_data, c1_r2, c2_r2)
# ---------------------------------------------------------------------------

def run_laferriere(surrogate_path, nq, n_child_init, titer, kappa, gamma, nu,
                   seed, noise, pretrained_gen=None):
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, _, test_x_hier = \
        load_gan_surrogate(surrogate_path)
    n_child = len(x_sub1); n_parent = len(test_x_hier)
    gt_max = torch.max(y_hier); gt_min = torch.min(y_hier)
    sub1_qc = torch.ones(n_child).double()
    sub2_qc = torch.ones(n_child).double()
    hier_qc = torch.ones(n_parent).double()

    np.random.seed(seed)

    has_pre = pretrained_gen is not None

    # Pick random FIXED config for the non-queried child
    fixed_child2_idx = np.random.randint(0, n_child)  # fixed disc config for gen queries
    fixed_child1_idx = np.random.randint(0, n_child)  # fixed gen config for disc queries

    ro_model_traj = []
    c1_r2_traj, c2_r2_traj = [], []
    heatmap_data = []

    # --- Child init phase ---
    if has_pre:
        sub1 = pretrained_gen
        sub1_like = sub1.likelihood
        sub1_qc = sub1.query_counter if sub1.query_counter is not None else sub1_qc
        n_gen_init = 0
    else:
        n_gen_init = n_child_init

    n_disc_init = n_child_init
    n_paid = n_gen_init + n_disc_init

    # Initialize gen child (if not pretrained)
    if not has_pre:
        gen_indices = np.random.randint(0, n_child, size=n_gen_init)
        gen_x = x_sub1[gen_indices].clone()
        gen_y = torch.zeros(n_gen_init).double()
        for i, gi in enumerate(gen_indices):
            true_4d = y_hier[gi, fixed_child2_idx].item()
            y_range = (torch.max(y_hier) - torch.min(y_hier)).item()
            gen_y[i] = true_4d + np.random.normal(0, noise * y_range)

        sub1_like = gpytorch.likelihoods.GaussianLikelihood()
        sub1 = models.ExactGPModel(gen_x, gen_y, sub1_like, query_counter=sub1_qc, nu=nu)
        sub1 = patch_gp_lengthscale_prior(sub1, LS_PRIOR)
        for i in range(n_gen_init):
            sub1_qc = sub1.increment_q_n(sub1_qc, gen_x[i], x_sub1)

    # Initialize disc child
    disc_indices = np.random.randint(0, n_child, size=n_disc_init)
    disc_x = x_sub2[disc_indices].clone()
    disc_y = torch.zeros(n_disc_init).double()
    for i, di in enumerate(disc_indices):
        true_4d = y_hier[fixed_child1_idx, di].item()
        y_range = (torch.max(y_hier) - torch.min(y_hier)).item()
        disc_y[i] = true_4d + np.random.normal(0, noise * y_range)

    sub2_like = gpytorch.likelihoods.GaussianLikelihood()
    sub2 = models.ExactGPModel(disc_x, disc_y, sub2_like, query_counter=sub2_qc, nu=nu)
    sub2 = patch_gp_lengthscale_prior(sub2, LS_PRIOR)
    for i in range(n_disc_init):
        sub2_qc = sub2.increment_q_n(sub2_qc, disc_x[i], x_sub2)

    # Get frozen child predictions
    sub1.eval(); sub1_like.eval()
    sub2.eval(); sub2_like.eval()
    with gpytorch.settings.lazily_evaluate_kernels(state=False):
        pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
        pred2 = models.make_prediction(sub2, x_sub2, sub2_like)
    y_mu1, y_conf1 = pred1.mean, pred1.stddev
    y_mu2, y_conf2 = pred2.mean, pred2.stddev

    # Build static prior map
    p1 = y_mu1 + gamma * y_conf1 / torch.sqrt(sub1_qc)
    p2 = y_mu2 + gamma * y_conf2 / torch.sqrt(sub2_qc)
    prior_map = torch.zeros(n_child, n_child).double()
    for ii in range(n_child):
        for jj in range(n_child):
            prior_map[ii, jj] = (p1[ii] + p2[jj]) / 2
    pm_max = torch.max(prior_map)

    # Record RO for each init query (model's best from prior map)
    prior_best_flat = torch.argmax(prior_map.flatten()).item()
    prior_best_i, prior_best_j = prior_best_flat // n_child, prior_best_flat % n_child
    prior_best_true = y_hier[prior_best_i, prior_best_j]
    for qi in range(n_paid):
        ro_model_traj.append(compute_ro(prior_best_true, gt_min, gt_max))
        c1_r2_traj.append(0.0); c2_r2_traj.append(0.0)
        heatmap_data.append(np.zeros(n_parent))

    remaining = nq - n_paid
    if remaining <= 0:
        return ro_model_traj, heatmap_data, c1_r2_traj, c2_r2_traj

    # First parent query from prior map
    first_q = models.get_next_query_pins(torch.flatten(prior_map), test_x_hier)
    val_r, _ = lookup_surrogate(first_q, test_x_hier, y_hier, noise=noise)
    resp = torch.tensor(val_r).double()
    train_x_hier = first_q.unsqueeze(0)
    train_y_hier = resp.unsqueeze(0)

    h_kernel = hmodel.hierarchical_kernel("add_kernel", sub1, sub2)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    norm_y = train_y_hier - train_y_hier.mean()
    master = hmodel.Lossless_Efficient_UCB_Hierarchical_GP(
        train_x_hier, norm_y, test_x_hier, likelihood, h_kernel,
        prior_map / pm_max, 'add_kernel', [sub1, sub2], kappa, hier_qc)
    hier_qc = master.increment_q_n(hier_qc, first_q, test_x_hier)
    master.eval(); likelihood.eval()
    with gpytorch.settings.lazily_evaluate_kernels(state=False):
        observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

    for q_r in range(remaining):
        cr = compute_child_r2([sub1, sub2], [x_sub1, x_sub2], [y_sub1, y_sub2])
        c1_r2_traj.append(cr[0]); c2_r2_traj.append(cr[1])

        acq, y_mu = models.get_acquisition_map(kappa, observed_pred, hier_qc)
        next_q = models.get_next_query_pins(acq, test_x_hier)
        val_r, _ = lookup_surrogate(next_q, test_x_hier, y_hier, noise=noise)
        response = torch.tensor(val_r).double()

        hier_qc = master.increment_q_n(hier_qc, next_q, test_x_hier)
        master = hmodel.update_kernel_parameters(master, sub1, sub2)

        train_x_hier = torch.cat([train_x_hier, next_q.unsqueeze(0)])
        train_y_hier = torch.cat([train_y_hier, response.unsqueeze(0)])
        norm_y = train_y_hier - train_y_hier.mean()
        if train_y_hier.std() > 0: norm_y = norm_y / train_y_hier.std()
        master.set_train_data(train_x_hier, norm_y, strict=False)

        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            master.train(); likelihood.train()
            master, likelihood = master.Hoptimize(likelihood, titer, train_x_hier, norm_y, verbose=False)
            master.eval(); likelihood.eval()
            observed_pred = hmodel.make_Hierarchique_prediction(master, test_x_hier, likelihood)

        pred_best = torch.argmax(y_mu).item()
        ro_model_traj.append(compute_ro(true_val_at_idx(pred_best, y_hier), gt_min, gt_max))
        heatmap_data.append(observed_pred.mean.detach().cpu().numpy())

    return ro_model_traj, heatmap_data, c1_r2_traj, c2_r2_traj


# ---------------------------------------------------------------------------
# ADD GP-UCB (two independent child GPs, additive UCB, no parent GP)
# Returns: (ro_model_traj, heatmap_data, c1_r2, c2_r2, sub1, sub2)
# ---------------------------------------------------------------------------

def run_add_gp_ucb(surrogate_path, nq, n_init, titer, kappa, nu, seed, noise,
                   pretrained_gen=None):
    x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, _, test_x_hier = \
        load_gan_surrogate(surrogate_path)
    n_child = len(x_sub1); n_parent = len(test_x_hier)
    gt_max = torch.max(y_hier); gt_min = torch.min(y_hier)

    sub1_qc = torch.ones(n_child).double()
    sub2_qc = torch.ones(n_child).double()

    np.random.seed(seed)
    ro_model_traj = []
    c1_r2_traj, c2_r2_traj = [], []
    heatmap_data = []

    has_pre = pretrained_gen is not None
    continue_from = None

    for q in range(nq):
        if q == 0:
            if has_pre:
                sub1 = pretrained_gen
                sub1_like = sub1.likelihood
                sub1_qc = sub1.query_counter if sub1.query_counter is not None else sub1_qc
                train_x_sub1 = sub1.train_inputs[0]
                train_y_sub1 = sub1.raw_train_y.clone()
            else:
                sub1 = None

            # First random parent query
            idx = np.random.randint(0, n_parent)
            query = test_x_hier[idx]
            val_r, _ = lookup_surrogate(query, test_x_hier, y_hier, noise=noise)
            response = torch.tensor(val_r).double()
            train_x_hier = query.unsqueeze(0)
            train_y_hier = response.unsqueeze(0)

            c1c, c2c = query[0:2], query[2:4]
            cont1, cont2 = response * 0.5, response * 0.5

            true_val = true_val_at_idx(idx, y_hier)

            if has_pre:
                train_x_sub2 = c2c.unsqueeze(0)
                train_y_sub2 = cont2.unsqueeze(0)
                sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like,
                                           query_counter=sub2_qc, nu=nu)
                sub2 = patch_gp_lengthscale_prior(sub2, LS_PRIOR)
                sub2_qc = sub2.increment_q_n(sub2_qc, c2c, x_sub2)
                sub1, sub1_like, train_x_sub1, train_y_sub1 = hmodel.update_model_2d_max_seen(
                    sub1, sub1_like, train_x_sub1, train_y_sub1, c1c, cont1, False, titer)
                sub1_qc = sub1.increment_q_n(sub1_qc, c1c, x_sub1)
            else:
                train_x_sub1 = c1c.unsqueeze(0)
                train_y_sub1 = cont1.unsqueeze(0)
                sub1_like = gpytorch.likelihoods.GaussianLikelihood()
                sub1 = models.ExactGPModel(train_x_sub1, train_y_sub1, sub1_like,
                                           query_counter=sub1_qc, nu=nu)
                sub1 = patch_gp_lengthscale_prior(sub1, LS_PRIOR)
                sub1_qc = sub1.increment_q_n(sub1_qc, c1c, x_sub1)

                train_x_sub2 = c2c.unsqueeze(0)
                train_y_sub2 = cont2.unsqueeze(0)
                sub2_like = gpytorch.likelihoods.GaussianLikelihood()
                sub2 = models.ExactGPModel(train_x_sub2, train_y_sub2, sub2_like,
                                           query_counter=sub2_qc, nu=nu)
                sub2 = patch_gp_lengthscale_prior(sub2, LS_PRIOR)
                sub2_qc = sub2.increment_q_n(sub2_qc, c2c, x_sub2)

            ro_model_traj.append(compute_ro(true_val, gt_min, gt_max))
            c1_r2_traj.append(0.0); c2_r2_traj.append(0.0)
            heatmap_data.append(np.zeros(n_parent))

            actual_init = 1 if has_pre else n_init
            for qi in range(1, actual_init):
                idx2 = np.random.randint(0, n_parent)
                query2 = test_x_hier[idx2]
                val_r2, _ = lookup_surrogate(query2, test_x_hier, y_hier, noise=noise)
                resp2 = torch.tensor(val_r2).double()
                c1c2, c2c2 = query2[0:2], query2[2:4]

                sub1, sub1_like, train_x_sub1, train_y_sub1 = \
                    hmodel.update_model_2d_max_seen(sub1, sub1_like, train_x_sub1, train_y_sub1,
                                                    c1c2, resp2*0.5, False, titer)
                sub1_qc = sub1.increment_q_n(sub1_qc, c1c2, x_sub1)
                sub2, sub2_like, train_x_sub2, train_y_sub2 = \
                    hmodel.update_model_2d_max_seen(sub2, sub2_like, train_x_sub2, train_y_sub2,
                                                    c2c2, resp2*0.5, False, titer)
                sub2_qc = sub2.increment_q_n(sub2_qc, c2c2, x_sub2)

                train_x_hier = torch.cat([train_x_hier, query2.unsqueeze(0)])
                train_y_hier = torch.cat([train_y_hier, resp2.unsqueeze(0)])

                best_obs = train_y_hier.argmax().item()
                fi = find_flat_idx(train_x_hier[best_obs], test_x_hier)
                ro_model_traj.append(compute_ro(true_val_at_idx(fi, y_hier), gt_min, gt_max))
                c1_r2_traj.append(0.0); c2_r2_traj.append(0.0)
                heatmap_data.append(np.zeros(n_parent))

            continue_from = actual_init
            continue

        if q < continue_from:
            continue

        # Predict on full child domains
        sub1.eval(); sub1_like.eval()
        sub2.eval(); sub2_like.eval()
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
            pred2 = models.make_prediction(sub2, x_sub2, sub2_like)
        y_mu1, y_conf1 = pred1.mean, pred1.stddev
        y_mu2, y_conf2 = pred2.mean, pred2.stddev

        # Additive UCB acquisition (no gamma, no parent GP)
        # Outer sum: result[i,j] = ucb1[i] + ucb2[j] (row=child1/gen, col=child2/disc)
        ucb1 = y_mu1 + kappa * torch.nan_to_num(y_conf1 / torch.sqrt(sub1_qc))
        ucb2 = y_mu2 + kappa * torch.nan_to_num(y_conf2 / torch.sqrt(sub2_qc))
        acq_2d_flat = (ucb1.unsqueeze(1) + ucb2.unsqueeze(0)).reshape(-1)

        tied = torch.where(acq_2d_flat == acq_2d_flat.max())[0]
        flat_idx = tied[np.random.randint(len(tied))].item()
        i_star = flat_idx // n_child
        j_star = flat_idx % n_child
        next_q = test_x_hier[flat_idx].clone()

        val_r, _ = lookup_surrogate(next_q, test_x_hier, y_hier, noise=noise)
        response = torch.tensor(val_r).double()

        sub1_qc = sub1.increment_q_n(sub1_qc, x_sub1[i_star], x_sub1)
        sub2_qc = sub2.increment_q_n(sub2_qc, x_sub2[j_star], x_sub2)

        # Both children receive the full response (env=True) — no soft decomposition.
        # This is the key difference from BIF, which splits the response between children.
        sub1, sub1_like, train_x_sub1, train_y_sub1 = hmodel.update_model_2d_max_seen(
            sub1, sub1_like, train_x_sub1, train_y_sub1, x_sub1[i_star], response, True, titer)
        sub2, sub2_like, train_x_sub2, train_y_sub2 = hmodel.update_model_2d_max_seen(
            sub2, sub2_like, train_x_sub2, train_y_sub2, x_sub2[j_star], response, True, titer)

        # Re-predict for metrics
        sub1.eval(); sub1_like.eval()
        sub2.eval(); sub2_like.eval()
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            pred1 = models.make_prediction(sub1, x_sub1, sub1_like)
            pred2 = models.make_prediction(sub2, x_sub2, sub2_like)
        y_mu1, y_conf1 = pred1.mean, pred1.stddev
        y_mu2, y_conf2 = pred2.mean, pred2.stddev

        mean_2d_flat = (y_mu1.unsqueeze(1) + y_mu2.unsqueeze(0)).reshape(-1)

        # Child R2
        with gpytorch.settings.lazily_evaluate_kernels(state=False):
            cr = compute_child_r2([sub1, sub2], [x_sub1, x_sub2], [y_sub1, y_sub2])
        c1_r2_traj.append(cr[0]); c2_r2_traj.append(cr[1])

        # Model-best RO from additive mean surface
        pred_best_idx = torch.argmax(mean_2d_flat).item()
        i_best = pred_best_idx // n_child
        j_best = pred_best_idx % n_child
        ro_model_traj.append(compute_ro(y_hier[i_best, j_best], gt_min, gt_max))
        heatmap_data.append(mean_2d_flat.detach().cpu().numpy())

    return ro_model_traj, heatmap_data, c1_r2_traj, c2_r2_traj, sub1, sub2


# ---------------------------------------------------------------------------
# Deep GP (variational, no hierarchy)
# Returns: (ro_model_traj, heatmap_data)
# ---------------------------------------------------------------------------

def run_deep_gp(surrogate_path, nq, n_init, titer, kappa, seed, noise,
                depth=3, num_inducing=32, hidden_dims=4):
    """
    Variational Deep GP BO on the GAN surrogate.
    Returns: (ro_model_traj, heatmap_data) — same shape as run_vanilla.
    """
    ro_traj, _, heatmap, _ = _dgp_run_repetition(
        kappa=kappa, nbr_query=nq, nbr_rand_init=n_init, training_iter=titer,
        depth=depth, num_inducing=num_inducing, hidden_dims=hidden_dims,
        seed=seed, noise=noise, disable_tqdm=True,
        surrogate_path=surrogate_path,
    )
    return ro_traj, heatmap


# ---------------------------------------------------------------------------
# CSV save/load + plotting from CSV
# ---------------------------------------------------------------------------

import pandas as pd

def save_results_csv(filepath, seeds, methods_data, y_hier_gt):
    """Save per-seed, per-query results for all methods to a single CSV."""
    rows = []
    for method_name, data in methods_data.items():
        for si, seed in enumerate(seeds):
            ro = data['ro'][si]
            # Parent R2 from heatmap
            hm = data['hm'][si]
            gt_flat = y_hier_gt.numpy().reshape(-1)
            for q in range(len(ro)):
                pr2 = 0.0
                if np.any(hm[q] != 0):
                    lr = linregress(gt_flat, hm[q])
                    pr2 = lr.rvalue ** 2 if not np.isnan(lr.rvalue) else 0.0
                # Child R2 (avg of c1 and c2)
                cr2 = np.nan
                if 'c1' in data and si < len(data['c1']):
                    cr2 = (data['c1'][si][q] + data['c2'][si][q]) / 2
                rows.append({
                    'method': method_name,
                    'seed': int(seed),
                    'query': q,
                    'ro': ro[q],
                    'parent_r2': pr2,
                    'child_r2': cr2,
                })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f'  Saved {filepath}')
    return df


def load_results_csv(filepath):
    return pd.read_csv(filepath)


# Paper colors
C_VAN_P   = '#1f77b4'   # blue
C_LAF_P   = '#ff7f0e'   # orange
C_BIF_UCB = '#2ca02c'   # green  (BIF with UCB)
C_BIF_EI  = '#17becf'   # cyan   (BIF with EI)
C_BIF_PI  = '#e377c2'   # pink   (BIF with PI)

XTICKS = [0, 5, 10, 15, 20]


def plot_from_df(df, experiment_name, ro_ylim, methods_style):
    """
    Generate figures from a DataFrame.
    methods_style: list of (method_name, label, color, linestyle)
    Produces:
      - v3_{experiment_name}_ro.png/svg  (single RO plot)
      - v3_{experiment_name}_r2.png/svg  (2-panel: parent R2 + child R2)
    """
    queries = np.sort(df['query'].unique())
    n_seeds = df.groupby('method')['seed'].nunique().max()

    # --- RO figure (single panel) ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    for method, label, color, ls in methods_style:
        sub = df[df['method'] == method]
        ro_arr = sub.pivot(index='seed', columns='query', values='ro').values
        m, s = ro_arr.mean(0), ro_arr.std(0) / np.sqrt(len(ro_arr))
        ax.plot(queries, m, label=label, color=color, ls=ls, lw=2)
        ax.fill_between(queries, m-s, m+s, alpha=0.15, color=color)
    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel('Relative Optimum', fontsize=12)
    ax.set_ylim(ro_ylim)
    ax.set_xticks(XTICKS)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'figures/v3_{experiment_name}_ro.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/v3_{experiment_name}_ro.svg', bbox_inches='tight')
    plt.close()

    # --- R2 figure (2-panel: parent + child) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for method, label, color, ls in methods_style:
        sub = df[df['method'] == method]
        r2_arr = sub.pivot(index='seed', columns='query', values='parent_r2').values
        m, s = r2_arr.mean(0), r2_arr.std(0) / np.sqrt(len(r2_arr))
        ax.plot(queries, m, label=label, color=color, ls=ls, lw=2)
        ax.fill_between(queries, m-s, m+s, alpha=0.15, color=color)
    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel(r'Parent $R^2$', fontsize=12)
    ax.set_title(r'(a) Parent $R^2$', fontsize=13)
    ax.set_ylim(-0.05, 1.05); ax.set_xticks(XTICKS); ax.legend(fontsize=9)

    ax = axes[1]
    for method, label, color, ls in methods_style:
        sub = df[df['method'] == method]
        cr2_arr = sub.pivot(index='seed', columns='query', values='child_r2').values
        if np.all(np.isnan(cr2_arr)):
            continue  # Vanilla has no children
        m = np.nanmean(cr2_arr, 0)
        s = np.nanstd(cr2_arr, 0) / np.sqrt(np.sum(~np.isnan(cr2_arr[:, 0])))
        ax.plot(queries, m, label=label, color=color, ls=ls, lw=2)
        ax.fill_between(queries, m-s, m+s, alpha=0.15, color=color)
    ax.set_xlabel('Query Number', fontsize=12)
    ax.set_ylabel(r'Child Avg $R^2$', fontsize=12)
    ax.set_title(r'(b) Child Avg $R^2$', fontsize=13)
    ax.set_ylim(-0.05, 1.05); ax.set_xticks(XTICKS); ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f'figures/v3_{experiment_name}_r2.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'figures/v3_{experiment_name}_r2.svg', bbox_inches='tight')
    plt.close()

    print(f'  Saved v3_{experiment_name}_ro + v3_{experiment_name}_r2')

    # --- Results table ---
    def sem(arr):
        return np.std(arr) / np.sqrt(len(arr))

    print(f'\n  === {experiment_name.upper()} Results ===')
    print(f'  {"Method":20s} {"RO (final)":>14s} {"Parent R2":>14s} {"Child R2":>14s} {"AUC (sum)":>14s}')
    print(f'  {"-"*20} {"-"*14} {"-"*14} {"-"*14} {"-"*14}')
    for method, label, _, _ in methods_style:
        sub = df[df['method'] == method]
        ro_final = sub[sub['query'] == queries[-1]]['ro'].values
        r2_final = sub[sub['query'] == queries[-1]]['parent_r2'].values
        cr2_final = sub[sub['query'] == queries[-1]]['child_r2'].values
        # AUC = sum(ro + parent_r2 + child_r2) across all queries, per seed
        # Matches efficient_general_2d.py: auc = np.sum(y + avg_child + r2_avg)
        # NaN child_r2 (methods without children) treated as 0
        combined_auc = sub.groupby('seed').apply(
            lambda g: np.nansum(g['ro'].values + g['parent_r2'].values +
                                np.nan_to_num(g['child_r2'].values, nan=0.0))
        ).values
        cr2_str = f'{np.nanmean(cr2_final)*100:.2f}+/-{sem(cr2_final[~np.isnan(cr2_final)])*100:.2f}%' \
                  if not np.all(np.isnan(cr2_final)) else '    N/A       '
        print(f'  {label:20s} {ro_final.mean()*100:.2f}+/-{sem(ro_final)*100:.2f}%'
              f' {r2_final.mean()*100:.2f}+/-{sem(r2_final)*100:.2f}%'
              f' {cr2_str}'
              f' {combined_auc.mean()*100:.2f}+/-{sem(combined_auc)*100:.2f}%')


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip experiments, just re-plot from CSV')
    args = parser.parse_args()

    os.makedirs('figures', exist_ok=True)
    csv_main = os.path.join(base_dir, 'figures', 'v3_results_main.csv')
    csv_mod = os.path.join(base_dir, 'figures', 'v3_results_modularity.csv')

    if not args.plot_only:
        # ==================================================================
        print('=== v3 MAIN EXPERIMENT (v2 Standard, k=20, nu=1.5, gamma=4) ===')
        # ==================================================================
        _, _, _, _, _, y_hier_a, _, _ = load_gan_surrogate(path_a)

        all_bif_ucb = {'ro':[], 'hm':[], 'c1':[], 'c2':[]}
        all_bif_ei  = {'ro':[], 'hm':[], 'c1':[], 'c2':[]}
        all_bif_pi  = {'ro':[], 'hm':[], 'c1':[], 'c2':[]}
        all_laf = {'ro':[], 'hm':[], 'c1':[], 'c2':[]}
        all_van = {'ro':[], 'hm':[]}
        all_dgp = {'ro':[], 'hm':[]}
        all_add = {'ro':[], 'hm':[], 'c1':[], 'c2':[]}
        pretrained_gens_ucb = {}
        pretrained_gens_ei  = {}
        pretrained_gens_pi  = {}
        add_pretrained_gens = {}

        for i, seed in enumerate(SEEDS):
            print(f'  Seed {i+1}/{NR}', end='', flush=True)
            ro, hm = run_vanilla(path_a, NQ, INIT, TITER, KAPPA, NU, seed, NOISE)
            all_van['ro'].append(ro); all_van['hm'].append(hm)

            for _acq, _store, _pg in [('ucb', all_bif_ucb, pretrained_gens_ucb),
                                       ('ei',  all_bif_ei,  pretrained_gens_ei),
                                       ('pi',  all_bif_pi,  pretrained_gens_pi)]:
                ro, hm, c1, c2, sub1_out, _ = run_bif(
                    path_a, NQ, INIT, TITER, KAPPA, GAMMA, NU, seed, NOISE, acq_func=_acq)
                _store['ro'].append(ro); _store['hm'].append(hm)
                _store['c1'].append(c1); _store['c2'].append(c2)
                _pg[seed] = sub1_out

            ro, hm, c1, c2 = run_laferriere(path_a, NQ, INIT, TITER, KAPPA, GAMMA, NU, seed, NOISE)
            all_laf['ro'].append(ro); all_laf['hm'].append(hm)
            all_laf['c1'].append(c1); all_laf['c2'].append(c2)

            ro, hm = run_deep_gp(path_a, NQ, INIT, TITER, KAPPA, seed, NOISE)
            all_dgp['ro'].append(ro); all_dgp['hm'].append(hm)

            ro, hm, c1, c2, sub1_add, _ = run_add_gp_ucb(path_a, NQ, INIT, TITER, KAPPA, NU, seed, NOISE)
            all_add['ro'].append(ro); all_add['hm'].append(hm)
            all_add['c1'].append(c1); all_add['c2'].append(c2)
            add_pretrained_gens[seed] = sub1_add
            print(' done')

        save_results_csv(csv_main, SEEDS,
                         {'BIF-UCB': all_bif_ucb, 'BIF-EI': all_bif_ei, 'BIF-PI': all_bif_pi,
                          'Laferriere': all_laf, 'Vanilla': all_van,
                          'Deep GP': all_dgp, 'ADD GP-UCB': all_add},
                         y_hier_a)

        # ==================================================================
        print('\n=== v3 MODULARITY (v2 SpectralNorm) ===')
        # ==================================================================
        _, _, _, _, _, y_hier_b, _, _ = load_gan_surrogate(path_b)

        mod_bif_ucb = {'ro':[], 'hm':[], 'c1':[], 'c2':[]}
        mod_bif_ei  = {'ro':[], 'hm':[], 'c1':[], 'c2':[]}
        mod_bif_pi  = {'ro':[], 'hm':[], 'c1':[], 'c2':[]}
        mod_laf = {'ro':[], 'hm':[], 'c1':[], 'c2':[]}
        mod_van = {'ro':[], 'hm':[]}
        mod_dgp = {'ro':[], 'hm':[]}
        mod_add = {'ro':[], 'hm':[], 'c1':[], 'c2':[]}

        for i, seed in enumerate(SEEDS):
            print(f'  Seed {i+1}/{NR}', end='', flush=True)
            ro, hm = run_vanilla(path_b, NQ, INIT, TITER, KAPPA, NU, seed, NOISE)
            mod_van['ro'].append(ro); mod_van['hm'].append(hm)

            for _acq, _store, _pg in [('ucb', mod_bif_ucb, pretrained_gens_ucb),
                                       ('ei',  mod_bif_ei,  pretrained_gens_ei),
                                       ('pi',  mod_bif_pi,  pretrained_gens_pi)]:
                ro, hm, c1, c2, _, _ = run_bif(
                    path_b, NQ, 1, TITER, KAPPA, GAMMA, NU, seed, NOISE,
                    pretrained_gen=copy.deepcopy(_pg[seed]), acq_func=_acq)
                _store['ro'].append(ro); _store['hm'].append(hm)
                _store['c1'].append(c1); _store['c2'].append(c2)

            ro, hm, c1, c2 = run_laferriere(
                path_b, NQ, INIT, TITER, KAPPA, GAMMA, NU, seed, NOISE,
                pretrained_gen=copy.deepcopy(pretrained_gens_ucb[seed]))
            mod_laf['ro'].append(ro); mod_laf['hm'].append(hm)
            mod_laf['c1'].append(c1); mod_laf['c2'].append(c2)

            ro, hm = run_deep_gp(path_b, NQ, INIT, TITER, KAPPA, seed, NOISE)
            mod_dgp['ro'].append(ro); mod_dgp['hm'].append(hm)

            sub1_add_pre = add_pretrained_gens[seed]
            ro, hm, c1, c2, _, _ = run_add_gp_ucb(
                path_b, NQ, 1, TITER, KAPPA, NU, seed, NOISE,
                pretrained_gen=copy.deepcopy(sub1_add_pre))
            mod_add['ro'].append(ro); mod_add['hm'].append(hm)
            mod_add['c1'].append(c1); mod_add['c2'].append(c2)
            print(' done')

        save_results_csv(csv_mod, SEEDS,
                         {'BIF-UCB (hot)': mod_bif_ucb, 'BIF-EI (hot)': mod_bif_ei,
                          'BIF-PI (hot)': mod_bif_pi,
                          'Laferriere (hot)': mod_laf, 'Vanilla': mod_van,
                          'Deep GP': mod_dgp, 'ADD GP-UCB (hot)': mod_add},
                         y_hier_b)

    # ==================================================================
    # Plot from CSV (works with --plot-only too)
    # ==================================================================
    print('\n=== Plotting from CSV ===')
    df_main = load_results_csv(csv_main)
    df_mod = load_results_csv(csv_mod)

    main_style = [
        ('Vanilla',    'Vanilla',    C_VAN_P,   ':'),
        ('Laferriere', 'Laferriere', C_LAF_P,   '--'),
        ('BIF-UCB',    'BIF (UCB)',  C_BIF_UCB, '-'),
        ('BIF-EI',     'BIF (EI)',   C_BIF_EI,  '-'),
        ('BIF-PI',     'BIF (PI)',   C_BIF_PI,  '-'),
        ('Deep GP',    'Deep GP',    C_DGP,     '-.'),
        ('ADD GP-UCB', 'ADD GP-UCB', C_ADD,     (0,(3,1,1,1))),
    ]
    mod_style = [
        ('Vanilla',          'Vanilla',              C_VAN_P,   ':'),
        ('Laferriere (hot)', 'Laferriere (hot)',      C_LAF_P,   '--'),
        ('BIF-UCB (hot)',    'BIF (UCB, hot-start)',  C_BIF_UCB, '-'),
        ('BIF-EI (hot)',     'BIF (EI, hot-start)',   C_BIF_EI,  '-'),
        ('BIF-PI (hot)',     'BIF (PI, hot-start)',   C_BIF_PI,  '-'),
        ('Deep GP',          'Deep GP',              C_DGP,     '-.'),
        ('ADD GP-UCB (hot)', 'ADD GP-UCB (hot)',      C_ADD,     (0,(3,1,1,1))),
    ]

    plot_from_df(df_main, 'fair_main', (0.80, 1.02), main_style)
    plot_from_df(df_mod,  'fair_modularity', (0.70, 1.02), mod_style)

    print('\nDone!')
