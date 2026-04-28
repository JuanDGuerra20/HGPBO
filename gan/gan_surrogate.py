"""
GAN surrogate dataset module.

Loads the pre-computed tabular surrogate (gan_surrogate.pt) and provides data
in the 8-tuple format expected by the BIF codebase:
    (x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier)

GP inputs are normalized to linspace(0, 1, 7) per dimension.
x_sub1/x_sub2 are (49, 2) tensors (7x7 grid per child).
x_hier is (49, 49, 4) and test_x_hier is (2401, 4).
"""

import os
import torch
import numpy as np


def _build_normalized_grids():
    """Build normalized [0, 1] coordinate grids for GP inputs."""
    coords_1d = torch.linspace(0.0, 1.0, 7).double()

    # Child 1 (Generator): 2D grid of (lr_gen_norm, z_dim_norm)
    # Shape: (49, 2)
    g1, g2 = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
    x_sub1 = torch.stack([g1.flatten(), g2.flatten()], dim=1).double()  # (49, 2)

    # Child 2 (Discriminator): 2D grid of (lr_disc_norm, dropout_norm)
    x_sub2 = x_sub1.clone()  # Same 7x7 grid, different interpretation

    # Parent: 4D grid of (lr_gen, z_dim, lr_disc, dropout)
    # Shape: (2401, 4) — all combinations of child1 x child2
    x_hier_list = []
    x_hier_3d = torch.zeros(49, 49, 4).double()
    for i in range(49):  # child1 configs
        for j in range(49):  # child2 configs
            point = torch.cat([x_sub1[i], x_sub2[j]])
            x_hier_list.append(point)
            x_hier_3d[i, j] = point

    test_x_hier = torch.stack(x_hier_list).double()  # (2401, 4)

    return x_sub1, x_sub2, x_hier_3d, test_x_hier


def load_gan_surrogate(surrogate_path="gan_surrogate.pt"):
    """
    Load the GAN tabular surrogate and return data in the standard 8-tuple format.

    Returns:
        x_sub1:     (49, 2)  — generator config coordinates (normalized)
        y_sub1:     (49,)    — generator marginal ground truth (mean neg-FID)
        x_sub2:     (49, 2)  — discriminator config coordinates (normalized)
        y_sub2:     (49,)    — discriminator marginal ground truth (mean neg-FID)
        x_hier:     (49, 49, 4) — parent config coordinates (for index lookups)
        y_hier:     (49, 49) — neg-FID values (parent ground truth)
        test_x:     (49, 2)  — alias for x_sub1 (not used but required by convention)
        test_x_hier: (2401, 4) — flattened parent test grid
    """
    data = torch.load(surrogate_path, weights_only=False)

    x_sub1, x_sub2, x_hier, test_x_hier = _build_normalized_grids()

    # Build y_hier (49, 49) from the 4D neg_fid_table (7,7,7,7)
    neg_fid_table = data['neg_fid_table'].double()

    # y_hier[i, j] where i indexes child1 (gen) configs, j indexes child2 (disc) configs
    # Child1 config i maps to (lr_gen_idx, z_dim_idx) = (i // 7, i % 7)
    # Child2 config j maps to (lr_disc_idx, dropout_idx) = (j // 7, j % 7)
    y_hier = torch.zeros(49, 49).double()
    for i in range(49):
        lr_gen_idx = i // 7
        z_dim_idx = i % 7
        for j in range(49):
            lr_disc_idx = j // 7
            dropout_idx = j % 7
            y_hier[i, j] = neg_fid_table[lr_gen_idx, z_dim_idx, lr_disc_idx, dropout_idx]

    # Child marginals (pre-computed in build_surrogate.py)
    y_sub1 = data['y_gen_marginal'].double()   # (49,)
    y_sub2 = data['y_disc_marginal'].double()  # (49,)

    test_x = x_sub1  # Convention placeholder

    return x_sub1, y_sub1, x_sub2, y_sub2, x_hier, y_hier, test_x, test_x_hier


def select_random_queries_2d(nbr_init, x_domain, y_domain, seed=None, noise=0.0):
    """
    Randomly select nbr_init points from a 2D child domain.

    Args:
        nbr_init: number of initial points
        x_domain: (N, 2) tensor of all domain points
        y_domain: (N,) tensor of function values
        seed: random seed (int or False)
        noise: additive Gaussian noise scale

    Returns:
        train_x: (nbr_init, 2) selected inputs
        train_y: (nbr_init,) selected outputs (with noise)
    """
    if seed is not None and seed is not False:
        np.random.seed(seed)

    indices = np.random.randint(0, len(x_domain), size=nbr_init)
    train_x = x_domain[indices].clone()
    train_y = y_domain[indices].clone()

    if noise > 0:
        y_range = torch.max(y_domain) - torch.min(y_domain)
        train_y = train_y + torch.tensor(
            np.random.normal(0, noise * y_range.item(), size=train_y.shape),
            dtype=train_y.dtype
        )

    return train_x, train_y


def lookup_surrogate(query_pins, test_x_hier, y_hier, noise=0.0):
    """
    Look up the surrogate value for a 4D query point.

    Args:
        query_pins: (4,) tensor — the query point in normalized coordinates
        test_x_hier: (2401, 4) — all test points
        y_hier: (49, 49) — the neg-FID values

    Returns:
        value: scalar tensor — the neg-FID (with optional noise)
    """
    # Find matching index in test_x_hier
    for idx in range(len(test_x_hier)):
        if torch.allclose(test_x_hier[idx], query_pins, atol=1e-6):
            # Convert flat index to (i, j)
            i = idx // 49
            j = idx % 49
            value = y_hier[i, j].clone()
            if noise > 0:
                y_range = torch.max(y_hier) - torch.min(y_hier)
                value = value + torch.tensor(
                    np.random.normal(0, noise * y_range.item()),
                    dtype=value.dtype
                )
            return value, value  # (random, mean) to match existing interface

    raise ValueError(f"Query point {query_pins} not found in test_x_hier")
