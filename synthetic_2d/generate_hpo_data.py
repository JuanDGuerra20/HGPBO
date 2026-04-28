"""
Pre-computation script for ElasticNet HPO benchmark dataset.
Run once from the synthetic_2d/ directory before running experiments:
    python generate_hpo_data.py

Generates a 32x32 grid of ElasticNet cross-validated R² on the diabetes
dataset, varying alpha x l1_ratio on linear scales. Saves to svm_hpo_data.npz.

Marginals are consistent slices of the joint: y_sub1 fixes l1_ratio at the
grid midpoint, y_sub2 fixes alpha at the grid midpoint.
"""

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

DIMENSION  = 32
ALPHA_VALS = np.linspace(0.001, 1.0, DIMENSION)   # regularization strength: 0.001 → 1.0
L1_VALS    = np.linspace(0.01,  0.99, DIMENSION)  # L1/L2 mix: 0.01 (≈Ridge) → 0.99 (≈Lasso)

MID       = DIMENSION // 2
ALPHA_MID = ALPHA_VALS[MID]
L1_MID    = L1_VALS[MID]

np.random.seed(42)
X, y = load_diabetes(return_X_y=True)

print(f"Computing {DIMENSION}-point marginals and {DIMENSION}x{DIMENSION} joint grid...")
print(f"alpha range: [{ALPHA_VALS[0]:.4f}, {ALPHA_VALS[-1]:.4f}]  (midpoint={ALPHA_MID:.4f})")
print(f"l1_ratio range: [{L1_VALS[0]:.3f}, {L1_VALS[-1]:.3f}]  (midpoint={L1_MID:.3f})")
print(f"Total CV evaluations: {2*DIMENSION + DIMENSION*DIMENSION}")

# y_sub1: vary alpha, l1_ratio fixed at grid midpoint (consistent slice of joint)
print("Computing y_sub1 (alpha marginal, l1_ratio fixed at midpoint)...")
y_sub1 = []
for a in tqdm(ALPHA_VALS):
    score = cross_val_score(ElasticNet(alpha=a, l1_ratio=L1_MID), X, y, cv=5).mean()
    y_sub1.append(score)

# y_sub2: vary l1_ratio, alpha fixed at grid midpoint (consistent slice of joint)
print("Computing y_sub2 (l1_ratio marginal, alpha fixed at midpoint)...")
y_sub2 = []
for l in tqdm(L1_VALS):
    score = cross_val_score(ElasticNet(alpha=ALPHA_MID, l1_ratio=l), X, y, cv=5).mean()
    y_sub2.append(score)

# y_hier: full factorial grid
print("Computing y_hier (full grid)...")
y_hier = np.zeros((DIMENSION, DIMENSION))
for i, a in enumerate(tqdm(ALPHA_VALS)):
    for j, l in enumerate(L1_VALS):
        y_hier[i, j] = cross_val_score(ElasticNet(alpha=a, l1_ratio=l), X, y, cv=5).mean()

np.savez('svm_hpo_data.npz',
         c_vals=ALPHA_VALS,
         gamma_vals=L1_VALS,
         y_sub1=np.array(y_sub1),
         y_sub2=np.array(y_sub2),
         y_hier=y_hier)

print(f"Saved to svm_hpo_data.npz")
print(f"y_hier range: [{y_hier.min():.4f}, {y_hier.max():.4f}]")
best_i, best_j = np.unravel_index(y_hier.argmax(), y_hier.shape)
print(f"Best: alpha={ALPHA_VALS[best_i]:.4f}, l1_ratio={L1_VALS[best_j]:.3f}, R²={y_hier.max():.4f}")
