"""
Generate appendix figure for the BIF paper:
  - 4 landscape heatmaps (gen + disc for both Task A and Task B)
  - Non-linearity analysis: test whether FID = f(gen) + g(disc) holds
    by comparing the additive reconstruction to the true surrogate.
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SURR_A = os.path.join(BASE_DIR, 'gan_surrogate_v2.pt')
SURR_B = os.path.join(BASE_DIR, 'gan_surrogate_v2_specnorm.pt')

# Grid labels
LR_GEN_LABELS  = ['1e-5', '5e-5', '1e-4', '5e-4', '1e-3', '5e-3', '1e-2']
WD_GEN_LABELS  = ['0', '1e-5', '1e-4', '1e-3', '5e-3', '1e-2', '5e-2']
LR_DISC_LABELS = ['5e-5', '1e-4', '2e-4', '5e-4', '1e-3', '2e-3', '5e-3']
DROPOUT_LABELS = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']


def annotate_cells(ax, data, fontsize=6):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = 'white' if val > np.median(data) else 'black'
            ax.text(i, j, f'{val:.0f}', ha='center', va='center',
                    fontsize=fontsize, color=color)


# ---------------------------------------------------------------
# Load surrogates
# ---------------------------------------------------------------
data_a = torch.load(SURR_A, weights_only=False)
data_b = torch.load(SURR_B, weights_only=False)
fid_a = data_a['fid_table'].numpy()  # (7,7,7,7)
fid_b = data_b['fid_table'].numpy()

best_a = np.unravel_index(np.argmin(fid_a), fid_a.shape)
best_b = np.unravel_index(np.argmin(fid_b), fid_b.shape)

neg_fid_a = -fid_a  # neg-FID (higher = better, matches the optimization)
neg_fid_b = -fid_b


# ===============================================================
# Figure 1: Appendix landscape (2x2: gen/disc x task A/B)
# ===============================================================
print('Generating appendix landscape figure...')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Gen landscape Task A
ax = axes[0, 0]
sl = fid_a[:, :, best_a[2], best_a[3]]
im = ax.imshow(sl.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_GEN_LABELS, rotation=45, fontsize=8)
ax.set_yticks(range(7)); ax.set_yticklabels(WD_GEN_LABELS, fontsize=8)
ax.set_xlabel('Generator LR'); ax.set_ylabel('Weight Decay')
ax.set_title(f'(a) Generator -- Standard Task\n'
             f'Disc: LR={LR_DISC_LABELS[best_a[2]]}, drop={DROPOUT_LABELS[best_a[3]]}',
             fontsize=11)
ax.plot(best_a[0], best_a[1], 'r*', ms=12, mec='white', mew=1)
plt.colorbar(im, ax=ax, shrink=0.8).set_label('FID')
annotate_cells(ax, sl)

# (b) Disc landscape Task A
ax = axes[0, 1]
sl = fid_a[best_a[0], best_a[1], :, :]
im = ax.imshow(sl.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_DISC_LABELS, rotation=45, fontsize=8)
ax.set_yticks(range(7)); ax.set_yticklabels(DROPOUT_LABELS, fontsize=8)
ax.set_xlabel('Discriminator LR'); ax.set_ylabel('Dropout')
ax.set_title(f'(b) Discriminator -- Standard Task\n'
             f'Gen: LR={LR_GEN_LABELS[best_a[0]]}, WD={WD_GEN_LABELS[best_a[1]]}',
             fontsize=11)
ax.plot(best_a[2], best_a[3], 'r*', ms=12, mec='white', mew=1)
plt.colorbar(im, ax=ax, shrink=0.8).set_label('FID')
annotate_cells(ax, sl)

# (c) Gen landscape Task B
ax = axes[1, 0]
sl = fid_b[:, :, best_b[2], best_b[3]]
im = ax.imshow(sl.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_GEN_LABELS, rotation=45, fontsize=8)
ax.set_yticks(range(7)); ax.set_yticklabels(WD_GEN_LABELS, fontsize=8)
ax.set_xlabel('Generator LR'); ax.set_ylabel('Weight Decay')
ax.set_title(f'(c) Generator -- SpectralNorm Task\n'
             f'Disc: LR={LR_DISC_LABELS[best_b[2]]}, drop={DROPOUT_LABELS[best_b[3]]}',
             fontsize=11)
ax.plot(best_b[0], best_b[1], 'r*', ms=12, mec='white', mew=1)
plt.colorbar(im, ax=ax, shrink=0.8).set_label('FID')
annotate_cells(ax, sl)

# (d) Disc landscape Task B
ax = axes[1, 1]
sl = fid_b[best_b[0], best_b[1], :, :]
im = ax.imshow(sl.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_DISC_LABELS, rotation=45, fontsize=8)
ax.set_yticks(range(7)); ax.set_yticklabels(DROPOUT_LABELS, fontsize=8)
ax.set_xlabel('Discriminator LR'); ax.set_ylabel('Dropout')
ax.set_title(f'(d) Discriminator -- SpectralNorm Task\n'
             f'Gen: LR={LR_GEN_LABELS[best_b[0]]}, WD={WD_GEN_LABELS[best_b[1]]}',
             fontsize=11)
ax.plot(best_b[2], best_b[3], 'r*', ms=12, mec='white', mew=1)
plt.colorbar(im, ax=ax, shrink=0.8).set_label('FID')
annotate_cells(ax, sl)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'appendix_gan_landscapes.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'appendix_gan_landscapes.svg'), bbox_inches='tight')
plt.close()
print('  Saved appendix_gan_landscapes.png/svg')


# ===============================================================
# Non-linearity analysis
# ===============================================================
print('\n=== Non-linearity Analysis ===')

def analyze_additivity(fid_table, task_name):
    """
    Test whether FID(gen, disc) = f(gen) + g(disc) + const.

    If the function were perfectly additive, the marginal means would
    reconstruct the full table exactly. The residual measures the
    non-additive (interaction) component.
    """
    # fid_table: (7, 7, 7, 7) = [lr_gen, wd_gen, lr_disc, dropout]
    # Reshape to (49, 49): gen_config x disc_config
    nf = -fid_table  # work with neg-FID (higher=better, matches optimization)
    flat = nf.reshape(49, 49)

    # Marginals
    gen_marginal = flat.mean(axis=1)   # (49,) mean over disc configs
    disc_marginal = flat.mean(axis=0)  # (49,) mean over gen configs
    grand_mean = flat.mean()

    # Additive reconstruction: f(i) + g(j) - grand_mean
    additive = gen_marginal[:, None] + disc_marginal[None, :] - grand_mean  # (49,49)

    # Residual = true - additive
    residual = flat - additive

    # Variance decomposition
    ss_total = np.sum((flat - grand_mean)**2)
    ss_additive = np.sum((additive - grand_mean)**2)
    ss_residual = np.sum(residual**2)

    # R2 of additive model
    r2_additive = 1 - ss_residual / ss_total

    # Relative interaction strength
    interaction_frac = ss_residual / ss_total

    print(f'\n  {task_name}:')
    print(f'    SS total     = {ss_total:.2f}')
    print(f'    SS additive  = {ss_additive:.2f}  ({ss_additive/ss_total*100:.1f}%)')
    print(f'    SS residual  = {ss_residual:.2f}  ({ss_residual/ss_total*100:.1f}%)')
    print(f'    R2 additive  = {r2_additive:.4f}')
    print(f'    Interaction  = {interaction_frac*100:.1f}% of total variance')

    # Linear regression: true vs additive
    lr = linregress(flat.flatten(), additive.flatten())
    print(f'    Linear R2 (true vs additive) = {lr.rvalue**2:.4f}')

    return flat, additive, residual, r2_additive, interaction_frac


flat_a, add_a, res_a, r2_a, int_a = analyze_additivity(fid_a, 'Task A (Standard)')
flat_b, add_b, res_b, r2_b, int_b = analyze_additivity(fid_b, 'Task B (SpectralNorm)')


# ===============================================================
# Figure 2: Non-linearity evidence (3 columns x 2 rows)
# Row 1: Task A, Row 2: Task B
# Col 1: True landscape, Col 2: Additive reconstruction, Col 3: Residual
# ===============================================================
print('\nGenerating non-linearity figure...')
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for row, (flat, add, res, r2, ifrac, name) in enumerate([
    (flat_a, add_a, res_a, r2_a, int_a, 'Standard'),
    (flat_b, add_b, res_b, r2_b, int_b, 'SpectralNorm')]):

    vmin_true = min(flat.min(), add.min())
    vmax_true = max(flat.max(), add.max())

    ax = axes[row, 0]
    im = ax.imshow(flat, origin='lower', aspect='auto', cmap='RdYlGn')
    ax.set_title(f'({chr(97+row*3)}) True neg-FID -- {name}', fontsize=11)
    ax.set_xlabel('Disc config (j)'); ax.set_ylabel('Gen config (i)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[row, 1]
    im = ax.imshow(add, origin='lower', aspect='auto', cmap='RdYlGn',
                   vmin=vmin_true, vmax=vmax_true)
    ax.set_title(f'({chr(98+row*3)}) Additive model ($R^2$={r2:.3f})', fontsize=11)
    ax.set_xlabel('Disc config (j)'); ax.set_ylabel('Gen config (i)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[row, 2]
    vlim = max(abs(res.min()), abs(res.max()))
    im = ax.imshow(res, origin='lower', aspect='auto', cmap='RdBu_r',
                   vmin=-vlim, vmax=vlim)
    ax.set_title(f'({chr(99+row*3)}) Interaction residual ({ifrac*100:.1f}% of var)',
                 fontsize=11)
    ax.set_xlabel('Disc config (j)'); ax.set_ylabel('Gen config (i)')
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle('Additivity Test: True FID vs. Additive Decomposition\n'
             '(Residual = non-linear gen-disc interaction)', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'appendix_gan_nonlinearity.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'appendix_gan_nonlinearity.svg'), bbox_inches='tight')
plt.close()
print('  Saved appendix_gan_nonlinearity.png/svg')


# ===============================================================
# Figure 3: Scatter plot — true vs additive (both tasks)
# ===============================================================
print('Generating scatter plot...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

for ax, flat, add, r2, name in [
    (ax1, flat_a, add_a, r2_a, 'Standard'),
    (ax2, flat_b, add_b, r2_b, 'SpectralNorm')]:
    ax.scatter(flat.flatten(), add.flatten(), alpha=0.3, s=8, color='steelblue')
    lims = [min(flat.min(), add.min()), max(flat.max(), add.max())]
    ax.plot(lims, lims, 'r--', lw=1.5, label='y = x (perfect additivity)')
    lr = linregress(flat.flatten(), add.flatten())
    xfit = np.linspace(lims[0], lims[1], 100)
    ax.plot(xfit, lr.slope * xfit + lr.intercept, 'k-', lw=1,
            label=f'Fit: $R^2$={lr.rvalue**2:.3f}')
    ax.set_xlabel('True neg-FID', fontsize=11)
    ax.set_ylabel('Additive reconstruction', fontsize=11)
    ax.set_title(f'{name} Task\n'
                 f'Additive $R^2$ = {r2:.3f}, Interaction = {(1-r2)*100:.1f}%',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'appendix_gan_scatter_additivity.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_DIR, 'appendix_gan_scatter_additivity.svg'), bbox_inches='tight')
plt.close()
print('  Saved appendix_gan_scatter_additivity.png/svg')

print('\nDone!')
