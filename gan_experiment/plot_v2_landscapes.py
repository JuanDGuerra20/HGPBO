"""
Generate landscape heatmap figures for the v2 GAN surrogates.

v2 surrogates use: lr_gen + weight_decay + lr_disc + dropout (fixed z_dim=64).
- Task A: standard BatchNorm discriminator  (gan_surrogate_v2.pt)
- Task B: SpectralNorm discriminator        (gan_surrogate_v2_specnorm.pt)

fid_table shape: (7, 7, 7, 7) indexed as [lr_gen_idx, wd_gen_idx, lr_disc_idx, dropout_idx]

Usage:
    python plot_v2_landscapes.py
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SURR_A = os.path.join(BASE_DIR, 'gan_surrogate_v2.pt')
SURR_B = os.path.join(BASE_DIR, 'gan_surrogate_v2_specnorm.pt')

# ---------------------------------------------------------------------------
# v2 Grid definitions (must match train_dcgan_v2.py)
# ---------------------------------------------------------------------------
LR_GEN_VALS  = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
WD_GEN_VALS  = [0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2]
LR_DISC_VALS = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
DROPOUT_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

LR_GEN_LABELS  = ['1e-5', '5e-5', '1e-4', '5e-4', '1e-3', '5e-3', '1e-2']
WD_GEN_LABELS  = ['0', '1e-5', '1e-4', '1e-3', '5e-3', '1e-2', '5e-2']
LR_DISC_LABELS = ['5e-5', '1e-4', '2e-4', '5e-4', '1e-3', '2e-3', '5e-3']
DROPOUT_LABELS = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def annotate_cells(ax, data, fontsize=7):
    """Annotate heatmap cells with FID values."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = 'white' if val > np.median(data) else 'black'
            ax.text(i, j, f'{val:.0f}', ha='center', va='center',
                    fontsize=fontsize, color=color)


def save_fig(fig, name):
    """Save figure as both PNG and SVG."""
    png_path = os.path.join(FIG_DIR, f'{name}.png')
    svg_path = os.path.join(FIG_DIR, f'{name}.svg')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {name}.png / .svg')


# ---------------------------------------------------------------------------
# Load surrogates
# ---------------------------------------------------------------------------
print('Loading v2 surrogates...')
data_a = torch.load(SURR_A, weights_only=False)
data_b = torch.load(SURR_B, weights_only=False)

fid_a = data_a['fid_table'].numpy()  # (7,7,7,7)
fid_b = data_b['fid_table'].numpy()

# Find global optima
best_a = np.unravel_index(np.argmin(fid_a), fid_a.shape)
best_b = np.unravel_index(np.argmin(fid_b), fid_b.shape)

print(f'\n=== Task A (Standard Disc) ===')
print(f'  Global optimum: FID = {fid_a[best_a]:.2f}')
print(f'    lr_gen      = {LR_GEN_VALS[best_a[0]]}  (idx {best_a[0]})')
print(f'    weight_decay= {WD_GEN_VALS[best_a[1]]}  (idx {best_a[1]})')
print(f'    lr_disc     = {LR_DISC_VALS[best_a[2]]}  (idx {best_a[2]})')
print(f'    dropout     = {DROPOUT_VALS[best_a[3]]}  (idx {best_a[3]})')
print(f'  FID range: [{fid_a.min():.1f}, {fid_a.max():.1f}], mean={fid_a.mean():.1f}')

print(f'\n=== Task B (SpectralNorm Disc) ===')
print(f'  Global optimum: FID = {fid_b[best_b]:.2f}')
print(f'    lr_gen      = {LR_GEN_VALS[best_b[0]]}  (idx {best_b[0]})')
print(f'    weight_decay= {WD_GEN_VALS[best_b[1]]}  (idx {best_b[1]})')
print(f'    lr_disc     = {LR_DISC_VALS[best_b[2]]}  (idx {best_b[2]})')
print(f'    dropout     = {DROPOUT_VALS[best_b[3]]}  (idx {best_b[3]})')
print(f'  FID range: [{fid_b.min():.1f}, {fid_b.max():.1f}], mean={fid_b.mean():.1f}')
print()


# ===================================================================
# Figure 1: generator_landscape_v2 (Task A)
# ===================================================================
print('Figure 1: generator_landscape_v2 (Task A)')
fig, ax = plt.subplots(figsize=(7, 5.5))
slice_gen_a = fid_a[:, :, best_a[2], best_a[3]]  # (7, 7) = lr_gen x wd_gen
im = ax.imshow(slice_gen_a.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_GEN_LABELS, rotation=45, fontsize=9)
ax.set_yticks(range(7)); ax.set_yticklabels(WD_GEN_LABELS, fontsize=9)
ax.set_xlabel('Generator Learning Rate', fontsize=12)
ax.set_ylabel('Weight Decay', fontsize=12)
lr_disc_opt = LR_DISC_LABELS[best_a[2]]
do_opt = DROPOUT_LABELS[best_a[3]]
ax.set_title(f'Generator Landscape (Task A -- Standard Disc)\n'
             f'Disc fixed at LR={lr_disc_opt}, dropout={do_opt}', fontsize=12)
ax.plot(best_a[0], best_a[1], 'r*', markersize=15,
        markeredgecolor='white', markeredgewidth=1)
cbar = plt.colorbar(im, ax=ax); cbar.set_label('FID (lower = better)', fontsize=11)
annotate_cells(ax, slice_gen_a)
plt.tight_layout()
save_fig(fig, 'generator_landscape_v2')


# ===================================================================
# Figure 2: discriminator_landscape_v2 (Task A)
# ===================================================================
print('Figure 2: discriminator_landscape_v2 (Task A)')
fig, ax = plt.subplots(figsize=(7, 5.5))
slice_disc_a = fid_a[best_a[0], best_a[1], :, :]  # (7, 7) = lr_disc x dropout
im = ax.imshow(slice_disc_a.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_DISC_LABELS, rotation=45, fontsize=9)
ax.set_yticks(range(7)); ax.set_yticklabels(DROPOUT_LABELS, fontsize=9)
ax.set_xlabel('Discriminator Learning Rate', fontsize=12)
ax.set_ylabel('Dropout Rate', fontsize=12)
lr_gen_opt = LR_GEN_LABELS[best_a[0]]
wd_opt = WD_GEN_LABELS[best_a[1]]
ax.set_title(f'Discriminator Landscape (Task A -- Standard Disc)\n'
             f'Gen fixed at LR={lr_gen_opt}, WD={wd_opt}', fontsize=12)
ax.plot(best_a[2], best_a[3], 'r*', markersize=15,
        markeredgecolor='white', markeredgewidth=1)
cbar = plt.colorbar(im, ax=ax); cbar.set_label('FID (lower = better)', fontsize=11)
annotate_cells(ax, slice_disc_a)
plt.tight_layout()
save_fig(fig, 'discriminator_landscape_v2')


# ===================================================================
# Figure 3: dcgan_combined2_v2 (Task A side-by-side)
# ===================================================================
print('Figure 3: dcgan_combined2_v2 (Task A)')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

im1 = ax1.imshow(slice_gen_a.T, origin='lower', aspect='auto', cmap='viridis_r')
ax1.set_xticks(range(7)); ax1.set_xticklabels(LR_GEN_LABELS, rotation=45, fontsize=9)
ax1.set_yticks(range(7)); ax1.set_yticklabels(WD_GEN_LABELS, fontsize=9)
ax1.set_xlabel('Generator Learning Rate', fontsize=12)
ax1.set_ylabel('Weight Decay', fontsize=12)
ax1.set_title(f'(a) Generator Child (Task A)\n'
              f'Disc: LR={lr_disc_opt}, dropout={do_opt}', fontsize=12)
ax1.plot(best_a[0], best_a[1], 'r*', markersize=15,
         markeredgecolor='white', markeredgewidth=1)
cbar1 = plt.colorbar(im1, ax=ax1); cbar1.set_label('FID', fontsize=10)
annotate_cells(ax1, slice_gen_a)

im2 = ax2.imshow(slice_disc_a.T, origin='lower', aspect='auto', cmap='viridis_r')
ax2.set_xticks(range(7)); ax2.set_xticklabels(LR_DISC_LABELS, rotation=45, fontsize=9)
ax2.set_yticks(range(7)); ax2.set_yticklabels(DROPOUT_LABELS, fontsize=9)
ax2.set_xlabel('Discriminator Learning Rate', fontsize=12)
ax2.set_ylabel('Dropout Rate', fontsize=12)
ax2.set_title(f'(b) Discriminator Child (Task A)\n'
              f'Gen: LR={lr_gen_opt}, WD={wd_opt}', fontsize=12)
ax2.plot(best_a[2], best_a[3], 'r*', markersize=15,
         markeredgecolor='white', markeredgewidth=1)
cbar2 = plt.colorbar(im2, ax=ax2); cbar2.set_label('FID', fontsize=10)
annotate_cells(ax2, slice_disc_a)

plt.suptitle('DCGAN on Fashion-MNIST (v2): Hyperparameter Landscape', fontsize=14, y=1.02)
plt.tight_layout()
save_fig(fig, 'dcgan_combined2_v2')


# ===================================================================
# Figure 4: generator_landscape_v2_specnorm (Task B)
# ===================================================================
print('Figure 4: generator_landscape_v2_specnorm (Task B)')
fig, ax = plt.subplots(figsize=(7, 5.5))
slice_gen_b = fid_b[:, :, best_b[2], best_b[3]]  # (7, 7)
im = ax.imshow(slice_gen_b.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_GEN_LABELS, rotation=45, fontsize=9)
ax.set_yticks(range(7)); ax.set_yticklabels(WD_GEN_LABELS, fontsize=9)
ax.set_xlabel('Generator Learning Rate', fontsize=12)
ax.set_ylabel('Weight Decay', fontsize=12)
lr_disc_opt_b = LR_DISC_LABELS[best_b[2]]
do_opt_b = DROPOUT_LABELS[best_b[3]]
ax.set_title(f'Generator Landscape (Task B -- SpectralNorm Disc)\n'
             f'Disc fixed at LR={lr_disc_opt_b}, dropout={do_opt_b}', fontsize=12)
ax.plot(best_b[0], best_b[1], 'r*', markersize=15,
        markeredgecolor='white', markeredgewidth=1)
cbar = plt.colorbar(im, ax=ax); cbar.set_label('FID (lower = better)', fontsize=11)
annotate_cells(ax, slice_gen_b)
plt.tight_layout()
save_fig(fig, 'generator_landscape_v2_specnorm')


# ===================================================================
# Figure 5: discriminator_landscape_v2_specnorm (Task B)
# ===================================================================
print('Figure 5: discriminator_landscape_v2_specnorm (Task B)')
fig, ax = plt.subplots(figsize=(7, 5.5))
slice_disc_b = fid_b[best_b[0], best_b[1], :, :]  # (7, 7)
im = ax.imshow(slice_disc_b.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_DISC_LABELS, rotation=45, fontsize=9)
ax.set_yticks(range(7)); ax.set_yticklabels(DROPOUT_LABELS, fontsize=9)
ax.set_xlabel('Discriminator Learning Rate', fontsize=12)
ax.set_ylabel('Dropout Rate', fontsize=12)
lr_gen_opt_b = LR_GEN_LABELS[best_b[0]]
wd_opt_b = WD_GEN_LABELS[best_b[1]]
ax.set_title(f'Discriminator Landscape (Task B -- SpectralNorm Disc)\n'
             f'Gen fixed at LR={lr_gen_opt_b}, WD={wd_opt_b}', fontsize=12)
ax.plot(best_b[2], best_b[3], 'r*', markersize=15,
        markeredgecolor='white', markeredgewidth=1)
cbar = plt.colorbar(im, ax=ax); cbar.set_label('FID (lower = better)', fontsize=11)
annotate_cells(ax, slice_disc_b)
plt.tight_layout()
save_fig(fig, 'discriminator_landscape_v2_specnorm')


# ===================================================================
# Figure 6: dcgan_combined2_v2_specnorm (Task B side-by-side)
# ===================================================================
print('Figure 6: dcgan_combined2_v2_specnorm (Task B)')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

im1 = ax1.imshow(slice_gen_b.T, origin='lower', aspect='auto', cmap='viridis_r')
ax1.set_xticks(range(7)); ax1.set_xticklabels(LR_GEN_LABELS, rotation=45, fontsize=9)
ax1.set_yticks(range(7)); ax1.set_yticklabels(WD_GEN_LABELS, fontsize=9)
ax1.set_xlabel('Generator Learning Rate', fontsize=12)
ax1.set_ylabel('Weight Decay', fontsize=12)
ax1.set_title(f'(a) Generator Child (Task B)\n'
              f'Disc: LR={lr_disc_opt_b}, dropout={do_opt_b}', fontsize=12)
ax1.plot(best_b[0], best_b[1], 'r*', markersize=15,
         markeredgecolor='white', markeredgewidth=1)
cbar1 = plt.colorbar(im1, ax=ax1); cbar1.set_label('FID', fontsize=10)
annotate_cells(ax1, slice_gen_b)

im2 = ax2.imshow(slice_disc_b.T, origin='lower', aspect='auto', cmap='viridis_r')
ax2.set_xticks(range(7)); ax2.set_xticklabels(LR_DISC_LABELS, rotation=45, fontsize=9)
ax2.set_yticks(range(7)); ax2.set_yticklabels(DROPOUT_LABELS, fontsize=9)
ax2.set_xlabel('Discriminator Learning Rate', fontsize=12)
ax2.set_ylabel('Dropout Rate', fontsize=12)
ax2.set_title(f'(b) Discriminator Child (Task B)\n'
              f'Gen: LR={lr_gen_opt_b}, WD={wd_opt_b}', fontsize=12)
ax2.plot(best_b[2], best_b[3], 'r*', markersize=15,
         markeredgecolor='white', markeredgewidth=1)
cbar2 = plt.colorbar(im2, ax=ax2); cbar2.set_label('FID', fontsize=10)
annotate_cells(ax2, slice_disc_b)

plt.suptitle('DCGAN + SpectralNorm Disc on Fashion-MNIST (v2): Hyperparameter Landscape',
             fontsize=14, y=1.02)
plt.tight_layout()
save_fig(fig, 'dcgan_combined2_v2_specnorm')


# ===================================================================
# Figure 7: disc_landscape_comparison_v2 (Task A vs Task B disc)
# ===================================================================
print('Figure 7: disc_landscape_comparison_v2')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Use Task A generator optimum for both -- same gen config, different disc arch
slice_disc_a_comp = fid_a[best_a[0], best_a[1], :, :]
slice_disc_b_comp = fid_b[best_a[0], best_a[1], :, :]  # same gen config

vmin = min(slice_disc_a_comp.min(), slice_disc_b_comp.min())
vmax = max(slice_disc_a_comp.max(), slice_disc_b_comp.max())

im1 = ax1.imshow(slice_disc_a_comp.T, origin='lower', aspect='auto',
                  cmap='viridis_r', vmin=vmin, vmax=vmax)
ax1.set_xticks(range(7)); ax1.set_xticklabels(LR_DISC_LABELS, rotation=45, fontsize=9)
ax1.set_yticks(range(7)); ax1.set_yticklabels(DROPOUT_LABELS, fontsize=9)
ax1.set_xlabel('Discriminator Learning Rate', fontsize=12)
ax1.set_ylabel('Dropout Rate', fontsize=12)
ax1.set_title(f'(a) Task A -- Standard Disc (BatchNorm)\n'
              f'Gen: LR={LR_GEN_LABELS[best_a[0]]}, WD={WD_GEN_LABELS[best_a[1]]}',
              fontsize=12)
ax1.plot(best_a[2], best_a[3], 'r*', markersize=15,
         markeredgecolor='white', markeredgewidth=1)
annotate_cells(ax1, slice_disc_a_comp)

im2 = ax2.imshow(slice_disc_b_comp.T, origin='lower', aspect='auto',
                  cmap='viridis_r', vmin=vmin, vmax=vmax)
ax2.set_xticks(range(7)); ax2.set_xticklabels(LR_DISC_LABELS, rotation=45, fontsize=9)
ax2.set_yticks(range(7)); ax2.set_yticklabels(DROPOUT_LABELS, fontsize=9)
ax2.set_xlabel('Discriminator Learning Rate', fontsize=12)
ax2.set_ylabel('Dropout Rate', fontsize=12)
# Find best disc config for Task B under same gen config
best_disc_b_comp = np.unravel_index(np.argmin(slice_disc_b_comp), slice_disc_b_comp.shape)
ax2.set_title(f'(b) Task B -- SpectralNorm Disc\n'
              f'Gen: LR={LR_GEN_LABELS[best_a[0]]}, WD={WD_GEN_LABELS[best_a[1]]}',
              fontsize=12)
ax2.plot(best_disc_b_comp[0], best_disc_b_comp[1], 'r*', markersize=15,
         markeredgecolor='white', markeredgewidth=1)
annotate_cells(ax2, slice_disc_b_comp)

cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, pad=0.02)
cbar.set_label('FID (shared scale)', fontsize=11)

plt.suptitle('Discriminator Landscape Shift: Standard vs SpectralNorm\n'
             '(Same generator config, different disc architecture)',
             fontsize=13, y=1.04)
plt.tight_layout()
save_fig(fig, 'disc_landscape_comparison_v2')


# ===================================================================
# Figure 8: gan_landscape_v2 (6-panel overview for Task A)
# ===================================================================
print('Figure 8: gan_landscape_v2 (6-panel overview)')
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# --- Row 1, Col 1: Generator slice (lr_gen x wd_gen, disc at optimum) ---
ax = axes[0, 0]
im = ax.imshow(slice_gen_a.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_GEN_LABELS, rotation=45, fontsize=8)
ax.set_yticks(range(7)); ax.set_yticklabels(WD_GEN_LABELS, fontsize=8)
ax.set_xlabel('Generator LR', fontsize=10)
ax.set_ylabel('Weight Decay', fontsize=10)
ax.set_title(f'(a) Generator Slice\nDisc: LR={lr_disc_opt}, dropout={do_opt}', fontsize=11)
ax.plot(best_a[0], best_a[1], 'r*', markersize=12,
        markeredgecolor='white', markeredgewidth=1)
plt.colorbar(im, ax=ax, shrink=0.8).set_label('FID', fontsize=9)
annotate_cells(ax, slice_gen_a, fontsize=6)

# --- Row 1, Col 2: Discriminator slice (lr_disc x dropout, gen at optimum) ---
ax = axes[0, 1]
im = ax.imshow(slice_disc_a.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_DISC_LABELS, rotation=45, fontsize=8)
ax.set_yticks(range(7)); ax.set_yticklabels(DROPOUT_LABELS, fontsize=8)
ax.set_xlabel('Discriminator LR', fontsize=10)
ax.set_ylabel('Dropout Rate', fontsize=10)
ax.set_title(f'(b) Discriminator Slice\nGen: LR={lr_gen_opt}, WD={wd_opt}', fontsize=11)
ax.plot(best_a[2], best_a[3], 'r*', markersize=12,
        markeredgecolor='white', markeredgewidth=1)
plt.colorbar(im, ax=ax, shrink=0.8).set_label('FID', fontsize=9)
annotate_cells(ax, slice_disc_a, fontsize=6)

# --- Row 1, Col 3: LR interaction (lr_gen x lr_disc, wd & dropout at optimum) ---
ax = axes[0, 2]
slice_lr_int = fid_a[:, best_a[1], :, best_a[3]]  # (7, 7) = lr_gen x lr_disc
im = ax.imshow(slice_lr_int.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_GEN_LABELS, rotation=45, fontsize=8)
ax.set_yticks(range(7)); ax.set_yticklabels(LR_DISC_LABELS, fontsize=8)
ax.set_xlabel('Generator LR', fontsize=10)
ax.set_ylabel('Discriminator LR', fontsize=10)
ax.set_title(f'(c) LR Interaction\nWD={wd_opt}, dropout={do_opt}', fontsize=11)
ax.plot(best_a[0], best_a[2], 'r*', markersize=12,
        markeredgecolor='white', markeredgewidth=1)
plt.colorbar(im, ax=ax, shrink=0.8).set_label('FID', fontsize=9)
annotate_cells(ax, slice_lr_int, fontsize=6)

# --- Row 2, Col 1: Generator marginal (mean FID over all disc configs) ---
ax = axes[1, 0]
gen_marginal = fid_a.mean(axis=(2, 3))  # (7, 7) = lr_gen x wd_gen
im = ax.imshow(gen_marginal.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_GEN_LABELS, rotation=45, fontsize=8)
ax.set_yticks(range(7)); ax.set_yticklabels(WD_GEN_LABELS, fontsize=8)
ax.set_xlabel('Generator LR', fontsize=10)
ax.set_ylabel('Weight Decay', fontsize=10)
ax.set_title('(d) Gen Marginal (mean over disc)', fontsize=11)
best_marg_gen = np.unravel_index(np.argmin(gen_marginal), gen_marginal.shape)
ax.plot(best_marg_gen[0], best_marg_gen[1], 'r*', markersize=12,
        markeredgecolor='white', markeredgewidth=1)
plt.colorbar(im, ax=ax, shrink=0.8).set_label('FID', fontsize=9)
annotate_cells(ax, gen_marginal, fontsize=6)

# --- Row 2, Col 2: Discriminator marginal (mean FID over all gen configs) ---
ax = axes[1, 1]
disc_marginal = fid_a.mean(axis=(0, 1))  # (7, 7) = lr_disc x dropout
im = ax.imshow(disc_marginal.T, origin='lower', aspect='auto', cmap='viridis_r')
ax.set_xticks(range(7)); ax.set_xticklabels(LR_DISC_LABELS, rotation=45, fontsize=8)
ax.set_yticks(range(7)); ax.set_yticklabels(DROPOUT_LABELS, fontsize=8)
ax.set_xlabel('Discriminator LR', fontsize=10)
ax.set_ylabel('Dropout Rate', fontsize=10)
ax.set_title('(e) Disc Marginal (mean over gen)', fontsize=11)
best_marg_disc = np.unravel_index(np.argmin(disc_marginal), disc_marginal.shape)
ax.plot(best_marg_disc[0], best_marg_disc[1], 'r*', markersize=12,
        markeredgecolor='white', markeredgewidth=1)
plt.colorbar(im, ax=ax, shrink=0.8).set_label('FID', fontsize=9)
annotate_cells(ax, disc_marginal, fontsize=6)

# --- Row 2, Col 3: FID histogram ---
ax = axes[1, 2]
all_fids = fid_a.flatten()
# Exclude capped values for cleaner histogram
non_capped = all_fids[all_fids < 300.0]
n_capped = int(np.sum(all_fids >= 300.0))

ax.hist(non_capped, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(np.median(all_fids), color='red', linestyle='--', linewidth=1.5,
           label=f'Median = {np.median(all_fids):.1f}')
ax.axvline(all_fids.mean(), color='orange', linestyle='--', linewidth=1.5,
           label=f'Mean = {all_fids.mean():.1f}')
ax.axvline(all_fids.min(), color='green', linestyle='-', linewidth=1.5,
           label=f'Min = {all_fids.min():.1f}')
if n_capped > 0:
    ax.annotate(f'{n_capped} capped at 300',
                xy=(295, 0), xytext=(200, ax.get_ylim()[1] * 0.7),
                fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_xlabel('FID Score', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title(f'(f) FID Distribution ({len(all_fids)} configs)', fontsize=11)
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.3)

plt.suptitle('GAN Hyperparameter Landscape Overview (Task A, v2)', fontsize=14, y=1.01)
plt.tight_layout()
save_fig(fig, 'gan_landscape_v2')


print('\nDone! All figures saved to figures/')
