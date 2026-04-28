"""
Build composite Figure 4 for the BIF paper.

Layout (3 rows x 2 cols):
  Row 1: (a) panelA.pdf          | (b) v3_fair_main_ro
  Row 2: (c) v3_fair_main_r2     | (d) panelB.pdf
  Row 3: (e) v3_fair_modularity_ro | (f) v3_fair_modularity_r2

Height ratios: row 1 and row 2 are taller (landscape+R2 panels),
row 3 same as row 1.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# Convert PDFs to images via matplotlib's PDF renderer
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

def pdf_to_image(pdf_path, dpi=200):
    """Render first page of a PDF to a numpy array."""
    # Use PIL if pdf2image is available, otherwise use fitz
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:  # RGBA
            img = img[:, :, :3]
        doc.close()
        return img
    except ImportError:
        pass

    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
        return np.array(images[0])
    except ImportError:
        pass

    # Fallback: try using Pillow directly (some installs support PDF)
    try:
        from PIL import Image
        img = Image.open(pdf_path)
        img.load()
        return np.array(img.convert('RGB'))
    except Exception:
        raise RuntimeError(f"Cannot render PDF {pdf_path}. Install PyMuPDF: pip install pymupdf")


def load_img(name):
    """Load PNG image from figures dir."""
    path = os.path.join(FIG_DIR, name)
    return mpimg.imread(path)


# Load all panels
print('Loading panels...')
panel_a = pdf_to_image(os.path.join(FIG_DIR, 'panelA.pdf'), dpi=250)
panel_b = load_img('v3_fair_main_ro.png')
panel_c = load_img('v3_fair_main_r2.png')
panel_d = pdf_to_image(os.path.join(FIG_DIR, 'panelB.pdf'), dpi=250)
panel_e = load_img('v3_fair_modularity_ro.png')
panel_f = load_img('v3_fair_modularity_r2.png')

print(f'  panelA: {panel_a.shape}')
print(f'  main_ro: {panel_b.shape}')
print(f'  main_r2: {panel_c.shape}')
print(f'  panelD: {panel_d.shape}')
print(f'  mod_ro: {panel_e.shape}')
print(f'  mod_r2: {panel_f.shape}')

# Build composite
# Layout: 3 rows x 2 columns
# Row heights proportional to content
fig = plt.figure(figsize=(16, 20))

# Use gridspec for flexible layout
gs = fig.add_gridspec(3, 2, hspace=0.08, wspace=0.05,
                      height_ratios=[1, 0.55, 1])

def add_panel(ax, img, label):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.8))

# Row 1: (a) panelA, (b) main RO
ax_a = fig.add_subplot(gs[0, 0])
add_panel(ax_a, panel_a, '(a)')

ax_b = fig.add_subplot(gs[0, 1])
add_panel(ax_b, panel_b, '(b)')

# Row 2: (c) main R2, (d) panelB
ax_c = fig.add_subplot(gs[1, 0])
add_panel(ax_c, panel_c, '(c)')

ax_d = fig.add_subplot(gs[1, 1])
add_panel(ax_d, panel_d, '(d)')

# Row 3: (e) modularity RO, (f) modularity R2
ax_e = fig.add_subplot(gs[2, 0])
add_panel(ax_e, panel_e, '(e)')

ax_f = fig.add_subplot(gs[2, 1])
add_panel(ax_f, panel_f, '(f)')

# Save
out_png = os.path.join(FIG_DIR, 'figure4_composite.png')
out_svg = os.path.join(FIG_DIR, 'figure4_composite.svg')
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(out_svg, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\nSaved {out_png}')
print(f'Saved {out_svg}')

# Also save as PDF
out_pdf = os.path.join(FIG_DIR, 'figure4_composite.pdf')
fig2 = plt.figure(figsize=(16, 20))
gs2 = fig2.add_gridspec(3, 2, hspace=0.08, wspace=0.05,
                        height_ratios=[1, 0.55, 1])

ax_a2 = fig2.add_subplot(gs2[0, 0]); add_panel(ax_a2, panel_a, '(a)')
ax_b2 = fig2.add_subplot(gs2[0, 1]); add_panel(ax_b2, panel_b, '(b)')
ax_c2 = fig2.add_subplot(gs2[1, 0]); add_panel(ax_c2, panel_c, '(c)')
ax_d2 = fig2.add_subplot(gs2[1, 1]); add_panel(ax_d2, panel_d, '(d)')
ax_e2 = fig2.add_subplot(gs2[2, 0]); add_panel(ax_e2, panel_e, '(e)')
ax_f2 = fig2.add_subplot(gs2[2, 1]); add_panel(ax_f2, panel_f, '(f)')

fig2.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out_pdf}')
