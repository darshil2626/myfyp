"""
report_style.py
================
Shared visual style for every figure that appears in the thesis report.

Design goals (agreed with the author):
  * One colour scheme per *quantity type*, used consistently everywhere:
        - SST / temperature fields .......... cmocean 'thermal'   (sequential)
        - |error| / MAE magnitude fields .... cmocean 'amp'       (sequential)
        - SST gradient magnitude ............ cmocean 'tempo'     (sequential)
        - signed fields (latent, anomalies) . cmocean 'balance'   (diverging, 0=white)
  * Line plots share a colourblind-safe palette (Okabe-Ito based):
        - the main model (SINN / combined) is the strong navy PRIMARY,
        - non-learned / reference series are muted grey / orange,
        - ordered families (patch size, latent dim) use a sequential ramp.
  * NO titles on any figure — the report supplies captions.
  * Slightly larger fonts than the originals for legibility.

Import this module and call apply_style() before plotting. Helper functions
build fields with shared colour bars and save at 300 dpi.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmocean

# ---------------------------------------------------------------------------
# Output directory (this folder)
# ---------------------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Colour maps by quantity type
# ---------------------------------------------------------------------------
CMAP_SST   = cmocean.cm.thermal     # absolute SST fields
CMAP_ERROR = cmocean.cm.amp         # |error| / MAE magnitude fields
CMAP_GRAD  = cmocean.cm.tempo       # SST gradient magnitude
CMAP_DIV   = cmocean.cm.balance     # signed fields (latent components, anomalies)

# Colour for land / missing data on maps
LAND_COLOUR = "#d9d9d9"

# Colour for overlays (crop boxes, markers) that must read on any background
HIGHLIGHT = "#00C2C7"

# ---------------------------------------------------------------------------
# Line palette — Okabe-Ito (colourblind-safe), with named roles
# ---------------------------------------------------------------------------
NAVY      = "#0B3D6B"   # PRIMARY: the main learned model
TEAL      = "#1f9e9e"   # SECONDARY: Stage 1 / bypass
ORANGE    = "#E69F00"   # reference baseline 1
GREY      = "#7A7A7A"   # reference baseline 2 / neutral
VERMILION = "#D55E00"   # contrast series
SKY       = "#56B4E9"   # extra
GREEN     = "#009E73"   # extra
BLACK     = "#1a1a1a"   # totals / sums

# ENSO year colours (physically intuitive: La Nina cool, El Nino warm)
ENSO_COLOURS = {2022: "#0072B2", 2023: "#D55E00", 2024: "#009E73"}


def ordered_colors(n, cmap=cmocean.cm.haline):
    """Return n colours sampled along a sequential map, for ordered families
    (patch size, latent dimension). Trimmed to avoid the near-white extreme."""
    return [cmap(v) for v in np.linspace(0.08, 0.88, n)]


# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------
def apply_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 13,
        "axes.titlesize": 14,      # titles are not used, but keep sane
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.8",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
    })


def grid(ax):
    """Consistent light grid for line plots."""
    ax.grid(True, which="major", color="0.85", lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Field-plotting helpers
# ---------------------------------------------------------------------------
def plot_field(ax, field, cmap, vmin=None, vmax=None, X=None, Y=None,
               geo=False):
    """Draw a 2-D field. If geo=True and X/Y (lon/lat grids) are given, plot in
    geographic coordinates; otherwise plot on the pixel grid (origin lower)."""
    if geo and X is not None and Y is not None:
        im = ax.pcolormesh(X, Y, field, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_aspect("auto")
    else:
        im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", aspect="auto", interpolation="nearest")
        ax.set_xlabel("grid column")
        ax.set_ylabel("grid row")
    return im


def cbar(fig, im, ax, label, **kw):
    cb = fig.colorbar(im, ax=ax, label=label, **kw)
    cb.ax.tick_params(labelsize=11)
    return cb


def panel_label(ax, text, loc="upper left"):
    """Small inset label replacing a per-panel title (e.g. latent dim id)."""
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=12, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7",
                      alpha=0.85, lw=0.6))


def save(fig, name):
    """Save a figure into this folder and close it. Returns the path."""
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {name}")
    return path
