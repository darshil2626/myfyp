"""
analysis_small_domain_field.py

Generates early / mid / late field reconstruction plots for the small-domain
experiment, matching the naming convention used for the v3 and v8 runs:

    figures/field_small_domain_<sz>_seed<s>_early_t<t>.png
    figures/field_small_domain_<sz>_seed<s>_mid_t<t>.png
    figures/field_small_domain_<sz>_seed<s>_late_t<t>.png

The figure aspect ratio matches the spatial domain so there is no stretching.
Wide domains (nx/ny > 2.5) are stacked vertically; others are side-by-side.

Usage:
    python analysis_small_domain_field.py              # runs both 80 and 125
    python analysis_small_domain_field.py --size 125   # just the 125 crop
    python analysis_small_domain_field.py --size 80    # just the 80 crop
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    B_THICK, DROPOUT, FIGURES_DIR, INCLUDE_T0, INCLUDE_TT,
    L2_REG, LR, N_PAST_STEPS, NUM_LAYERS, NUM_UNITS,
    RESULTS_DIR, SINN_V3_MODULE, TEST_END, TEST_START,
)
from utils import apply_year_split, load_data, load_sinn_class, load_sinn_weights

ensure_dirs = lambda: os.makedirs(FIGURES_DIR, exist_ok=True)

_VERTICAL_THRESHOLD = 2.5
_MAX_VERTICAL_W = 10.0


def _build_horizontal(ny, nx, panel_h=4.0):
    aspect    = nx / ny
    panel_w   = aspect * panel_h
    cbar_w    = max(0.15, panel_w * 0.05)
    pad       = max(0.06, panel_w * 0.02)
    gap       = max(0.25, panel_w * 0.07)
    ml, mr, mb, mt = 0.50, 0.15, 0.55, 0.70

    n       = 3
    total_w = ml + n * panel_w + n * (pad + cbar_w) + (n - 1) * gap + mr
    total_h = mb + panel_h + mt

    fig = plt.figure(figsize=(total_w, total_h))
    x2f = lambda x: x / total_w
    y2f = lambda y: y / total_h

    bottom, height = y2f(mb), y2f(panel_h)
    cursor = ml
    data_axes, cbar_axes = [], []
    for _ in range(n):
        data_axes.append(fig.add_axes([x2f(cursor), bottom, x2f(panel_w), height]))
        cursor += panel_w + pad
        cbar_axes.append(fig.add_axes([x2f(cursor), bottom, x2f(cbar_w), height]))
        cursor += cbar_w + gap
    return fig, tuple(data_axes), tuple(cbar_axes)


def _build_vertical(ny, nx, max_w=_MAX_VERTICAL_W):
    aspect  = nx / ny
    cbar_w  = 0.20
    pad     = 0.08
    gap_h   = 0.40
    ml, mr, mb, mt = 0.50, 0.15, 0.55, 0.70

    panel_w = max_w - ml - mr - pad - cbar_w
    panel_h = panel_w / aspect

    n       = 3
    total_w = max_w
    total_h = mt + n * panel_h + (n - 1) * gap_h + mb

    fig = plt.figure(figsize=(total_w, total_h))
    x2f = lambda x: x / total_w
    y2f = lambda y: y / total_h

    data_axes, cbar_axes = [], []
    for i in range(n):
        bottom = y2f(mb + (n - 1 - i) * (panel_h + gap_h))
        height = y2f(panel_h)
        data_axes.append(fig.add_axes([x2f(ml), bottom, x2f(panel_w), height]))
        cbar_axes.append(fig.add_axes([x2f(ml + panel_w + pad), bottom, x2f(cbar_w), height]))
    return fig, tuple(data_axes), tuple(cbar_axes)


def build_figure(ny, nx, panel_h=4.0):
    if nx / ny > _VERTICAL_THRESHOLD:
        return _build_vertical(ny, nx)
    return _build_horizontal(ny, nx, panel_h)


def plot_snapshot(res, prefix, label, ny, nx):
    """Render one early/mid/late snapshot and save it."""
    u_true  = res["u_true"]
    u_pred  = res["u_pred"]
    abs_err = np.abs(np.nan_to_num(res["u_error"], nan=0.0))
    bm      = res.get("boundary_mask")
    if bm is not None:
        abs_err[bm] = np.nan
    t       = res["t_index"]

    vmin = min(np.percentile(u_true, 2), np.percentile(u_pred, 2))
    vmax = max(np.percentile(u_true, 98), np.percentile(u_pred, 98))
    ev   = np.nanpercentile(abs_err, 95)

    fig, (ax_t, ax_p, ax_e), (cax_t, cax_p, cax_e) = build_figure(ny, nx)

    im_kw = dict(origin="lower", aspect="auto", interpolation="nearest")

    im_t = ax_t.imshow(u_true,   cmap="viridis", vmin=vmin, vmax=vmax, **im_kw)
    im_p = ax_p.imshow(u_pred,   cmap="viridis", vmin=vmin, vmax=vmax, **im_kw)
    im_e = ax_e.imshow(abs_err,  cmap="hot",     vmin=0,    vmax=ev,   **im_kw)

    for im, cax, lbl in [(im_t, cax_t, "SST (°C)"),
                          (im_p, cax_p, "SST (°C)"),
                          (im_e, cax_e, "|Error| (°C)")]:
        cb = fig.colorbar(im, cax=cax, label=lbl)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(lbl, fontsize=8)

    ax_t.set_title(f"True Field (t={t})",  fontsize=9, pad=4)
    ax_p.set_title("Predicted Field",       fontsize=9, pad=4)
    ax_e.set_title("Absolute Error",        fontsize=9, pad=4)

    ax_e.text(
        0.02, 0.98,
        f"MSE: {res['mse']:.4e}\nMAE: {res['mae']:.4e}\nMax: {res['max_error']:.4e}",
        transform=ax_e.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    for ax in (ax_t, ax_p, ax_e):
        ax.set_xlabel("x (col)", fontsize=7)
        ax.set_ylabel("y (row)", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.text(0.5, 0.98, f"Field Reconstruction at t={t}  [{prefix}  {label}]",
             ha="center", va="top", fontsize=10, fontweight="bold")

    save_path = os.path.join(FIGURES_DIR, f"field_{prefix}_{label}_t{t}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def run_for_size(crop_sz: int, seed: int):
    prefix   = f"small_domain_{crop_sz}_seed{seed}"
    pkl_path = os.path.join(RESULTS_DIR, f"small_domain_results_{crop_sz}.pkl")

    if not os.path.exists(pkl_path):
        print(f"[skip] {pkl_path} not found.")
        return

    meta = pickle.load(open(pkl_path, "rb"))
    cr   = meta["crop_region"]
    y0, y1, x0, x1 = cr["y_start"], cr["y_end"], cr["x_start"], cr["x_end"]
    ny, nx = y1 - y0, x1 - x0
    print(f"\n=== {prefix}  crop y=[{y0}:{y1}] x=[{x0}:{x1}]  ({ny}x{nx}) ===")

    # ---- Load and crop data ----
    data = load_data()
    U = np.asarray(data["U"], np.float32)[:, y0:y1, x0:x1]
    X = np.asarray(data["X"])[x0:x1]
    Y = np.asarray(data["Y"])[y0:y1]
    T = data["T"]

    # ---- Rebuild solver ----
    sinn_class = load_sinn_class(SINN_V3_MODULE)
    solver     = sinn_class(X, Y, U, T, debug=False)
    apply_year_split(solver)
    solver.standardise_u(time_indices=solver.train_time_indices)
    solver.split_interior_boundary(B_THICK, INCLUDE_T0, INCLUDE_TT)
    solver.build_models(
        num_latentdim=1,
        num_units=NUM_UNITS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        l2_reg=L2_REG,
        lr=LR,
        n_past_steps=N_PAST_STEPS,
    )
    load_sinn_weights(solver, prefix)

    # ---- Reconstruct early / mid / late ----
    t_early = TEST_START
    t_mid   = (TEST_START + TEST_END) // 2
    t_late  = TEST_END - 1

    for label, t in [("early", t_early), ("mid", t_mid), ("late", t_late)]:
        print(f"  Reconstructing {label} (t={t})...")
        res = solver.reconstruct_field_at_timestep(int(t), use_fast_solver=True)
        plot_snapshot(res, prefix, label, ny, nx)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=None,
                        help="Crop size to process (80 or 125). Omit to run both.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()

    sizes = [args.size] if args.size else [80, 125]
    for sz in sizes:
        run_for_size(sz, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
