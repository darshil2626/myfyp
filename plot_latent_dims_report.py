"""
plot_latent_dims_report.py

Reproduce figures_for_report/latent_dims_v3_seed42_mid_t730_crop.png
with fixed colormap limits of ±0.04 so fine-grained spatial structure
in the latent dimensions is visible.

Usage:
    python plot_latent_dims_report.py
    python plot_latent_dims_report.py --vmax 0.04
    python plot_latent_dims_report.py --pkl results/test_results_v3_seed42.pkl --t 730 --vmax 0.04
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from config import (
    B_THICK, DEFAULT_SEED, DROPOUT, FIGURES_DIR, INCLUDE_T0, INCLUDE_TT,
    L2_REG, LR, N_PAST_STEPS, NUM_LAYERS, NUM_LATENTDIM, NUM_UNITS,
    RESULTS_DIR, SINN_V3_MODULE, TEST_END, TEST_START,
)
from utils import (
    apply_year_split, load_data, load_sinn_class, load_sinn_weights, setup_sinn,
)

REPORT_DIR = os.path.join(os.path.dirname(__file__), "figures_for_report")


def _load_solver(seed: int):
    data = load_data()
    sinn_class = load_sinn_class(SINN_V3_MODULE)
    solver = setup_sinn(sinn_class, data, seed=seed)
    load_sinn_weights(solver, f"v3_seed{seed}")
    return solver, solver.ny, solver.nx


def _plot_latent_only(solver, t_index: int, ny: int, nx: int,
                      vmax: float, out_path: str):
    print(f"  Reconstructing t={t_index} ...")
    res = solver.reconstruct_field_at_timestep(
        int(t_index), use_fast_solver=True, return_latent=True
    )

    latent_full = res["latent_field"]        # (ny, nx, r)
    bm = res.get("boundary_mask")
    r = latent_full.shape[2]

    aspect = nx / ny
    panel_h = 2.5
    panel_w = aspect * panel_h

    n_cols = min(r, 5)
    n_rows = int(np.ceil(r / n_cols))

    fig_w = n_cols * panel_w + 1.2
    fig_h = n_rows * panel_h + 0.5

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for k in range(r):
        row, col = divmod(k, n_cols)
        ax = axes[row][col]
        lf = latent_full[:, :, k].copy()
        if bm is not None:
            lf[bm] = np.nan

        im = ax.imshow(lf, cmap="RdBu_r", origin="lower", aspect="auto",
                       interpolation="nearest", vmin=-vmax, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("value", fontsize=6)
        ax.set_title(f"$\\ell_{k}$", fontsize=8, pad=3)
        ax.tick_params(labelsize=5)
        ax.set_xlabel("x", fontsize=6)
        ax.set_ylabel("y", fontsize=6)

    # Hide unused axes in the last row
    for k in range(r, n_rows * n_cols):
        row, col = divmod(k, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"v3  seed=42  t={t_index}  |  colormap limits ±{vmax}",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl", default=None,
                        help="Optional: path to test_results_v3_seed42.pkl "
                             "(only used if you want to verify; solver is re-loaded from weights)")
    parser.add_argument("--t", type=int,
                        default=(TEST_START + TEST_END) // 2,
                        help=f"Timestep index (default: mid = {(TEST_START + TEST_END) // 2})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used when training (default: 42)")
    parser.add_argument("--vmax", type=float, default=0.04,
                        help="Symmetric colormap limit ±vmax (default: 0.04)")
    parser.add_argument("--out", default=None,
                        help="Output path (default: figures_for_report/latent_dims_v3_seed<seed>_fixed<vmax>_t<t>.png)")
    args = parser.parse_args()

    t = args.t
    seed = args.seed
    vmax = args.vmax

    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(
            REPORT_DIR,
            f"latent_dims_v3_seed{seed}_fixed{vmax}_t{t}.png",
        )

    solver, ny, nx = _load_solver(seed)
    print(f"Domain: {ny} x {nx}  latent_dim={solver.num_latentdim}")

    _plot_latent_only(solver, t, ny, nx, vmax, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
