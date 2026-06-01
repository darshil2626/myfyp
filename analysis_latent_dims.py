"""
analysis_latent_dims.py

Visualise the interior latent field ell(x,y) for a given test_results_*.pkl.

For each chosen timestep (early / mid / late) one figure is saved:
  - Top row   : True SST | Prediction | |Error|
  - Lower grid: every latent dimension (RdBu_r, symmetric about 0)
  - Suptitle includes MAE and a smoothness ratio vs. the true SST.

Figure aspect ratio tracks the domain shape (nx/ny) to avoid stretching.
Figures saved to: figures/latent_dims_<prefix>_<label>_t<t>.png

Usage:
    python analysis_latent_dims.py results/test_results_v3_seed42.pkl
    python analysis_latent_dims.py results/test_results_v8_seed42.pkl
    python analysis_latent_dims.py results/test_results_small_domain_125_seed42.pkl
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

from config import (
    B_THICK, DEFAULT_SEED, DROPOUT, FIGURES_DIR, INCLUDE_T0, INCLUDE_TT,
    L2_REG, LR, N_PAST_STEPS, NUM_LAYERS, NUM_LATENTDIM, NUM_UNITS,
    RESULTS_DIR, SINN_V3_MODULE, SINN_V8_MODULE, TEST_END, TEST_START,
)
from utils import (
    apply_year_split, load_data, load_sinn_class, load_sinn_weights, setup_sinn,
)


# ---------------------------------------------------------------------------
# Solver loader — resolves module, crop region, weight prefix from pkl metadata
# ---------------------------------------------------------------------------

def _load_solver(model_name: str, seed: int):
    """
    Returns a ready-to-use solver (weights loaded) for the given model/seed.
    Also returns (ny, nx) of the domain for figure sizing.
    """
    data = load_data()

    # ---- Small domain models ----
    if model_name.startswith("small_domain_"):
        parts = model_name.split("_")
        sz    = int(parts[2])
        meta_path = os.path.join(RESULTS_DIR, f"small_domain_results_{sz}.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Crop metadata not found: {meta_path}\n"
                f"Run run_small_domain.py --crop-size {sz} first."
            )
        meta = pickle.load(open(meta_path, "rb"))
        cr   = meta["crop_region"]
        y0, y1, x0, x1 = cr["y_start"], cr["y_end"], cr["x_start"], cr["x_end"]

        U = np.asarray(data["U"], np.float32)[:, y0:y1, x0:x1]
        X = np.asarray(data["X"])[x0:x1]
        Y = np.asarray(data["Y"])[y0:y1]
        T = data["T"]
        ny, nx = y1 - y0, x1 - x0

        sinn_class = load_sinn_class(SINN_V3_MODULE)
        solver     = sinn_class(X, Y, U, T, debug=False)
        apply_year_split(solver)
        solver.standardise_u(time_indices=solver.train_time_indices)
        solver.split_interior_boundary(B_THICK, INCLUDE_T0, INCLUDE_TT)
        solver.build_models(
            num_latentdim=1,
            num_units=NUM_UNITS, num_layers=NUM_LAYERS,
            dropout=DROPOUT, l2_reg=L2_REG, lr=LR, n_past_steps=N_PAST_STEPS,
        )
        prefix = f"small_domain_{sz}_seed{seed}"
        load_sinn_weights(solver, prefix)
        return solver, ny, nx

    # ---- Full domain models ----
    module_map = {
        "v3_baseline": SINN_V3_MODULE,
        "v8_bypass":   SINN_V8_MODULE,
        "twostage":    SINN_V8_MODULE,  # stage-1 SINN is v8
    }
    # Weight prefix for the SINN stage (twostage reuses v8 stage-1 weights)
    prefix_map = {
        "v3_baseline": f"v3_seed{seed}",
        "v8_bypass":   f"v8_seed{seed}",
        "twostage":    f"v8_seed{seed}",
    }
    module = module_map.get(model_name, SINN_V3_MODULE)
    prefix = prefix_map.get(model_name, f"{model_name.replace('_baseline','').replace('_bypass','')}_seed{seed}")

    sinn_class = load_sinn_class(module)
    solver     = setup_sinn(sinn_class, data, seed=seed)
    load_sinn_weights(solver, prefix)

    ny, nx = solver.ny, solver.nx
    return solver, ny, nx


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def _make_figure(ny: int, nx: int, r: int, panel_h: float = 2.5):
    """
    Build a figure with:
      - Top section  : 3 panels (SST true / pred / error)
      - Bottom section: r panels in a grid (latent dims), max 5 per row

    Each panel's width/height equals nx/ny to match the domain aspect ratio.
    Returns fig, (ax_true, ax_pred, ax_err), list_of_latent_axes.
    """
    aspect = nx / ny
    panel_w = aspect * panel_h

    n_cols_lat  = min(r, 5)
    n_rows_lat  = int(np.ceil(r / n_cols_lat))
    panel_h_lat = panel_h * 0.75   # latent panels slightly smaller

    # Total figure dimensions
    sst_row_w   = 3 * panel_w
    lat_row_w   = n_cols_lat * panel_w
    fig_w       = max(sst_row_w, lat_row_w) + 1.0   # +1 for colorbars/margins
    fig_h       = panel_h + n_rows_lat * panel_h_lat + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Main grid: 2 rows (SST section / latent section)
    gs_main = GridSpec(
        2, 1, figure=fig,
        height_ratios=[panel_h, n_rows_lat * panel_h_lat],
        hspace=0.45,
        top=0.90, bottom=0.05, left=0.05, right=0.97,
    )

    # SST section — 3 columns
    gs_sst = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[0], wspace=0.35)
    ax_true = fig.add_subplot(gs_sst[0])
    ax_pred = fig.add_subplot(gs_sst[1])
    ax_err  = fig.add_subplot(gs_sst[2])

    # Latent section — n_cols_lat columns
    gs_lat  = GridSpecFromSubplotSpec(
        n_rows_lat, n_cols_lat,
        subplot_spec=gs_main[1], wspace=0.35, hspace=0.45,
    )
    lat_axes = []
    for i in range(r):
        row, col = divmod(i, n_cols_lat)
        lat_axes.append(fig.add_subplot(gs_lat[row, col]))
    # Hide unused cells in the last row
    for i in range(r, n_rows_lat * n_cols_lat):
        row, col = divmod(i, n_cols_lat)
        fig.add_subplot(gs_lat[row, col]).set_visible(False)

    return fig, (ax_true, ax_pred, ax_err), lat_axes


def _add_image(fig, ax, data, cmap, vmin, vmax, title, cbar_label, fontsize=8):
    im = ax.imshow(data, cmap=cmap, origin="lower", aspect="auto",
                   interpolation="nearest", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(cbar_label, fontsize=6)
    ax.set_title(title, fontsize=fontsize, pad=3)
    ax.tick_params(labelsize=5)
    ax.set_xlabel("x", fontsize=6)
    ax.set_ylabel("y", fontsize=6)


# ---------------------------------------------------------------------------
# Smoothness utility
# ---------------------------------------------------------------------------

def _mean_gradient(field_2d: np.ndarray) -> float:
    f = np.where(np.isnan(field_2d), 0.0, field_2d.astype(np.float64))
    gy, gx = np.gradient(f)
    return float(np.nanmean(np.sqrt(gx**2 + gy**2)))


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_latent_dims(solver, t_index: int, prefix: str, label: str,
                     ny: int, nx: int, data_U: np.ndarray):
    print(f"  Reconstructing {label} (t={t_index})...")
    res = solver.reconstruct_field_at_timestep(
        int(t_index), use_fast_solver=True, return_latent=True
    )

    latent_full = res["latent_field"]     # (ny, nx, r)
    u_true  = res["u_true"]
    u_pred  = res["u_pred"]
    u_error = np.abs(res["u_error"])
    bm      = res.get("boundary_mask")
    interior = ~bm if bm is not None else np.ones((ny, nx), dtype=bool)

    if bm is not None:
        u_error[bm] = np.nan

    r = latent_full.shape[2]

    # ---- Colour scales ----
    vmin_f  = float(np.nanpercentile(u_true[interior], 2))
    vmax_f  = float(np.nanpercentile(u_true[interior], 98))
    vmax_e  = float(np.nanpercentile(u_error[interior], 95))

    # ---- Smoothness stats ----
    sst_grad = _mean_gradient(data_U[t_index])
    lat_grads = [_mean_gradient(latent_full[:, :, k]) for k in range(r)]
    mean_lat_grad = float(np.mean(lat_grads))
    ratio = sst_grad / (mean_lat_grad + 1e-12)

    # ---- Build figure ----
    fig, (ax_t, ax_p, ax_e), lat_axes = _make_figure(ny, nx, r)

    _add_image(fig, ax_t, u_true,  "viridis", vmin_f, vmax_f,
               f"True SST  (t={t_index})", "SST (degC)")
    _add_image(fig, ax_p, u_pred,  "viridis", vmin_f, vmax_f,
               "Prediction", "SST (degC)")
    _add_image(fig, ax_e, u_error, "hot",     0,      vmax_e,
               f"|Error|  MAE={res['mae']:.4f}", "|error| (degC)")

    ax_e.text(0.02, 0.98,
              f"MSE: {res['mse']:.3e}\nMAE: {res['mae']:.3e}",
              transform=ax_e.transAxes, fontsize=6, va="top",
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Latent dims
    for k, ax in enumerate(lat_axes):
        lf  = latent_full[:, :, k]
        lv  = float(np.nanpercentile(np.abs(lf[interior]), 98))
        lv  = lv if lv > 1e-9 else 1.0
        _add_image(fig, ax, lf, "RdBu_r", -lv, lv,
                   f"Latent dim {k}", "value", fontsize=7)

    smooth_tag = "smoother" if ratio > 1 else "comparable"
    fig.suptitle(
        f"{prefix}  |  {label}  t={t_index}  |  MAE={res['mae']:.4f} degC\n"
        f"SST grad={sst_grad:.4f}  latent grad={mean_lat_grad:.4f}  "
        f"ratio={ratio:.1f}x  ({smooth_tag})",
        fontsize=9, fontweight="bold",
    )

    out = os.path.join(FIGURES_DIR, f"latent_dims_{prefix}_{label}_t{t_index}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return res


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pkl", help="Path to a test_results_*.pkl file")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    pkl_data   = pickle.load(open(args.pkl, "rb"))
    model_name = pkl_data.get("model", "unknown")
    seed       = pkl_data.get("seed",  DEFAULT_SEED)

    if "results" not in pkl_data or "u_true" not in pkl_data["results"][0]:
        raise ValueError(
            f"'{args.pkl}' does not contain per-timestep field data.\n"
            f"Latent visualisation requires a test_results_*.pkl with u_true/u_pred."
        )

    stem   = os.path.splitext(os.path.basename(args.pkl))[0]
    prefix = stem.replace("test_results_", "")
    print(f"Model: {model_name}  seed={seed}  prefix={prefix}")

    solver, ny, nx = _load_solver(model_name, seed)
    print(f"Domain: {ny} x {nx}  latent_dim={solver.num_latentdim}")

    # Load original (unstandardised) U for gradient comparison
    raw_data = load_data()
    U_raw    = np.asarray(raw_data["U"], np.float32)
    if U_raw.shape[1] != ny:          # cropped domain — crop U_raw too
        results = pkl_data["results"]
        ny_r, nx_r = results[0]["u_true"].shape
        # Recover crop region from small_domain meta
        parts = model_name.split("_")
        sz    = int(parts[2]) if model_name.startswith("small_domain_") else None
        if sz is not None:
            meta = pickle.load(open(
                os.path.join(RESULTS_DIR, f"small_domain_results_{sz}.pkl"), "rb"))
            cr   = meta["crop_region"]
            U_raw = U_raw[:, cr["y_start"]:cr["y_end"], cr["x_start"]:cr["x_end"]]

    for label, t in [
        ("early", TEST_START),
        ("mid",   (TEST_START + TEST_END) // 2),
        ("late",  TEST_END - 1),
    ]:
        plot_latent_dims(solver, t, prefix, label, ny, nx, U_raw)

    print("\nDone.")


if __name__ == "__main__":
    main()
