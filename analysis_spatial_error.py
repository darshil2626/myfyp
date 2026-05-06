"""
Spatial error decomposition analysis.

Produces a multi-panel figure showing:
  1. Time-averaged |error| for Stage 1 SINN
  2. Time-averaged |error| for combined (Stage 1 + Stage 2)
  3. Time-averaged |∇SST| from ground truth (SST gradient magnitude)
  4. Scatter: error vs gradient magnitude per grid cell

This addresses the supervisor's feedback:
  "Investigate the model and data in a physical sense."

Usage:
    python analysis_spatial_error.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    B_THICK, DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR, ensure_dirs,
)
from utils import load_results, load_data

ensure_dirs()


def compute_gradient_magnitude(U, time_indices):
    """Compute time-averaged spatial gradient magnitude of SST."""
    grad_mag_sum = np.zeros((U.shape[1], U.shape[2]), dtype=np.float64)
    for t in time_indices:
        gy, gx = np.gradient(U[t].astype(np.float64))
        grad_mag_sum += np.sqrt(gx**2 + gy**2)
    return (grad_mag_sum / len(time_indices)).astype(np.float32)


def main():
    # Load data and results
    data = load_data()
    U = np.asarray(data["U"], dtype=np.float32)

    ts_data = load_results(f"test_results_twostage_seed{DEFAULT_SEED}.pkl")
    results = ts_data["results"]
    test_indices = ts_data["test_time_indices"]

    ny, nx = U.shape[1], U.shape[2]

    # Compute time-averaged error fields
    s1_error_sum = np.zeros((ny, nx), dtype=np.float64)
    combined_error_sum = np.zeros((ny, nx), dtype=np.float64)
    count = 0

    for r in results:
        s1_err = np.abs(r["u_pred_stage1"] - r["u_true"])
        combined_err = np.abs(r["u_pred"] - r["u_true"])
        s1_error_sum += s1_err
        combined_error_sum += combined_err
        count += 1

    s1_error_avg = (s1_error_sum / count).astype(np.float32)
    combined_error_avg = (combined_error_sum / count).astype(np.float32)

    # Compute gradient magnitude
    grad_mag = compute_gradient_magnitude(U, test_indices)

    # Interior mask
    obs_mask = np.zeros((ny, nx), dtype=bool)
    obs_mask[:B_THICK, :] = True; obs_mask[-B_THICK:, :] = True
    obs_mask[:, :B_THICK] = True; obs_mask[:, -B_THICK:] = True
    interior = ~obs_mask

    # Mask out boundary for clean plotting
    s1_error_avg[obs_mask] = np.nan
    combined_error_avg[obs_mask] = np.nan
    grad_mag[obs_mask] = np.nan

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    err_vmax = np.nanpercentile(s1_error_avg, 95)

    im0 = axes[0, 0].imshow(s1_error_avg, cmap="hot", origin="lower",
                              vmin=0, vmax=err_vmax, aspect="auto")
    axes[0, 0].set_title("Stage 1 SINN — Time-Averaged |Error|", fontsize=11)
    fig.colorbar(im0, ax=axes[0, 0], label="MAE (°C)")

    im1 = axes[0, 1].imshow(combined_error_avg, cmap="hot", origin="lower",
                              vmin=0, vmax=err_vmax, aspect="auto")
    axes[0, 1].set_title("Stage 1 + Stage 2 — Time-Averaged |Error|", fontsize=11)
    fig.colorbar(im1, ax=axes[0, 1], label="MAE (°C)")

    im2 = axes[1, 0].imshow(grad_mag, cmap="viridis", origin="lower", aspect="auto")
    axes[1, 0].set_title("Time-Averaged SST Gradient Magnitude", fontsize=11)
    fig.colorbar(im2, ax=axes[1, 0], label="|∇SST| (°C/gridcell)")

    # Scatter: error vs gradient (interior points only)
    int_y, int_x = np.where(interior)
    s1_err_flat = s1_error_avg[int_y, int_x]
    grad_flat = grad_mag[int_y, int_x]
    valid = ~(np.isnan(s1_err_flat) | np.isnan(grad_flat))

    axes[1, 1].scatter(grad_flat[valid], s1_err_flat[valid],
                        s=1, alpha=0.3, c="steelblue", rasterized=True)
    corr = np.corrcoef(grad_flat[valid], s1_err_flat[valid])[0, 1]
    axes[1, 1].set_xlabel("|∇SST| (°C/gridcell)", fontsize=11)
    axes[1, 1].set_ylabel("Stage 1 MAE (°C)", fontsize=11)
    axes[1, 1].set_title(f"Error vs Gradient (r = {corr:.3f})", fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Spatial Error Decomposition — Test Set (2023+2024)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/spatial_error_decomposition.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved spatial_error_decomposition.png")

    # Save the arrays for potential re-use
    np.savez(f"{RESULTS_DIR}/spatial_error_fields.npz",
             s1_error_avg=s1_error_avg,
             combined_error_avg=combined_error_avg,
             grad_mag=grad_mag)
    print("Saved spatial_error_fields.npz")


if __name__ == "__main__":
    main()
