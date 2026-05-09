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

    # Prefer twostage results (has both stage1 and combined predictions).
    # Fall back to v8 bypass results if twostage hasn't been run yet —
    # in that case stage1 and combined are identical (no CNN correction).
    twostage_pkl = f"test_results_twostage_seed{DEFAULT_SEED}.pkl"
    v8_pkl = f"test_results_v8_seed{DEFAULT_SEED}.pkl"

    try:
        ts_data = load_results(twostage_pkl)
        has_combined = True
        print(f"[info] Loaded twostage results ({twostage_pkl})")
    except FileNotFoundError:
        try:
            ts_data = load_results(v8_pkl)
            has_combined = False
            print(f"[info] Twostage results not found; falling back to v8 bypass ({v8_pkl})")
            print("[info] Stage 1 and combined panels will be identical until twostage is run.")
        except FileNotFoundError:
            print(f"ERROR: Neither {twostage_pkl} nor {v8_pkl} found in results/. "
                  "Run run_v8_bypass.py or run_twostage.py first.")
            return

    results = ts_data["results"]
    test_indices = ts_data["test_time_indices"]

    ny, nx = U.shape[1], U.shape[2]

    # Compute time-averaged error fields
    s1_error_sum = np.zeros((ny, nx), dtype=np.float64)
    combined_error_sum = np.zeros((ny, nx), dtype=np.float64)
    count = 0

    for r in results:
        # twostage results have u_pred_stage1; v8-only results use u_pred for both
        s1_pred = r.get("u_pred_stage1", r["u_pred"])
        s1_err = np.abs(s1_pred - r["u_true"])
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

    combined_title = ("Stage 1 + Stage 2 — Time-Averaged |Error|" if has_combined
                      else "Stage 1 Only — Time-Averaged |Error| (no CNN yet)")
    im1 = axes[0, 1].imshow(combined_error_avg, cmap="hot", origin="lower",
                              vmin=0, vmax=err_vmax, aspect="auto")
    axes[0, 1].set_title(combined_title, fontsize=11)
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

    # ---- Bonus figure: v3 baseline vs twostage side-by-side ----
    # Requires both v3 and twostage result pkls to exist
    v3_pkl       = f"test_results_v3_seed{DEFAULT_SEED}.pkl"
    ts_pkl       = f"test_results_twostage_seed{DEFAULT_SEED}.pkl"
    try:
        v3_data = load_results(v3_pkl)
        ts_data = load_results(ts_pkl)

        v3_results  = v3_data["results"]
        ts_results  = ts_data["results"]
        ts_indices  = ts_data["test_time_indices"]

        # Build index map so we align timesteps
        ts_t_map = {r["t_index"]: r for r in ts_results}

        v3_sum = np.zeros((ny, nx), dtype=np.float64)
        diff_sum = np.zeros((ny, nx), dtype=np.float64)
        n = 0
        for r_v3 in v3_results:
            t = r_v3["t_index"]
            if t not in ts_t_map:
                continue
            r_ts = ts_t_map[t]
            v3_sum  += np.abs(r_v3["u_pred"] - r_v3["u_true"])
            diff_sum += (np.abs(r_ts["u_pred"] - r_ts["u_true"]) -
                         np.abs(r_v3["u_pred"] - r_v3["u_true"]))
            n += 1

        if n > 0:
            v3_avg   = (v3_sum  / n).astype(np.float32)
            diff_avg = (diff_sum / n).astype(np.float32)
            v3_avg[obs_mask]   = np.nan
            diff_avg[obs_mask] = np.nan

            fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
            ev = np.nanpercentile(np.maximum(v3_avg, combined_error_avg), 95)
            dv = np.nanpercentile(np.abs(diff_avg), 95)

            im = axes2[0].imshow(v3_avg, cmap="hot", origin="lower",
                                  vmin=0, vmax=ev, aspect="auto")
            axes2[0].set_title("v3 Baseline — Time-Avg |Error|", fontsize=12)
            fig2.colorbar(im, ax=axes2[0], label="MAE (°C)", fraction=0.046)

            im = axes2[1].imshow(combined_error_avg, cmap="hot", origin="lower",
                                  vmin=0, vmax=ev, aspect="auto")
            axes2[1].set_title("Two-Stage — Time-Avg |Error|", fontsize=12)
            fig2.colorbar(im, ax=axes2[1], label="MAE (°C)", fraction=0.046)

            im = axes2[2].imshow(diff_avg, cmap="RdBu_r", origin="lower",
                                  vmin=-dv, vmax=dv, aspect="auto")
            axes2[2].set_title("Difference (two-stage − baseline)\nBlue = improvement",
                               fontsize=12)
            fig2.colorbar(im, ax=axes2[2], label="ΔMAE (°C)", fraction=0.046)

            for ax in axes2:
                ax.axis("off")

            fig2.suptitle(
                f"Spatial Error: v3 Baseline vs Two-Stage ({n} test timesteps)\n"
                f"Baseline MAE: {np.nanmean(v3_avg[~obs_mask]):.4f} °C  |  "
                f"Two-Stage MAE: {np.nanmean(combined_error_avg[~obs_mask]):.4f} °C",
                fontsize=13, fontweight="bold"
            )
            plt.tight_layout()
            out2 = f"{FIGURES_DIR}/spatial_error_v3_vs_twostage.png"
            plt.savefig(out2, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved {out2}")

    except FileNotFoundError as e:
        print(f"[info] Skipping v3 vs twostage comparison: {e}")


if __name__ == "__main__":
    main()
