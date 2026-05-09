"""
Standardisation bias analysis.

Computes and visualises the spatial distribution shift between the
training year (2022) and test years (2023–2024). If the test mean
differs significantly from the training mean in specific regions,
the model's fixed standardisation statistics are systematically off.

Produces:
  1. Spatial bias map: test_mean - train_mean
  2. Pixel-wise std comparison: train vs test
  3. Post-hoc boundary-anchored correction and its effect on test MAE
  4. figures/standardisation_bias.png

Usage:
    python analysis_standardisation_bias.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config import (
    DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR,
    N_DAYS_2022, B_THICK, ensure_dirs,
)
from utils import load_data, load_results

ensure_dirs()


def main():
    data = load_data()
    U = np.asarray(data["U"], dtype=np.float32)
    nt, ny, nx = U.shape

    train_end = N_DAYS_2022   # day 365

    U_train = U[:train_end]           # 2022
    U_test  = U[train_end:]           # 2023+2024

    # ---- Pixel-wise statistics ----
    train_mean = np.nanmean(U_train, axis=0)   # (ny, nx)
    test_mean  = np.nanmean(U_test,  axis=0)
    train_std  = np.nanstd(U_train,  axis=0)
    test_std   = np.nanstd(U_test,   axis=0)

    bias = test_mean - train_mean              # positive = test warmer

    # Interior mask
    obs_mask = np.zeros((ny, nx), dtype=bool)
    obs_mask[:B_THICK, :] = True; obs_mask[-B_THICK:, :] = True
    obs_mask[:, :B_THICK] = True; obs_mask[:, -B_THICK:] = True
    interior = ~obs_mask

    print(f"{'='*55}")
    print("Standardisation Bias Analysis")
    print(f"{'='*55}")
    print(f"  Train mean SST:  {np.nanmean(U_train):.3f} °C")
    print(f"  Test  mean SST:  {np.nanmean(U_test):.3f} °C")
    print(f"  Global bias:     {np.nanmean(bias):.3f} °C")
    print(f"  Max |bias|:      {np.nanmax(np.abs(bias)):.3f} °C")
    print(f"  Interior bias:   {np.nanmean(bias[interior]):.3f} °C")
    print(f"  Interior |bias|: {np.nanmean(np.abs(bias[interior])):.3f} °C")

    bias_interior = bias[interior]
    std_ratio = test_std / (train_std + 1e-8)
    print(f"\n  STD ratio (test/train) — interior:")
    print(f"    mean: {np.nanmean(std_ratio[interior]):.3f}")
    print(f"    max:  {np.nanmax(std_ratio[interior]):.3f}")

    # ---- Boundary-anchored post-hoc correction ----
    # At each test timestep, shift prediction by the offset between
    # the observed boundary mean and the training-period boundary mean
    bnd_y, bnd_x = np.where(obs_mask)
    train_bnd_mean = float(np.nanmean(U_train[:, bnd_y, bnd_x]))

    try:
        test_data = load_results(f"test_results_v3_seed{DEFAULT_SEED}.pkl")
        results   = test_data["results"]
        print(f"\n  Loaded {len(results)} test results for correction experiment")

        corrected_maes = []
        original_maes  = []
        for r in results:
            t_idx = r["t_index"]
            original_maes.append(r["mae"])

            # Observed boundary mean at this timestep
            bnd_mean_t = float(np.nanmean(U[t_idx, bnd_y, bnd_x]))
            offset = bnd_mean_t - train_bnd_mean

            # Apply offset correction to interior prediction only
            u_pred_corrected = r["u_pred"].copy()
            int_y, int_x = np.where(interior)
            u_pred_corrected[int_y, int_x] += offset
            int_err = np.abs(u_pred_corrected - r["u_true"])[int_y, int_x]
            corrected_maes.append(float(np.nanmean(int_err)))

        orig_mean    = float(np.mean(original_maes))
        corr_mean    = float(np.mean(corrected_maes))
        improvement  = orig_mean - corr_mean
        print(f"\n  Boundary-anchored correction:")
        print(f"    Original MAE:  {orig_mean:.4f} °C")
        print(f"    Corrected MAE: {corr_mean:.4f} °C")
        print(f"    Improvement:   {improvement:+.4f} °C  ({improvement/orig_mean*100:+.1f}%)")
        if abs(improvement) < 0.01:
            print("    → Correction negligible: standardisation bias is not a major factor")
        elif improvement > 0:
            print("    → Correction helps: standardisation bias contributes to test error")
        else:
            print("    → Correction hurts: bias correction overshoots; bias is not uniform")
    except FileNotFoundError:
        corrected_maes = None
        original_maes  = None
        print("  [warn] test_results_v3 not found — skipping correction experiment")

    # ---- Figure ----
    n_panels = 4
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    im0 = axes[0].imshow(train_mean, cmap="viridis", origin="lower", aspect="auto")
    axes[0].set_title("Train mean SST (2022)\n(°C)", fontsize=11)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(test_mean, cmap="viridis", origin="lower", aspect="auto")
    axes[1].set_title("Test mean SST (2023–24)\n(°C)", fontsize=11)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    bias_vmax = np.nanpercentile(np.abs(bias), 98)
    im2 = axes[2].imshow(bias, cmap="RdBu_r", origin="lower",
                          vmin=-bias_vmax, vmax=bias_vmax, aspect="auto")
    axes[2].set_title(f"Bias: test − train (°C)\nmean={np.nanmean(bias):.2f} °C", fontsize=11)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, label="°C")

    sr_vmax = np.nanpercentile(std_ratio[interior], 97)
    im3 = axes[3].imshow(std_ratio, cmap="PuOr", origin="lower",
                          vmin=1/sr_vmax, vmax=sr_vmax, aspect="auto")
    axes[3].set_title(f"STD ratio (test/train)\nmean={np.nanmean(std_ratio[interior]):.2f}", fontsize=11)
    fig.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "Standardisation Bias: Distribution Shift Between 2022 (Train) and 2023–2024 (Test)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out = f"{FIGURES_DIR}/standardisation_bias.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
