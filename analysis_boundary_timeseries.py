"""
Boundary dynamics analysis — connecting boundary oscillation to model performance.

Addresses supervisor feedback:
  "Observe how the boundary value near the complex region changes over
   time and if it is oscillatory then it means there is a flux of flow
   through the boundary, hence the Laplacian should be able to model
   it somewhat."

This script answers three questions:
  1. Is the boundary SST near the complex region oscillatory? (PSD)
  2. Does the SINN capture the boundary-driven component? (boundary vs interior comparison)
  3. Does boundary variability correlate with model error? (correlation analysis)

Usage:
    python analysis_boundary_timeseries.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from config import (
    B_THICK, FIGURES_DIR, RESULTS_DIR, DEFAULT_SEED,
    N_DAYS_2022, N_DAYS_2023, N_DAYS_2024, N_TOTAL,
    TEST_START, TEST_END, ensure_dirs,
)
from utils import load_data, load_results

ensure_dirs()


def select_boundary_and_interior_pairs(ny, nx, n_points=4):
    """
    Select boundary points on the right edge (top half, near complex region)
    and their nearest interior neighbours.
    Returns: list of (bnd_y, bnd_x, int_y, int_x) tuples
    """
    bnd_x = nx - 1          # rightmost column (boundary)
    int_x = nx - 1 - B_THICK  # first interior column

    y_min = ny // 2
    y_max = ny - B_THICK - 1
    ys = np.linspace(y_min, y_max, n_points, dtype=int)

    pairs = []
    for y in ys:
        pairs.append((int(y), bnd_x, int(y), int_x))
    return pairs


def main():
    data = load_data()
    U = np.asarray(data["U"], dtype=np.float32)
    nt, ny, nx = U.shape

    # Load v3 baseline test results
    try:
        v3_data = load_results(f"test_results_v3_seed{DEFAULT_SEED}.pkl")
        v3_results = v3_data["results"]
        has_model_results = True
    except FileNotFoundError:
        print("  WARNING: v3 results not found. Plotting boundary data only.")
        has_model_results = False

    pairs = select_boundary_and_interior_pairs(ny, nx, n_points=4)
    print(f"  Boundary-interior pairs: {pairs}")

    test_indices = np.arange(TEST_START, TEST_END, dtype=np.int32)
    all_days = np.arange(nt)

    colours = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    # ================================================================
    # FIGURE 1: Boundary oscillation + PSD (answering Q1)
    # ================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios": [2, 1.5]})

    # Panel 1: Boundary SST time series (NO detrending — we WANT the seasonal signal)
    ax = axes[0]
    for i, (by, bx, iy, ix) in enumerate(pairs):
        sst = U[:, by, bx]
        ax.plot(all_days, sst, color=colours[i], lw=1.0, alpha=0.85,
                label=f"Boundary ({by}, {bx})")

    ax.axvline(N_DAYS_2022, color="black", ls="--", lw=1, alpha=0.6)
    ax.axvline(N_DAYS_2022 + N_DAYS_2023, color="black", ls="--", lw=1, alpha=0.6)
    ax.axvspan(0, N_DAYS_2022, alpha=0.04, color="green")
    ax.axvspan(N_DAYS_2022, nt, alpha=0.04, color="red")
    ax.set_ylabel("SST (°C)", fontsize=11)
    ax.set_title("Boundary SST Near Complex Region (Right Edge, Top Half)", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.text(N_DAYS_2022 // 2, ax.get_ylim()[0] + 0.3, "2022 (Train)",
            ha="center", fontsize=9, style="italic")
    ax.text(N_DAYS_2022 + N_DAYS_2023 // 2, ax.get_ylim()[0] + 0.3, "2023 (Test)",
            ha="center", fontsize=9, style="italic")
    ax.text(N_DAYS_2022 + N_DAYS_2023 + N_DAYS_2024 // 2, ax.get_ylim()[0] + 0.3,
            "2024 (Test)", ha="center", fontsize=9, style="italic")

    # Panel 2: PSD — NO detrending, seasonal signal is what we want to see
    ax = axes[1]
    for i, (by, bx, iy, ix) in enumerate(pairs):
        sst = U[:, by, bx].astype(np.float64)
        # Remove only the mean, keep the seasonal oscillation
        sst_centered = sst - np.mean(sst)
        freqs, psd = welch(sst_centered, fs=1.0, nperseg=min(512, nt))
        periods = 1.0 / freqs[1:]
        ax.semilogy(periods, psd[1:], color=colours[i], lw=1.5, alpha=0.8,
                    label=f"({by}, {bx})")

    ax.axvline(365, color="black", ls=":", lw=1.5, alpha=0.7, label="Annual (365 d)")
    ax.axvline(182, color="gray", ls=":", lw=1.2, alpha=0.6, label="Semi-annual (182 d)")
    ax.set_xlabel("Period (days)", fontsize=11)
    ax.set_ylabel("PSD", fontsize=10)
    ax.set_title("Power Spectral Density — Boundary SST (Peaks = Oscillatory Modes)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(5, 600)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/boundary_oscillation_psd.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved boundary_oscillation_psd.png")

    # ================================================================
    # FIGURE 2: Boundary SST vs interior reconstruction (answering Q2)
    # Only if model results are available
    # ================================================================
    if has_model_results:
        fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 3.5 * len(pairs)),
                                  sharex=True)

        for i, (by, bx, iy, ix) in enumerate(pairs):
            ax = axes[i]

            # True boundary SST over test period
            bnd_sst_test = U[test_indices, by, bx]

            # True and predicted interior SST at the paired interior point
            int_sst_true = np.array([r["u_true"][iy, ix] for r in v3_results])
            int_sst_pred = np.array([r["u_pred"][iy, ix] for r in v3_results])

            # Normalise to [0,1] for overlay (different SST ranges at different points)
            def norm01(x):
                return (x - x.min()) / (x.max() - x.min() + 1e-8)

            ax.plot(test_indices, norm01(bnd_sst_test), color=colours[i],
                    lw=1.0, alpha=0.6, label=f"Boundary SST ({by},{bx}) [normalised]")
            ax.plot(test_indices, norm01(int_sst_true), "k-", lw=1.2, alpha=0.8,
                    label=f"True interior ({iy},{ix})")
            ax.plot(test_indices, norm01(int_sst_pred), "k--", lw=1.2, alpha=0.5,
                    label=f"SINN prediction ({iy},{ix})")

            # Shade the error
            ax.fill_between(test_indices,
                            norm01(int_sst_pred),
                            norm01(int_sst_true),
                            alpha=0.15, color="red", label="Reconstruction error")

            ax.set_ylabel("Normalised SST", fontsize=10)
            ax.set_title(f"Boundary ({by},{bx}) → Interior ({iy},{ix}): "
                         f"Does the Laplacian propagate boundary oscillation?",
                         fontsize=10)
            ax.legend(fontsize=8, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Timestep", fontsize=11)
        plt.suptitle("Boundary-to-Interior Signal Propagation via Laplacian",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/boundary_interior_propagation.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved boundary_interior_propagation.png")

    # ================================================================
    # FIGURE 3: Boundary rate of change vs model error (answering Q3)
    # ================================================================
    if has_model_results:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 1]})

        # Compute mean boundary dSST/dt across the selected points
        bnd_rate = np.zeros(len(test_indices), dtype=np.float64)
        for by, bx, _, _ in pairs:
            sst_test = U[test_indices, by, bx].astype(np.float64)
            rate = np.abs(np.gradient(sst_test))
            bnd_rate += rate
        bnd_rate /= len(pairs)

        # Model MAE per timestep
        mae_per_step = np.array([r["mae"] for r in v3_results])

        # Smooth both for clearer trend
        def smooth(x, w=15):
            return np.convolve(x, np.ones(w)/w, mode="same")

        ax = axes[0]
        ax.plot(test_indices, smooth(bnd_rate), "b-", lw=1.5, alpha=0.8,
                label="Boundary |dSST/dt| (smoothed)")
        ax.set_ylabel("|dSST/dt| (°C/day)", fontsize=11, color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_title("Boundary Rate of Change vs Model Error", fontsize=13,
                     fontweight="bold")
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(test_indices, smooth(mae_per_step), "r-", lw=1.5, alpha=0.8,
                 label="SINN MAE (smoothed)")
        ax2.set_ylabel("MAE (°C)", fontsize=11, color="red")
        ax2.set_xlabel("Timestep", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(True, alpha=0.3)

        # Compute correlation
        corr = np.corrcoef(smooth(bnd_rate), smooth(mae_per_step))[0, 1]
        ax2.text(0.98, 0.95, f"Pearson r = {corr:.3f}",
                 transform=ax2.transAxes, fontsize=11, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="gray", alpha=0.9))

        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/boundary_rate_vs_error.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved boundary_rate_vs_error.png")

        # Print interpretation
        print(f"\n{'='*60}")
        print("BOUNDARY OSCILLATION ANALYSIS")
        print(f"{'='*60}")
        print(f"  Correlation between boundary |dSST/dt| and model MAE: r = {corr:.3f}")
        if corr > 0.3:
            print("  → POSITIVE correlation: when boundary SST changes rapidly,")
            print("    model error increases. The Laplacian captures the low-frequency")
            print("    boundary-driven component but struggles with rapid transitions.")
            print("    Supervisor's hypothesis partially confirmed: boundary oscillation")
            print("    creates flux that the Laplacian propagates, but the maximum")
            print("    principle smooths the interior response, losing sharp features.")
        elif corr < -0.3:
            print("  → NEGATIVE correlation: rapid boundary change corresponds to")
            print("    lower error. This would suggest the model benefits from strong")
            print("    boundary forcing (unusual — investigate further).")
        else:
            print("  → WEAK correlation: boundary rate of change does not strongly")
            print("    predict model error. Interior dynamics may be driven by")
            print("    processes not captured at the boundary (e.g., interior eddies).")
            print("    This suggests the complex region's difficulty stems from")
            print("    interior physics, not boundary flux.")

    # ================================================================
    # Location map
    # ================================================================
    mean_sst = U.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(mean_sst, cmap="RdYlBu_r", origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax, label="Time-averaged SST (°C)")
    for i, (by, bx, iy, ix) in enumerate(pairs):
        ax.plot(bx, by, "o", color=colours[i], markersize=10,
                markeredgecolor="white", markeredgewidth=2,
                label=f"Bnd ({by},{bx})")
        ax.plot(ix, iy, "s", color=colours[i], markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate("", xy=(ix, iy), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="->", color=colours[i], lw=1.5))
    ax.set_title("Boundary → Interior Point Pairs (Right Edge, Top Half)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/boundary_points_location.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved boundary_points_location.png")


if __name__ == "__main__":
    main()