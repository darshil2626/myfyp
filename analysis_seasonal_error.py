"""
Seasonal error analysis.

Maps test timesteps to calendar dates, plots MAE over time for all models,
overlaid with SST spatial variability. Annotates seasons.

Produces a key thesis figure for Chapter 5 and a simplified version for Section 4.2.

Usage:
    python analysis_seasonal_error.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date, timedelta

from config import (
    B_THICK, DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR,
    TEST_START, TEST_END, ensure_dirs,
)
from utils import load_results, load_data

ensure_dirs()


def timestep_to_date(t):
    """Convert global timestep index to calendar date."""
    return date(2022, 1, 1) + timedelta(days=int(t))


def timestep_to_doy_and_year(t):
    """Convert global timestep to (day_of_year, year)."""
    d = timestep_to_date(t)
    return d.timetuple().tm_yday, d.year


def main():
    data = load_data()
    U = np.asarray(data["U"], dtype=np.float32)

    # Load all available results
    results_sets = {}

    # Try loading each model's results
    for name, pkl_name in [
        ("v3_baseline", f"test_results_v3_seed{DEFAULT_SEED}.pkl"),
        ("v8_bypass", f"test_results_v8_seed{DEFAULT_SEED}.pkl"),
        ("twostage", f"test_results_twostage_seed{DEFAULT_SEED}.pkl"),
        ("baselines", f"baseline_results_seed{DEFAULT_SEED}.pkl"),
    ]:
        try:
            results_sets[name] = load_results(pkl_name)
        except FileNotFoundError:
            print(f"  Skipping {name} (not found)")

    test_indices = np.arange(TEST_START, TEST_END, dtype=np.int32)
    dates = [timestep_to_date(t) for t in test_indices]

    # Compute spatial std of true field at each test timestep
    obs_mask = np.zeros((U.shape[1], U.shape[2]), dtype=bool)
    obs_mask[:B_THICK, :] = True; obs_mask[-B_THICK:, :] = True
    obs_mask[:, :B_THICK] = True; obs_mask[:, -B_THICK:] = True
    int_y, int_x = np.where(~obs_mask)

    spatial_std = np.array([np.std(U[t, int_y, int_x]) for t in test_indices])

    # ---- Full figure (Chapter 5) ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.5]})

    ax = axes[0]

    # Plot available models
    if "baselines" in results_sets:
        bl = results_sets["baselines"]
        ax.plot(dates, bl["interp_maes"], "g-", alpha=0.4, lw=1,
                label=f"Boundary Interp ({bl['mean_interp']:.3f})")
        ax.plot(dates, bl["persist_maes"], color="orange", alpha=0.4, lw=1,
                label=f"Persistence ({bl['mean_persist']:.3f})")
        ax.plot(dates, bl["cnn_maes"], "m-", alpha=0.5, lw=1,
                label=f"Pure CNN ({bl['mean_cnn']:.3f})")

    if "v3_baseline" in results_sets:
        v3 = results_sets["v3_baseline"]
        v3_maes = [r["mae"] for r in v3["results"]]
        ax.plot(dates, v3_maes, "c--", alpha=0.6, lw=1.5,
                label=f"v3 SINN ({v3['mean_mae']:.4f})")

    if "v8_bypass" in results_sets:
        v8 = results_sets["v8_bypass"]
        v8_maes = [r["mae"] for r in v8["results"]]
        ax.plot(dates, v8_maes, "r--", alpha=0.7, lw=1.5,
                label=f"Bypass SINN ({v8['mean_mae']:.4f})")

    if "twostage" in results_sets:
        ts = results_sets["twostage"]
        s1_maes = [r["mae_stage1"] for r in ts["results"]]
        combined_maes = [r["mae"] for r in ts["results"]]
        ax.plot(dates, combined_maes, "b-", lw=2,
                label=f"Bypass + CNN ({ts['combined_test_mae']:.4f})")

    # Year boundary — call autoscale first so get_ylim() is finalised
    ax.autoscale(enable=True, axis="y", tight=False)
    ylo, yhi = ax.get_ylim()
    ax.axvline(date(2024, 1, 1), color="black", ls="--", lw=0.8, alpha=0.5)
    ax.text(date(2023, 7, 1), ylo + (yhi - ylo) * 0.93, "2023 (El Niño)",
            ha="center", fontsize=10, style="italic")
    ax.text(date(2024, 7, 1), ylo + (yhi - ylo) * 0.93, "2024 (Neutral)",
            ha="center", fontsize=10, style="italic")

    ax.set_ylabel("MAE (°C)", fontsize=12)
    ax.set_title("Test MAE Over Time — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    # Spatial variability panel
    ax2 = axes[1]
    ax2.fill_between(dates, spatial_std, alpha=0.3, color="gray")
    ax2.plot(dates, spatial_std, "k-", lw=1, alpha=0.7)
    ax2.axvline(date(2024, 1, 1), color="black", ls="--", lw=0.8, alpha=0.5)
    ax2.set_ylabel("Spatial σ (°C)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_title("Spatial SST Variability Within ROI", fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/seasonal_error_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved seasonal_error_analysis.png")

    # ---- Simplified figure for Section 4.2 (v3 baseline only, showing temporal degradation) ----
    if "v3_baseline" in results_sets:
        v3 = results_sets["v3_baseline"]
        v3_maes = [r["mae"] for r in v3["results"]]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, v3_maes, "b-", lw=1.5, alpha=0.8)
        ax.axvline(date(2024, 1, 1), color="black", ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("MAE (°C)", fontsize=12)
        ax.set_title("v3 Baseline SINN — Test MAE Shows Temporal Degradation",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/v3_temporal_degradation.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved v3_temporal_degradation.png")


if __name__ == "__main__":
    main()
