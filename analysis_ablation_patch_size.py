"""
Patch size ablation — per-step MAE over time.

Plots MAE at each test timestep for every patch size evaluated in the
ablation, overlaid on a single axis so degradation profiles can be compared.

Usage:
    python analysis_ablation_patch_size.py
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES_DIR, RESULTS_DIR, ensure_dirs

ensure_dirs()


def main():
    pkl_path = f"{RESULTS_DIR}/ablation_patch_size_results.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    patch_sizes = sorted(data.keys())
    n_steps = len(data[patch_sizes[0]]["per_step_mae"])
    time_steps = np.arange(n_steps)

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(patch_sizes) - 1)) for i in range(len(patch_sizes))]

    fig, ax = plt.subplots(figsize=(14, 6))

    for ps, color in zip(patch_sizes, colors):
        per_step = np.array(data[ps]["per_step_mae"])
        mean_mae = data[ps]["mean_mae"]
        label = f"{ps}×{ps}×{ps}  (mean={mean_mae:.4f} °C)"
        ax.plot(time_steps, per_step, lw=1.2, alpha=0.8, color=color, label=label)

    ax.set_xlabel("Test Time Step", fontsize=12)
    ax.set_ylabel("MAE (°C)", fontsize=12)
    ax.set_title(
        "Patch Size Ablation — Per-Step Test MAE\nv3 Baseline SINN, cubic patches n×n×n",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", title="Patch size", title_fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{FIGURES_DIR}/ablation_patch_size_per_step_mae.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
