"""
Patch size ablation (Version B from supervisor feedback).

Trains the v3 baseline SINN with three patch sizes:
    [6, 6, 6], [10, 10, 10], [14, 14, 14]

Larger patches are more likely to include the complex high-error region.
If larger patches improve performance in the complex region specifically,
that supports the training-dynamics explanation (the complex region is
too small relative to the full domain for the loss to care about it).

Single seed. Results saved alongside the latent-dim ablation results.

Usage:
    python run_ablation_patch_size.py
"""

import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config import (
    DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR,
    SINN_V3_MODULE, STAGE1_EPOCHS, NUM_PATCHES, ensure_dirs,
)
from utils import (
    set_seed, load_sinn_class, load_data, setup_sinn,
    save_sinn_weights, save_results,
)

random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
tf.random.set_seed(DEFAULT_SEED)

ensure_dirs()

PATCH_SIZES = [4, 6, 8, 10, 12, 14, 16]    # cubic patches: [t, y, x] all equal


def main():
    sinn_class = load_sinn_class(SINN_V3_MODULE)
    data = load_data()
    all_results = {}

    for ps in PATCH_SIZES:
        patch_dim = [ps, ps, ps]
        print(f"\n{'='*60}")
        print(f"PATCH SIZE ABLATION: patch_dim = {patch_dim}")
        print(f"{'='*60}")

        solver = setup_sinn(sinn_class, data, seed=DEFAULT_SEED)
        loss_history = solver.train(STAGE1_EPOCHS, patch_dim, NUM_PATCHES)
        print(f"  Final train loss: {loss_history['total'][-1]:.6e}")

        prefix = f"ablation_patch{ps}"
        save_sinn_weights(solver, prefix)

        test_summary = solver.evaluate_on_test_timesteps(verbose=False)
        print(f"  Test MAE: {test_summary['mae']:.4f} °C")

        all_results[ps] = {
            "patch_dim": patch_dim,
            "mean_mae":  test_summary["mae"],
            "mean_mse":  test_summary["mse"],
            "per_step_mae": [s["mae"] for s in test_summary["per_step"]],
        }

    save_results(all_results, "ablation_patch_size_results.pkl")

    # ---- Plot ----
    ps_list = sorted(all_results.keys())
    maes    = [all_results[ps]["mean_mae"] for ps in ps_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ps_list, maes, "go-", lw=2, markersize=9)
    for ps, m in zip(ps_list, maes):
        ax.annotate(f"{m:.4f} °C", (ps, m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)
    ax.set_xlabel("Patch size (cubic, n×n×n)", fontsize=12)
    ax.set_ylabel("Mean Test MAE (°C)", fontsize=12)
    ax.set_title("Patch Size Ablation — v3 Baseline SINN\n"
                 "Larger patches sample complex region more often", fontsize=12,
                 fontweight="bold")
    ax.set_xticks(ps_list)
    ax.set_xticklabels([f"{ps}×{ps}×{ps}" for ps in ps_list])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{FIGURES_DIR}/ablation_patch_size.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")

    print("\nSummary:")
    for ps, m in zip(ps_list, maes):
        print(f"  Patch {ps}×{ps}×{ps}: MAE = {m:.4f} °C")


if __name__ == "__main__":
    main()
