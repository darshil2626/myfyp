"""
Latent dimension ablation: train v3 baseline SINN with r = 1, 3, 5, 10, 15.
Single seed, saves results for each r value.

Usage:
    python run_ablation_latent_dim.py
"""

import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    ABLATION_LATENT_DIMS, DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR,
    SINN_V3_MODULE, STAGE1_EPOCHS, PATCH_DIM, NUM_PATCHES, ensure_dirs,
)
from utils import (
    set_seed, load_sinn_class, load_data, setup_sinn,
    save_sinn_weights, save_results,
)

# Global seeds — per-r seed is set again inside setup_sinn
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
tf.random.set_seed(DEFAULT_SEED)

ensure_dirs()


def main():
    sinn_class = load_sinn_class(SINN_V3_MODULE)
    data = load_data()

    all_results = {}

    for r in ABLATION_LATENT_DIMS:
        print(f"\n{'='*60}")
        print(f"ABLATION: r = {r}")
        print(f"{'='*60}")

        solver = setup_sinn(sinn_class, data, seed=DEFAULT_SEED, num_latentdim=r)

        loss_history = solver.train(STAGE1_EPOCHS, PATCH_DIM, NUM_PATCHES)
        print(f"  Final loss: {loss_history['total'][-1]:.6e}")

        prefix = f"ablation_r{r}"
        save_sinn_weights(solver, prefix)

        test_summary = solver.evaluate_on_test_timesteps(verbose=False)
        mean_mae = test_summary["mae"]
        print(f"  Test MAE: {mean_mae:.6e}")

        all_results[r] = {
            "mean_mae": mean_mae,
            "mean_mse": test_summary["mse"],
            "per_step_mae": [s["mae"] for s in test_summary["per_step"]],
        }

        # Save A matrix
        A = solver.get_latent_operator_matrix().numpy()
        np.save(f"{RESULTS_DIR}/A_matrix_ablation_r{r}.npy", A)

    # Save combined results
    save_results(all_results, "ablation_latent_dim_results.pkl")

    # Plot MAE vs r
    rs = sorted(all_results.keys())
    maes = [all_results[r]["mean_mae"] for r in rs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, maes, "bo-", lw=2, markersize=8)
    for r, m in zip(rs, maes):
        ax.annotate(f"{m:.4f}", (r, m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("Number of latent dimensions (r)", fontsize=12)
    ax.set_ylabel("Mean Test MAE", fontsize=12)
    ax.set_title("Latent Dimension Ablation (v3 baseline SINN)", fontsize=13)
    ax.set_xticks(rs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/ablation_latent_dim.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved ablation_latent_dim.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
