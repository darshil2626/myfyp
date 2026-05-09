"""
Convergence analysis: train v3 baseline for 150 epochs, saving
weights and evaluating test MAE at checkpoints every 25 epochs.

Produces a MAE-vs-epoch curve that shows whether the model has
converged or would benefit from more training. If MAE plateaus
by epoch 50-75 and shows no improvement with more epochs, the
failure is structural — not undertrained.

Results saved to:
    results/convergence_v3_seed{seed}.pkl
Figures saved to:
    figures/convergence_v3_seed{seed}.png

Usage:
    python run_v3_convergence.py
    python run_v3_convergence.py --seed 42 --total-epochs 150 --checkpoint-every 25
"""

import argparse
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR,
    SINN_V3_MODULE, PATCH_DIM, NUM_PATCHES, ensure_dirs,
)
from utils import (
    set_seed, load_sinn_class, load_data, setup_sinn,
    save_sinn_weights, save_results,
)

random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
tf.random.set_seed(DEFAULT_SEED)

ensure_dirs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--total-epochs", type=int, default=150)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    set_seed(args.seed)

    checkpoints = list(range(args.checkpoint_every,
                             args.total_epochs + 1,
                             args.checkpoint_every))

    print(f"Convergence run: {args.total_epochs} epochs, "
          f"checkpoints at {checkpoints} (seed={args.seed})")

    sinn_class = load_sinn_class(SINN_V3_MODULE)
    data = load_data()
    solver = setup_sinn(sinn_class, data, seed=args.seed)

    prefix = f"v3_convergence_seed{args.seed}"
    epoch_results = {}
    epochs_done = 0

    for ckpt_epoch in checkpoints:
        epochs_this_chunk = ckpt_epoch - epochs_done
        print(f"\n--- Training epochs {epochs_done+1}–{ckpt_epoch} ---")
        loss_history = solver.train(epochs_this_chunk, PATCH_DIM, NUM_PATCHES)
        epochs_done = ckpt_epoch

        # Save weights at this checkpoint
        ckpt_prefix = f"{prefix}_ckpt{ckpt_epoch}"
        save_sinn_weights(solver, ckpt_prefix)

        # Evaluate on test set (fast solver)
        print(f"  Evaluating at epoch {ckpt_epoch}...")
        summary = solver.evaluate_on_test_timesteps(verbose=False)
        epoch_results[ckpt_epoch] = {
            "mae": summary["mae"],
            "mse": summary["mse"],
            "final_train_loss": loss_history["total"][-1],
        }
        print(f"  Epoch {ckpt_epoch:3d}  test MAE={summary['mae']:.4f} °C  "
              f"train loss={loss_history['total'][-1]:.4e}")

    save_results(epoch_results, f"convergence_{prefix}.pkl")

    # ---- Plot ----
    epochs_list = sorted(epoch_results.keys())
    maes        = [epoch_results[e]["mae"] for e in epochs_list]
    losses      = [epoch_results[e]["final_train_loss"] for e in epochs_list]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(epochs_list, maes, "bo-", lw=2, markersize=8)
    for e, m in zip(epochs_list, maes):
        ax.annotate(f"{m:.4f}", (e, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Training epochs", fontsize=12)
    ax.set_ylabel("Mean test MAE (°C)", fontsize=12)
    ax.set_title("Convergence: Test MAE vs Epochs (v3 Baseline)", fontsize=13,
                 fontweight="bold")
    ax.set_xticks(epochs_list)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs_list, losses, "rs-", lw=2, markersize=8)
    ax.set_xlabel("Training epochs", fontsize=12)
    ax.set_ylabel("Final training loss", fontsize=12)
    ax.set_title("Training Loss at Each Checkpoint", fontsize=13)
    ax.set_yscale("log")
    ax.set_xticks(epochs_list)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"v3 Baseline Convergence Analysis (seed={args.seed})\n"
        f"If test MAE plateaus early → failure is structural, not undertrained",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    out = f"{FIGURES_DIR}/convergence_{prefix}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")
    print("\nSummary:")
    for e, m, l in zip(epochs_list, maes, losses):
        print(f"  Epoch {e:3d}: test MAE = {m:.4f} °C,  train loss = {l:.4e}")


if __name__ == "__main__":
    main()
