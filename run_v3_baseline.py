"""
Run v3 baseline SINN: train on 2022, evaluate on 2023+2024.
Saves weights and results for later analysis.

Usage:
    python run_v3_baseline.py
    python run_v3_baseline.py --load   # skip training, load saved weights
"""

import argparse
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")

from config import (
    DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR,
    NUM_LATENTDIM, SINN_V3_MODULE, STAGE1_EPOCHS,
    PATCH_DIM, NUM_PATCHES, ensure_dirs,
)
from utils import (
    set_seed, load_sinn_class, load_data, setup_sinn,
    save_sinn_weights, load_sinn_weights,
    save_results, save_config_snapshot,
)

# Global seeds — per-run seed is set again inside setup_sinn
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
tf.random.set_seed(DEFAULT_SEED)

ensure_dirs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Load saved weights instead of training")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    set_seed(args.seed)  # Explicit per-run seed control

    prefix = f"v3_seed{args.seed}"

    # Setup
    sinn_class = load_sinn_class(SINN_V3_MODULE)
    data = load_data()
    solver = setup_sinn(sinn_class, data, seed=args.seed)

    if args.load:
        load_sinn_weights(solver, prefix)
    else:
        # Train
        print(f"\n{'='*60}")
        print(f"TRAINING v3 baseline (seed={args.seed})")
        print(f"{'='*60}")
        loss_history = solver.train(STAGE1_EPOCHS, PATCH_DIM, NUM_PATCHES)
        print(f"\nTraining complete! Final loss: {loss_history['total'][-1]:.6e}")
        solver.plot_training_history(
            loss_history,
            save_path=f"{FIGURES_DIR}/training_loss_{prefix}.png",
            show=False,
        )
        save_sinn_weights(solver, prefix)
        save_results(loss_history, f"loss_history_{prefix}.pkl")

    # Evaluate on test
    print(f"\n{'='*60}")
    print(f"EVALUATING v3 baseline on test set")
    print(f"{'='*60}")
    test_summary = solver.evaluate_on_test_timesteps()
    print(f"\n  Mean MAE: {test_summary['mae']:.6e}")
    print(f"  Mean MSE: {test_summary['mse']:.6e}")

    # Save test results
    payload = {
        "test_time_indices": np.array(solver.test_time_indices, dtype=np.int32),
        "results": test_summary["per_step"],
        "mean_mae": test_summary["mae"],
        "mean_mse": test_summary["mse"],
        "seed": args.seed,
        "model": "v3_baseline",
    }
    save_results(payload, f"test_results_{prefix}.pkl")

    # Save A matrix for diagnostics
    A_final = solver.get_latent_operator_matrix().numpy()
    np.save(f"{RESULTS_DIR}/A_matrix_{prefix}.npy", A_final)
    print(f"\nA matrix eigenvalues: {np.linalg.eigvalsh(A_final)}")
    print(f"Frobenius dist from I: {np.linalg.norm(A_final - np.eye(NUM_LATENTDIM)):.4f}")

    # Plot MAE over time
    solver.plot_test_mae_over_time(
        test_summary["per_step"],
        save_path=f"{FIGURES_DIR}/test_mae_{prefix}.png",
        show=False,
    )

    # Field snapshots: early, mid, late
    test_steps = test_summary["per_step"]
    for label, idx in [("early", 0), ("mid", len(test_steps) // 2), ("late", -1)]:
        res = test_steps[idx]
        solver.plot_field_reconstruction(
            res, save_path=f"{FIGURES_DIR}/field_{prefix}_{label}_t{res['t_index']}.png",
            show=False,
        )

    save_config_snapshot(extra={"model": "v3_baseline", "seed": args.seed},
                         filename=f"config_{prefix}.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
