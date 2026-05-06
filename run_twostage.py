"""
Run two-stage SINN: bypass SINN (Stage 1) + CNN corrector (Stage 2).
Train on 2022, evaluate on 2023+2024. Saves weights for both stages.

Usage:
    python run_twostage.py
    python run_twostage.py --load           # load both stages from saved weights
    python run_twostage.py --load-stage1    # load Stage 1, train Stage 2
"""

import argparse
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    B_THICK, DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR,
    SINN_V8_MODULE, STAGE1_EPOCHS, PATCH_DIM, NUM_PATCHES,
    STAGE2_EPOCHS, STAGE2_BATCH_SIZE, ensure_dirs,
)
from utils import (
    set_seed, load_sinn_class, load_data, setup_sinn,
    save_sinn_weights, load_sinn_weights,
    save_cnn_weights, load_cnn_weights,
    save_results, save_config_snapshot,
)
from cnn_corrector import ResidualCorrectorCNN

# Global seeds — per-run seed is set again inside setup_sinn
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
tf.random.set_seed(DEFAULT_SEED)

ensure_dirs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Load both stages")
    parser.add_argument("--load-stage1", action="store_true", help="Load Stage 1 only, train Stage 2")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    set_seed(args.seed)  # Explicit per-run seed control

    prefix_s1 = f"twostage_s1_seed{args.seed}"
    prefix_s2 = f"twostage_s2_seed{args.seed}"
    prefix = f"twostage_seed{args.seed}"

    # ==================================================================
    # STAGE 1: Bypass SINN
    # ==================================================================
    sinn_class = load_sinn_class(SINN_V8_MODULE)
    data = load_data()
    solver = setup_sinn(sinn_class, data, seed=args.seed)

    if args.load or args.load_stage1:
        load_sinn_weights(solver, prefix_s1)
    else:
        print(f"\n{'='*60}")
        print(f"STAGE 1: Training bypass SINN (seed={args.seed})")
        print(f"{'='*60}")
        loss_history_s1 = solver.train(STAGE1_EPOCHS, PATCH_DIM, NUM_PATCHES)
        print(f"\nStage 1 complete! Final loss: {loss_history_s1['total'][-1]:.6e}")
        solver.plot_training_history(
            loss_history_s1,
            save_path=f"{FIGURES_DIR}/training_loss_{prefix_s1}.png",
            show=False,
        )
        save_sinn_weights(solver, prefix_s1)
        save_results(loss_history_s1, f"loss_history_{prefix_s1}.pkl")

    # Generate Stage 1 predictions for TRAINING data (needed by CNN)
    print("\nGenerating Stage 1 predictions for training data...")
    train_results = []
    for i, t in enumerate(solver.train_time_indices):
        res = solver.reconstruct_field_at_timestep(int(t), use_fast_solver=True)
        train_results.append(res)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(solver.train_time_indices)} done")
    print(f"  Stage 1 train MAE: {np.mean([r['mae'] for r in train_results]):.4f}")

    # ==================================================================
    # STAGE 2: CNN corrector
    # ==================================================================
    obs_mask = np.zeros((solver.ny, solver.nx), dtype=bool)
    obs_mask[:B_THICK, :] = True; obs_mask[-B_THICK:, :] = True
    obs_mask[:, :B_THICK] = True; obs_mask[:, -B_THICK:] = True
    interior_mask = ~obs_mask

    corrector = ResidualCorrectorCNN(solver.ny, solver.nx, interior_mask)

    if args.load:
        load_cnn_weights(corrector, prefix_s2)
    else:
        loss_history_s2 = corrector.train(
            train_results, solver.U_original,
            epochs=STAGE2_EPOCHS, batch_size=STAGE2_BATCH_SIZE,
        )
        save_cnn_weights(corrector, prefix_s2)
        save_results({"loss": loss_history_s2}, f"loss_history_{prefix_s2}.pkl")

        # Plot Stage 2 loss
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(loss_history_s2) + 1), loss_history_s2, "k-", lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Stage 2 CNN Training Loss")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/training_loss_{prefix_s2}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ==================================================================
    # EVALUATE: Combined on test set
    # ==================================================================
    print(f"\n{'='*60}")
    print("EVALUATION: Combined Stage 1 + Stage 2")
    print(f"{'='*60}")

    test_results = []
    for i, t in enumerate(solver.test_time_indices):
        res1 = solver.reconstruct_field_at_timestep(int(t), use_fast_solver=True)

        u_pred_s1 = res1["u_pred"]
        u_true = res1["u_true"]
        bnd_mask = res1["boundary_mask"].astype(np.float32)
        bnd_vals = u_true * bnd_mask

        correction = corrector.predict_correction(u_pred_s1, bnd_vals, bnd_mask)
        u_pred_combined = u_pred_s1 + correction

        # Keep boundary exact
        bnd_y, bnd_x = np.where(bnd_mask > 0.5)
        u_pred_combined[bnd_y, bnd_x] = u_true[bnd_y, bnd_x]

        int_y, int_x = np.where(interior_mask)
        int_err = np.abs(u_pred_combined - u_true)[int_y, int_x]

        result = {
            "u_pred": u_pred_combined,
            "u_pred_stage1": u_pred_s1,
            "u_true": u_true,
            "u_error": np.abs(u_pred_combined - u_true),
            "correction": correction,
            "boundary_mask": bnd_mask > 0.5,
            "interior_mask": interior_mask,
            "mse": float(np.mean(int_err ** 2)),
            "mae": float(np.mean(int_err)),
            "max_error": float(np.max(int_err)),
            "t_index": int(t),
            "mae_stage1": res1["mae"],
        }
        test_results.append(result)

        if (i + 1) % 50 == 0 or i == 0 or i == len(solver.test_time_indices) - 1:
            imp = (1 - result["mae"] / res1["mae"]) * 100
            print(f"  t={t:4d} | S1 MAE {res1['mae']:.4f} | "
                  f"Combined MAE {result['mae']:.4f} | Improvement {imp:.1f}%")

    s1_maes = [r["mae_stage1"] for r in test_results]
    combined_maes = [r["mae"] for r in test_results]

    print(f"\n  Stage 1 only  — Mean MAE: {np.mean(s1_maes):.4f}")
    print(f"  Combined S1+S2 — Mean MAE: {np.mean(combined_maes):.4f}")
    print(f"  Improvement:     {(1 - np.mean(combined_maes)/np.mean(s1_maes))*100:.1f}%")

    # Save
    payload = {
        "test_time_indices": np.array(solver.test_time_indices, dtype=np.int32),
        "results": test_results,
        "stage1_test_mae": float(np.mean(s1_maes)),
        "combined_test_mae": float(np.mean(combined_maes)),
        "seed": args.seed,
        "model": "twostage",
    }
    save_results(payload, f"test_results_{prefix}.pkl")

    # A matrix
    A_final = solver.get_latent_operator_matrix().numpy()
    np.save(f"{RESULTS_DIR}/A_matrix_{prefix}.npy", A_final)

    # MAE comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))
    t_idx = [r["t_index"] for r in test_results]
    ax.plot(t_idx, s1_maes, "r--", lw=1.5, alpha=0.7, label=f"Stage 1 ({np.mean(s1_maes):.3f})")
    ax.plot(t_idx, combined_maes, "b-", lw=2, label=f"Combined ({np.mean(combined_maes):.3f})")
    ax.set_xlabel("Timestep"); ax.set_ylabel("MAE")
    ax.set_title("Test MAE: SINN vs SINN + CNN"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/test_mae_comparison_{prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    save_config_snapshot(extra={"model": "twostage", "seed": args.seed},
                         filename=f"config_{prefix}.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
