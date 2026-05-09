"""
Two-stage SINN with cross-validation CNN training (honest residuals).

The standard two-stage approach trains the CNN corrector on Stage 1
residuals from the TRAINING set — residuals the SINN has already
seen. At test time, residuals are larger and differently structured,
creating a distribution mismatch.

This script fixes that by training the SINN on the first 300 days of
2022 (days 0–299), generating Stage 1 predictions on the held-out 65
days (days 300–364), and training the CNN on those unseen residuals.

The SINN is then retrained on the full 365 days before evaluation.

Usage:
    python run_twostage_cv.py
    python run_twostage_cv.py --held-out-start 300
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
    STAGE2_EPOCHS, STAGE2_BATCH_SIZE, N_DAYS_2022, ensure_dirs,
)
from utils import (
    set_seed, load_sinn_class, load_data, setup_sinn,
    save_sinn_weights, load_sinn_weights,
    save_cnn_weights, save_results, save_config_snapshot,
)
from cnn_corrector import ResidualCorrectorCNN

random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
tf.random.set_seed(DEFAULT_SEED)

ensure_dirs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--held-out-start", type=int, default=300,
                        help="Day index where held-out period starts (default: 300)")
    args = parser.parse_args()
    set_seed(args.seed)

    held_out_start = args.held_out_start
    held_out_end   = N_DAYS_2022   # 365

    prefix_s1_partial = f"twostage_cv_s1partial_seed{args.seed}"
    prefix_s1_full    = f"twostage_cv_s1full_seed{args.seed}"
    prefix_s2         = f"twostage_cv_s2_seed{args.seed}"
    prefix            = f"twostage_cv_seed{args.seed}"

    print(f"Cross-validation two-stage (seed={args.seed})")
    print(f"  SINN trained on days 0–{held_out_start-1} (partial year)")
    print(f"  CNN trained on held-out residuals from days {held_out_start}–{held_out_end-1}")
    print(f"  SINN retrained on full 2022 (days 0–{N_DAYS_2022-1}) for final evaluation")

    sinn_class = load_sinn_class(SINN_V8_MODULE)
    data       = load_data()

    # ============================================================
    # STEP 1: Train SINN on partial year (days 0–held_out_start-1)
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 1: Train SINN on partial 2022 year")
    print(f"{'='*60}")

    solver_partial = setup_sinn(sinn_class, data, seed=args.seed)
    # Override train indices to partial year only
    solver_partial.train_time_indices = np.arange(0, held_out_start, dtype=np.int32)
    # Standardise on partial-year data only
    solver_partial._is_standardised = False
    solver_partial.standardise_u(time_indices=solver_partial.train_time_indices)

    loss_partial = solver_partial.train(STAGE1_EPOCHS, PATCH_DIM, NUM_PATCHES)
    print(f"  Final train loss: {loss_partial['total'][-1]:.6e}")
    save_sinn_weights(solver_partial, prefix_s1_partial)

    # ============================================================
    # STEP 2: Generate held-out residuals (days held_out_start–364)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"STEP 2: Predict held-out days {held_out_start}–{held_out_end-1}")
    print(f"{'='*60}")

    held_out_indices = np.arange(held_out_start, held_out_end, dtype=np.int32)
    held_out_results = []
    for i, t in enumerate(held_out_indices):
        res = solver_partial.reconstruct_field_at_timestep(int(t), use_fast_solver=True)
        held_out_results.append(res)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i+1}/{len(held_out_indices)} done  MAE={res['mae']:.4f}")

    held_out_mae = float(np.mean([r["mae"] for r in held_out_results]))
    print(f"  Held-out MAE (unseen to SINN): {held_out_mae:.4f} °C")

    # ============================================================
    # STEP 3: Train CNN on held-out residuals
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 3: Train CNN on held-out residuals")
    print(f"{'='*60}")

    obs_mask = np.zeros((solver_partial.ny, solver_partial.nx), dtype=bool)
    obs_mask[:B_THICK, :] = True; obs_mask[-B_THICK:, :] = True
    obs_mask[:, :B_THICK] = True; obs_mask[:, -B_THICK:] = True
    interior_mask = ~obs_mask

    corrector_cv = ResidualCorrectorCNN(solver_partial.ny, solver_partial.nx, interior_mask)
    # Pass U_original for accessing true values during training
    loss_s2 = corrector_cv.train(
        held_out_results, solver_partial.U_original,
        epochs=STAGE2_EPOCHS, batch_size=STAGE2_BATCH_SIZE,
    )
    save_cnn_weights(corrector_cv, prefix_s2)
    save_results({"loss": loss_s2}, f"loss_history_{prefix_s2}.pkl")
    print(f"  CNN final train loss: {loss_s2[-1]:.6e}")

    # ============================================================
    # STEP 4: Retrain SINN on FULL 2022 year
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 4: Retrain SINN on full 2022 for final evaluation")
    print(f"{'='*60}")

    solver_full = setup_sinn(sinn_class, data, seed=args.seed)
    loss_full = solver_full.train(STAGE1_EPOCHS, PATCH_DIM, NUM_PATCHES)
    print(f"  Final train loss: {loss_full['total'][-1]:.6e}")
    save_sinn_weights(solver_full, prefix_s1_full)

    # ============================================================
    # STEP 5: Evaluate combined model on test set
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 5: Evaluate on test set")
    print(f"{'='*60}")

    test_results = []
    bnd_y, bnd_x = np.where(obs_mask)
    int_y, int_x = np.where(interior_mask)

    for i, t in enumerate(solver_full.test_time_indices):
        res1 = solver_full.reconstruct_field_at_timestep(int(t), use_fast_solver=True)
        u_pred_s1 = res1["u_pred"]
        u_true    = res1["u_true"]
        bnd_mask  = res1["boundary_mask"].astype(np.float32)
        bnd_vals  = u_true * bnd_mask

        correction      = corrector_cv.predict_correction(u_pred_s1, bnd_vals, bnd_mask)
        u_pred_combined = u_pred_s1 + correction
        u_pred_combined[bnd_y, bnd_x] = u_true[bnd_y, bnd_x]

        int_err = np.abs(u_pred_combined - u_true)[int_y, int_x]
        test_results.append({
            "t_index":    int(t),
            "mae":        float(np.mean(int_err)),
            "mse":        float(np.mean(int_err**2)),
            "mae_stage1": res1["mae"],
        })

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  t={t:4d} | S1={res1['mae']:.4f} | Combined={test_results[-1]['mae']:.4f}")

    s1_maes       = [r["mae_stage1"]  for r in test_results]
    combined_maes = [r["mae"]          for r in test_results]

    print(f"\n  Stage 1 (full retrain): {np.mean(s1_maes):.4f} °C")
    print(f"  CV two-stage combined:  {np.mean(combined_maes):.4f} °C")

    # Load standard twostage for comparison
    try:
        from utils import load_results
        std_data   = load_results(f"test_results_twostage_seed{DEFAULT_SEED}.pkl")
        std_combined = float(np.mean([r["mae"] for r in std_data["results"]]))
        print(f"  Standard two-stage:     {std_combined:.4f} °C")
        print(f"  CV improvement over standard: {std_combined - np.mean(combined_maes):+.4f} °C")
    except FileNotFoundError:
        pass

    payload = {
        "test_time_indices": np.array(solver_full.test_time_indices, dtype=np.int32),
        "results":           test_results,
        "stage1_test_mae":   float(np.mean(s1_maes)),
        "combined_test_mae": float(np.mean(combined_maes)),
        "held_out_mae":      held_out_mae,
        "held_out_start":    held_out_start,
        "seed":              args.seed,
        "model":             "twostage_cv",
    }
    save_results(payload, f"test_results_{prefix}.pkl")
    save_config_snapshot(extra={"model": "twostage_cv", "held_out_start": held_out_start},
                         filename=f"config_{prefix}.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
