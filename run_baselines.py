"""
Run all baselines on the same train/test split.
Baselines 1 & 2 need no training; Baseline 3 (pure CNN) trains from scratch.
Baseline 4 (interp + CNN corrector) trains a CNN to fix boundary-interp errors,
mirroring the two-stage architecture but with interpolation instead of SINN
as Stage 1 — isolates the contribution of the SINN.

Usage:
    python run_baselines.py
    python run_baselines.py --twostage-pkl results/test_results_twostage_seed42.pkl
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from config import (
    DATA_PATH, B_THICK, TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR, STAGE2_EPOCHS, STAGE2_BATCH_SIZE,
    ensure_dirs,
)
from utils import set_seed, load_data, save_results
from cnn_corrector import PureCNN, ResidualCorrectorCNN

ensure_dirs()


def _interp_one_timestep(U_t, obs_mask, bnd_pts, int_pts):
    """Cubic interpolation from boundary values to interior for one timestep.
    Returns the full (ny, nx) field with boundary preserved exactly."""
    ny, nx = obs_mask.shape
    int_y, int_x = np.where(~obs_mask)
    bnd_y, bnd_x = np.where(obs_mask)

    bnd_vals = U_t[bnd_y, bnd_x]
    u_int = griddata(bnd_pts, bnd_vals, int_pts, method="cubic")

    nan_mask = np.isnan(u_int)
    if nan_mask.any():
        u_nearest = griddata(bnd_pts, bnd_vals, int_pts[nan_mask], method="nearest")
        u_int[nan_mask] = u_nearest

    out = np.zeros((ny, nx), dtype=np.float32)
    out[bnd_y, bnd_x] = bnd_vals
    out[int_y, int_x] = u_int.astype(np.float32)
    return out


def baseline_boundary_interpolation(U_original, test_indices, obs_mask):
    """Cubic interpolation from boundary values to interior.
    Returns (maes, records) — records include full u_pred / u_true for rich metrics.
    """
    int_y, int_x = np.where(~obs_mask)
    bnd_y, bnd_x = np.where(obs_mask)
    bnd_pts = np.column_stack([bnd_y, bnd_x])
    int_pts = np.column_stack([int_y, int_x])
    interior_mask = ~obs_mask

    maes = []
    records = []
    for t in test_indices:
        u_pred = _interp_one_timestep(U_original[t], obs_mask, bnd_pts, int_pts)
        u_true = U_original[t].astype(np.float32)
        err = u_pred[int_y, int_x] - u_true[int_y, int_x]
        mae = float(np.mean(np.abs(err)))
        maes.append(mae)
        records.append({
            "t_index": int(t),
            "u_pred": u_pred.astype(np.float32),
            "u_true": u_true,
            "boundary_mask": obs_mask,
            "interior_mask": interior_mask,
            "mae": mae,
            "mse": float(np.mean(err ** 2)),
            "max_error": float(np.max(np.abs(err))),
        })
    return maes, records


def baseline_persistence(U_original, train_indices, test_indices, obs_mask):
    """Use last training timestep's field as prediction for all test steps.
    Returns (maes, records) — records include full u_pred / u_true for rich metrics.
    """
    u_persist = U_original[train_indices[-1]].astype(np.float32)
    int_y, int_x = np.where(~obs_mask)
    interior_mask = ~obs_mask

    maes = []
    records = []
    for t in test_indices:
        u_true = U_original[t].astype(np.float32)
        err = u_persist[int_y, int_x] - u_true[int_y, int_x]
        mae = float(np.mean(np.abs(err)))
        maes.append(mae)
        records.append({
            "t_index": int(t),
            "u_pred": u_persist.copy(),
            "u_true": u_true,
            "boundary_mask": obs_mask,
            "interior_mask": interior_mask,
            "mae": mae,
            "mse": float(np.mean(err ** 2)),
            "max_error": float(np.max(np.abs(err))),
        })
    return maes, records


def baseline_interp_plus_cnn(U_original, train_indices, test_indices, obs_mask,
                              epochs, batch_size):
    """Boundary cubic interpolation as Stage 1, residual CNN as Stage 2.
    Mirrors run_twostage.py's pipeline but replaces SINN with interpolation.

    Returns:
        train_results, test_results — lists of per-timestep dicts with
        keys u_pred, u_true, boundary_mask, t_index, mae, mse, max_error.
    """
    ny, nx = obs_mask.shape
    bnd_y, bnd_x = np.where(obs_mask)
    int_y, int_x = np.where(~obs_mask)
    bnd_pts = np.column_stack([bnd_y, bnd_x])
    int_pts = np.column_stack([int_y, int_x])
    interior_mask = ~obs_mask

    def make_stage1_records(indices):
        records = []
        for t in indices:
            u_pred = _interp_one_timestep(U_original[t], obs_mask, bnd_pts, int_pts)
            err = u_pred[int_y, int_x] - U_original[t, int_y, int_x]
            records.append({
                "t_index": int(t),
                "u_pred": u_pred,
                "u_true": U_original[t].astype(np.float32),
                "boundary_mask": obs_mask,
                "interior_mask": interior_mask,
                "mae": float(np.mean(np.abs(err))),
                "mse": float(np.mean(err ** 2)),
                "max_error": float(np.max(np.abs(err))),
            })
        return records

    print("  Building Stage-1 (boundary interp) predictions on training set...")
    train_recs = make_stage1_records(train_indices)
    print(f"  Stage-1 train MAE: {np.mean([r['mae'] for r in train_recs]):.4f}")

    corrector = ResidualCorrectorCNN(ny, nx, interior_mask)
    corrector.train(train_recs, U_original, epochs=epochs, batch_size=batch_size)

    print("  Building Stage-1 (boundary interp) predictions on test set...")
    test_recs = make_stage1_records(test_indices)
    s1_maes = np.array([r["mae"] for r in test_recs], dtype=np.float64)

    combined_recs = []
    combined_maes = []
    for rec in test_recs:
        u_pred_s1 = rec["u_pred"]
        u_true = rec["u_true"]
        bnd_mask_f = obs_mask.astype(np.float32)
        bnd_vals = u_true * bnd_mask_f

        correction = corrector.predict_correction(u_pred_s1, bnd_vals, bnd_mask_f)
        u_pred_combined = u_pred_s1 + correction
        u_pred_combined[bnd_y, bnd_x] = u_true[bnd_y, bnd_x]

        err = u_pred_combined[int_y, int_x] - u_true[int_y, int_x]
        combined_recs.append({
            "t_index": rec["t_index"],
            "u_pred": u_pred_combined.astype(np.float32),
            "u_pred_stage1": u_pred_s1.astype(np.float32),
            "u_true": u_true.astype(np.float32),
            "boundary_mask": obs_mask,
            "interior_mask": interior_mask,
            "mae": float(np.mean(np.abs(err))),
            "mse": float(np.mean(err ** 2)),
            "max_error": float(np.max(np.abs(err))),
            "mae_stage1": rec["mae"],
        })
        combined_maes.append(combined_recs[-1]["mae"])

    return s1_maes.tolist(), combined_maes, combined_recs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-interp-cnn", action="store_true",
                        help="Skip the boundary-interp + CNN baseline (slow).")
    args = parser.parse_args()

    set_seed(args.seed)
    data = load_data()
    U = np.asarray(data["U"], dtype=np.float32)

    train_indices = np.arange(TRAIN_START, TRAIN_END, dtype=np.int32)
    test_indices = np.arange(TEST_START, TEST_END, dtype=np.int32)

    obs_mask = np.zeros((U.shape[1], U.shape[2]), dtype=bool)
    obs_mask[:B_THICK, :] = True; obs_mask[-B_THICK:, :] = True
    obs_mask[:, :B_THICK] = True; obs_mask[:, -B_THICK:] = True

    # Baseline 1: Boundary interpolation
    print("Running Baseline 1: Boundary interpolation...")
    interp_maes, interp_records = baseline_boundary_interpolation(U, test_indices, obs_mask)
    print(f"  Mean MAE: {np.mean(interp_maes):.4f}")

    # Baseline 2: Persistence
    print("Running Baseline 2: Persistence...")
    persist_maes, persist_records = baseline_persistence(U, train_indices, test_indices, obs_mask)
    print(f"  Mean MAE: {np.mean(persist_maes):.4f}")

    # Baseline 3: Pure CNN
    print("Running Baseline 3: Pure CNN...")
    ny, nx = U.shape[1], U.shape[2]
    cnn = PureCNN(ny, nx, ~obs_mask)
    cnn_maes = cnn.train_and_evaluate(
        U, obs_mask, train_indices, test_indices,
        epochs=200, batch_size=8,
    )
    print(f"  Mean MAE: {np.mean(cnn_maes):.4f}")

    # Baseline 4: Boundary-interp Stage 1 + CNN corrector Stage 2
    interp_cnn_s1_maes, interp_cnn_combined_maes, interp_cnn_recs = (
        [], [], [])
    if not args.skip_interp_cnn:
        print("Running Baseline 4: Boundary interp + CNN corrector...")
        interp_cnn_s1_maes, interp_cnn_combined_maes, interp_cnn_recs = (
            baseline_interp_plus_cnn(
                U, train_indices, test_indices, obs_mask,
                epochs=STAGE2_EPOCHS, batch_size=STAGE2_BATCH_SIZE,
            ))
        print(f"  Stage 1 (interp)  : {np.mean(interp_cnn_s1_maes):.4f}")
        print(f"  Stage 1+2 (combined): {np.mean(interp_cnn_combined_maes):.4f}")

    # Summary
    print(f"\n{'='*65}")
    print("BASELINE RESULTS SUMMARY (Test MAE, interior only, °C)")
    print(f"{'='*65}")
    print(f"  Boundary interpolation : {np.mean(interp_maes):.4f}")
    print(f"  Persistence            : {np.mean(persist_maes):.4f}")
    print(f"  Pure CNN               : {np.mean(cnn_maes):.4f}")
    if interp_cnn_combined_maes:
        print(f"  Interp + CNN corrector : {np.mean(interp_cnn_combined_maes):.4f}")

    # Save
    payload = {
        "test_time_indices": test_indices,
        # MAE arrays (kept for backward compatibility with plotting)
        "interp_maes": interp_maes,
        "persist_maes": persist_maes,
        "cnn_maes": cnn_maes,
        "mean_interp": float(np.mean(interp_maes)),
        "mean_persist": float(np.mean(persist_maes)),
        "mean_cnn": float(np.mean(cnn_maes)),
        "seed": args.seed,
        # Full per-timestep records — enables rich metrics (RMSE, R², bias, PCC)
        # in analysis_results_table.py
        "interp_results": interp_records,
        "persist_results": persist_records,
    }
    if interp_cnn_combined_maes:
        payload["interp_cnn_s1_maes"] = list(interp_cnn_s1_maes)
        payload["interp_cnn_maes"] = list(interp_cnn_combined_maes)
        payload["mean_interp_cnn_s1"] = float(np.mean(interp_cnn_s1_maes))
        payload["mean_interp_cnn"] = float(np.mean(interp_cnn_combined_maes))
        payload["interp_cnn_results"] = interp_cnn_recs
    save_results(payload, f"baseline_results_seed{args.seed}.pkl")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_indices, interp_maes, "g-", alpha=0.5, lw=1.5,
            label=f"Boundary Interp ({np.mean(interp_maes):.3f})")
    ax.plot(test_indices, persist_maes, color="orange", alpha=0.5, lw=1.5,
            label=f"Persistence ({np.mean(persist_maes):.3f})")
    ax.plot(test_indices, cnn_maes, "m-", alpha=0.7, lw=1.5,
            label=f"Pure CNN ({np.mean(cnn_maes):.3f})")
    if interp_cnn_combined_maes:
        ax.plot(test_indices, interp_cnn_combined_maes, color="#377eb8",
                lw=1.7, alpha=0.8,
                label=f"Interp + CNN ({np.mean(interp_cnn_combined_maes):.3f})")
    ax.set_xlabel("Timestep"); ax.set_ylabel("MAE (°C, interior)")
    ax.set_title("Baseline Comparison: Test MAE Over Time")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/baseline_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved baseline_comparison.png")


if __name__ == "__main__":
    main()
