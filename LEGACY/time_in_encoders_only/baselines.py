"""
Three baselines for SST two-stage SINN evaluation.

Baseline 1: Boundary interpolation (cubic) — no learning
Baseline 2: Persistence — use last training field for all test steps  
Baseline 3: Pure CNN (U-Net) — no SINN, no elliptic PDE

Baselines 1 & 2 run from test_results_twostage.pkl (no training needed).
Baseline 3 needs the original SST data pickle to train.

Usage:
    python baselines.py --results test_results_twostage.pkl --data sst_2023_2024_combined.pkl
    
If --data is omitted, only baselines 1 & 2 run.
"""

import numpy as np
import pickle
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import griddata


# =========================================================================
# Baseline 1: Boundary interpolation
# =========================================================================
def baseline_boundary_interpolation(results):
    """Cubic interpolation from boundary values to interior."""
    maes = []
    for r in results:
        u_true = r["u_true"]
        bm = r["boundary_mask"]
        im = r["interior_mask"]

        bnd_y, bnd_x = np.where(bm)
        bnd_vals = u_true[bnd_y, bnd_x]
        int_y, int_x = np.where(im)

        u_interp = griddata(
            np.column_stack([bnd_y, bnd_x]), bnd_vals,
            np.column_stack([int_y, int_x]), method="cubic",
        )
        nan_mask = np.isnan(u_interp)
        if nan_mask.any():
            u_nearest = griddata(
                np.column_stack([bnd_y, bnd_x]), bnd_vals,
                np.column_stack([int_y[nan_mask], int_x[nan_mask]]),
                method="nearest",
            )
            u_interp[nan_mask] = u_nearest

        mae = float(np.mean(np.abs(u_interp - u_true[int_y, int_x])))
        maes.append(mae)
    return maes


# =========================================================================
# Baseline 2: Persistence
# =========================================================================
def baseline_persistence(results):
    """Use the first test timestep's field as prediction for all."""
    u_persist = results[0]["u_true"]
    maes = []
    for r in results:
        u_true = r["u_true"]
        im = r["interior_mask"]
        int_y, int_x = np.where(im)
        mae = float(np.mean(np.abs(u_persist[int_y, int_x] - u_true[int_y, int_x])))
        maes.append(mae)
    return maes


# =========================================================================
# Baseline 3: Pure CNN (U-Net) — no SINN at all
# =========================================================================
def baseline_pure_cnn(data_path, results_pkl):
    """
    Train a U-Net directly:  (boundary_values, boundary_mask, time) -> interior field.
    
    Same architecture as the Stage 2 CNN corrector but predicting
    the full field, not a residual. Same train/test split.
    """
    import tensorflow as tf
    from keras.layers import (
        Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate,
    )
    from keras.models import Model

    # Load original data
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    U = np.asarray(data["U"], dtype=np.float32)  # (nt, ny, nx)
    nt, ny, nx = U.shape
    print(f"  Data shape: {U.shape}")

    # Load test results to get the same split and masks
    with open(results_pkl, "rb") as f:
        d = pickle.load(f)
    test_indices = d["test_time_indices"]
    bm_example = d["results"][0]["boundary_mask"]
    im_example = d["results"][0]["interior_mask"]
    
    # Infer train indices (everything before test)
    train_indices = np.arange(0, test_indices[0], dtype=np.int32)
    print(f"  Train: {len(train_indices)} steps, Test: {len(test_indices)} steps")

    # Standardise using training data only
    U_mean = np.mean(U[train_indices], axis=0)
    U_std = np.std(U[train_indices], axis=0)
    U_std = np.maximum(U_std, 1e-8)
    U_norm = (U - U_mean) / U_std

    obs_mask = bm_example.astype(np.float32)
    int_mask = im_example.astype(np.float32)
    t_max = float(nt - 1)

    # Build training arrays
    def make_inputs(indices):
        """Create (N, ny, nx, 3) input: [boundary_values, boundary_mask, time_channel]"""
        N = len(indices)
        X = np.zeros((N, ny, nx, 3), dtype=np.float32)
        Y = np.zeros((N, ny, nx, 1), dtype=np.float32)
        for i, t in enumerate(indices):
            bnd_vals = U_norm[t] * obs_mask
            t_norm = float(t) / t_max
            X[i, :, :, 0] = bnd_vals
            X[i, :, :, 1] = obs_mask
            X[i, :, :, 2] = t_norm
            Y[i, :, :, 0] = U_norm[t]
        return X, Y

    print("  Building training data...")
    X_train, Y_train = make_inputs(train_indices)
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")

    # U-Net (same size as Stage 2 corrector for fair comparison)
    inp = Input(shape=(ny, nx, 3))
    c1 = Conv2D(16, 3, activation="relu", padding="same")(inp)
    c1 = Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = MaxPooling2D(2)(c1)

    c2 = Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = MaxPooling2D(2)(c2)

    c3 = Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = Conv2D(64, 3, activation="relu", padding="same")(c3)

    u2 = UpSampling2D(2)(c3)
    u2 = Concatenate()([u2, c2])
    c4 = Conv2D(32, 3, activation="relu", padding="same")(u2)
    c4 = Conv2D(32, 3, activation="relu", padding="same")(c4)

    u1 = UpSampling2D(2)(c4)
    u1 = Concatenate()([u1, c1])
    c5 = Conv2D(16, 3, activation="relu", padding="same")(u1)
    c5 = Conv2D(16, 3, activation="relu", padding="same")(c5)

    out = Conv2D(1, 1, activation=None, padding="same")(c5)
    model = Model(inp, out)

    # Interior-only loss
    int_mask_tf = tf.constant(int_mask.reshape(1, ny, nx, 1), dtype=tf.float32)

    def interior_mse(y_true, y_pred):
        diff = (y_true - y_pred) * int_mask_tf
        return tf.reduce_sum(diff ** 2) / tf.reduce_sum(int_mask_tf)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=interior_mse)
    model.summary()

    # Train
    print("  Training pure CNN...")
    history = model.fit(
        X_train, Y_train,
        epochs=200,
        batch_size=8,
        verbose=1,
    )

    # Evaluate on test
    print("  Evaluating on test set...")
    X_test, Y_test = make_inputs(test_indices)
    preds = model.predict(X_test, batch_size=8)

    int_y, int_x = np.where(im_example)
    maes = []
    for i, t in enumerate(test_indices):
        u_pred_norm = preds[i, :, :, 0]
        # Unstandardise
        u_pred = u_pred_norm * U_std + U_mean
        u_true = U[t]
        # Force boundary exact
        bnd_y, bnd_x = np.where(bm_example)
        u_pred[bnd_y, bnd_x] = u_true[bnd_y, bnd_x]
        mae = float(np.mean(np.abs(u_pred[int_y, int_x] - u_true[int_y, int_x])))
        maes.append(mae)

    # Save training loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["loss"], "k-", lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Pure CNN Training Loss")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pure_cnn_training_loss.png", dpi=300, bbox_inches="tight")
    print(f"  Saved pure_cnn_training_loss.png")

    return maes


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Baselines for SST SINN evaluation")
    parser.add_argument("--results", type=str, default="test_results_twostage.pkl",
                        help="Path to two-stage test results pickle")
    parser.add_argument(
        "--data",
        type=str,
        default=r"c:\Users\darsh\Documents\fyp\myfyp\time_in_encoders_only\sst\sst_2023_2024_combined.pkl",
        help="Path to original SST data pickle (Baseline 3, Pure CNN)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Baseline 3 data file not found: {args.data}. "
            "Pass a valid path with --data."
        )

    # Load test results
    with open(args.results, "rb") as f:
        d = pickle.load(f)
    results = d["results"]
    test_t = d["test_time_indices"]

    s1_maes = [r["mae_stage1"] for r in results]
    combined_maes = [r["mae"] for r in results]

    # Baseline 1
    print("Running Baseline 1: Boundary interpolation...")
    interp_maes = baseline_boundary_interpolation(results)
    print(f"  Mean MAE: {np.mean(interp_maes):.4f}")

    # Baseline 2
    print("Running Baseline 2: Persistence...")
    persist_maes = baseline_persistence(results)
    print(f"  Mean MAE: {np.mean(persist_maes):.4f}")

    # Baseline 3 (always run)
    print("Running Baseline 3: Pure CNN...")
    cnn_maes = baseline_pure_cnn(args.data, args.results)
    print(f"  Mean MAE: {np.mean(cnn_maes):.4f}")

    # ---- Summary ----
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY (Test MAE, interior only)")
    print("=" * 65)
    print(f"  Boundary interpolation : {np.mean(interp_maes):.4f}")
    print(f"  Persistence (t=365)    : {np.mean(persist_maes):.4f}")
    print(f"  Pure CNN (no SINN)     : {np.mean(cnn_maes):.4f}")
    print(f"  Stage 1 SINN           : {np.mean(s1_maes):.4f}")
    print(f"  Stage 1 + Stage 2 CNN  : {np.mean(combined_maes):.4f}")

    # ---- Quartile breakdown ----
    n = len(results)
    labels_slices = [
        ("Q1 (early)", slice(0, n // 4)),
        ("Q2", slice(n // 4, n // 2)),
        ("Q3", slice(n // 2, 3 * n // 4)),
        ("Q4 (late)", slice(3 * n // 4, n)),
    ]
    print("\nPer-quartile breakdown:")
    header = f"  {'':12s} | {'Interp':>8s} | {'Persist':>8s}"
    header += f" | {'PureCNN':>8s}"
    header += f" | {'S1':>8s} | {'S1+S2':>8s}"
    print(header)
    for label, sl in labels_slices:
        row = (f"  {label:12s} | {np.mean(interp_maes[sl]):8.4f} | "
               f"{np.mean(persist_maes[sl]):8.4f}")
        row += f" | {np.mean(cnn_maes[sl]):8.4f}"
        row += f" | {np.mean(s1_maes[sl]):8.4f} | {np.mean(combined_maes[sl]):8.4f}"
        print(row)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_t, interp_maes, "g-", alpha=0.5, lw=1.5,
            label=f"Boundary Interp ({np.mean(interp_maes):.3f})")
    ax.plot(test_t, persist_maes, color="orange", alpha=0.5, lw=1.5,
            label=f"Persistence ({np.mean(persist_maes):.3f})")
    ax.plot(test_t, cnn_maes, "m-", alpha=0.7, lw=1.5,
            label=f"Pure CNN ({np.mean(cnn_maes):.3f})")
    ax.plot(test_t, s1_maes, "r--", alpha=0.7, lw=1.5,
            label=f"Stage 1 SINN ({np.mean(s1_maes):.3f})")
    ax.plot(test_t, combined_maes, "b-", lw=2,
            label=f"Stage 1 + Stage 2 ({np.mean(combined_maes):.3f})")
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("MAE (interior)", fontsize=12)
    ax.set_title("Baseline Comparison: Test MAE Over Time", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("baseline_comparison.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved baseline_comparison.png")


if __name__ == "__main__":
    main()
