"""
Two-stage SINN: bypass SINN (smooth) + CNN residual correction (sharp).

Stage 1: Bypass SINN — trains the time-conditioned SINN with decoder
         skip connection. Produces smooth predictions bounded by the
         elliptic PDE's maximum principle.

Stage 2: CNN corrector — a lightweight U-Net that takes the Stage 1
         prediction, observed boundary values, and a boundary mask,
         and predicts the residual error field. Operates directly on
         the 2D spatial field with NO elliptic PDE constraint, so it
         can produce sharp, non-smooth corrections.

Final prediction: u_final = u_stage1 + u_correction

This script imports the bypass SINN class from a separate file.
If the file isn't available, set SINN_MODULE below to the correct path.
"""

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate
from keras.models import Model
import pickle
import sys
import os
import importlib.util


# =========================================================================
# Load bypass SINN class from file
# =========================================================================
def load_sinn_class(path):
    """Import the sinn class from the bypass SINN file."""
    spec = importlib.util.spec_from_file_location("v8_bypass_sinn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sinn


# =========================================================================
# Stage 2: CNN residual corrector
# =========================================================================
class ResidualCorrectorCNN:
    """
    Lightweight U-Net that predicts the residual between the true field
    and the Stage 1 SINN prediction.

    Input: (ny, nx, 3) — [stage1_pred, boundary_values, boundary_mask]
    Output: (ny, nx, 1) — correction field

    The loss is masked to interior points only (boundary is exact).
    """

    def __init__(self, ny, nx, interior_mask):
        self.ny = ny
        self.nx = nx
        self.interior_mask = interior_mask.astype(np.float32)  # (ny, nx)
        self.interior_mask_tf = tf.constant(
            self.interior_mask.reshape(1, ny, nx, 1), dtype=tf.float32
        )
        self.model = self._build_unet()
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def _build_unet(self):
        """Small U-Net: 3 encoder levels, ~80k parameters."""
        inp = Input(shape=(self.ny, self.nx, 3))

        # Encoder
        # 168x200 -> 84x100 -> 42x50 (all even, no size mismatch)
        c1 = Conv2D(16, 3, activation='relu', padding='same')(inp)
        c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
        p1 = MaxPooling2D(2)(c1)

        c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
        c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
        p2 = MaxPooling2D(2)(c2)

        # Bottleneck
        c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
        c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

        # Decoder
        u2 = UpSampling2D(2)(c3)
        u2 = Concatenate()([u2, c2])
        c4 = Conv2D(32, 3, activation='relu', padding='same')(u2)
        c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)

        u1 = UpSampling2D(2)(c4)
        u1 = Concatenate()([u1, c1])
        c5 = Conv2D(16, 3, activation='relu', padding='same')(u1)
        c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)

        # Output: single channel correction
        out = Conv2D(1, 1, activation=None, padding='same')(c5)

        model = Model(inp, out, name="residual_corrector")
        return model

    def prepare_input(self, stage1_pred, boundary_values, boundary_mask):
        """
        Stack inputs into (1, ny, nx, 3) tensor.

        stage1_pred: (ny, nx) — Stage 1 predicted field (unstandardised)
        boundary_values: (ny, nx) — observed boundary values (unstd), 0 at interior
        boundary_mask: (ny, nx) — 1 at boundary, 0 at interior
        """
        x = np.stack([stage1_pred, boundary_values, boundary_mask], axis=-1)
        return x.reshape(1, self.ny, self.nx, 3).astype(np.float32)

    @tf.function
    def _train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            pred = self.model(x_batch, training=True)
            # Masked MSE: only interior points contribute
            diff = (pred - y_batch) * self.interior_mask_tf
            n_interior = tf.reduce_sum(self.interior_mask_tf)
            loss = tf.reduce_sum(tf.square(diff)) / n_interior
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, stage1_results, U_original, epochs=50, batch_size=8):
        """
        Train on residuals from Stage 1 predictions.

        stage1_results: list of dicts from SINN reconstruct_field_at_timestep
        U_original: (nt, ny, nx) original physical field
        """
        print(f"\n{'='*60}")
        print(f"STAGE 2: Training CNN residual corrector")
        print(f"  Trainable params: {self.model.count_params():,}")
        print(f"  Training samples: {len(stage1_results)}")
        print(f"  Epochs: {epochs}")
        print(f"{'='*60}")

        # Precompute all training pairs
        X_all = []
        Y_all = []
        for res in stage1_results:
            t = res["t_index"]
            u_true = U_original[t, :, :].astype(np.float32)
            u_pred = res["u_pred"].astype(np.float32)
            residual = u_true - u_pred  # what Stage 1 missed

            bnd_mask = res["boundary_mask"].astype(np.float32)
            bnd_vals = u_true * bnd_mask  # observed values at boundary

            inp = np.stack([u_pred, bnd_vals, bnd_mask], axis=-1)
            X_all.append(inp)
            Y_all.append(residual.reshape(self.ny, self.nx, 1))

        X_all = np.array(X_all, dtype=np.float32)  # (N, ny, nx, 3)
        Y_all = np.array(Y_all, dtype=np.float32)  # (N, ny, nx, 1)

        # --- Normalise inputs and targets ---
        # Channel 0: stage1 prediction, Channel 1: boundary values, Channel 2: binary mask (skip)
        eps = 1e-8
        self.x_mean_ch0 = float(np.mean(X_all[:, :, :, 0]))
        self.x_std_ch0 = float(np.std(X_all[:, :, :, 0])) + eps
        self.x_mean_ch1 = float(np.mean(X_all[:, :, :, 1]))
        self.x_std_ch1 = float(np.std(X_all[:, :, :, 1])) + eps
        self.y_mean = float(np.mean(Y_all))
        self.y_std = float(np.std(Y_all)) + eps

        X_all[:, :, :, 0] = (X_all[:, :, :, 0] - self.x_mean_ch0) / self.x_std_ch0
        X_all[:, :, :, 1] = (X_all[:, :, :, 1] - self.x_mean_ch1) / self.x_std_ch1
        # Channel 2 (mask) stays as 0/1
        Y_all = (Y_all - self.y_mean) / self.y_std

        print(f"  Input ch0 (pred):  mean={self.x_mean_ch0:.2f}, std={self.x_std_ch0:.2f}")
        print(f"  Input ch1 (bnd):   mean={self.x_mean_ch1:.2f}, std={self.x_std_ch1:.2f}")
        print(f"  Target (residual): mean={self.y_mean:.4f}, std={self.y_std:.4f}")

        N = X_all.shape[0]
        loss_history = []

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(N)
            X_shuf = X_all[perm]
            Y_shuf = Y_all[perm]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N, batch_size):
                xb = tf.constant(X_shuf[i:i+batch_size])
                yb = tf.constant(Y_shuf[i:i+batch_size])
                loss = self._train_step(xb, yb)
                epoch_loss += float(loss.numpy())
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  epoch {epoch+1:04d} | loss {avg_loss:.6e}")

        return loss_history

    def predict_correction(self, stage1_pred, boundary_values, boundary_mask):
        """Return the correction field for a single timestep (in original scale)."""
        x = self.prepare_input(stage1_pred, boundary_values, boundary_mask)
        # Normalise using training stats
        x[0, :, :, 0] = (x[0, :, :, 0] - self.x_mean_ch0) / self.x_std_ch0
        x[0, :, :, 1] = (x[0, :, :, 1] - self.x_mean_ch1) / self.x_std_ch1
        # Channel 2 (mask) stays as 0/1
        correction_norm = self.model(tf.constant(x), training=False).numpy()
        # Denormalise output back to physical residual scale
        correction = correction_norm * self.y_std + self.y_mean
        return correction.reshape(self.ny, self.nx)


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    # ---- Config ----
    SINN_FILE = "v3_bypass_sinn.py"  # path to bypass SINN file

    # Stage 1 config
    b_thick = 1
    include_t0 = True
    include_tT = True
    num_latentdim = 10
    num_units = 128
    num_layers = 3
    dropout = 0.0
    l2_reg = 1e-5
    lr = 1e-3
    patch_dim = [10, 10, 10]
    num_patches = 100
    stage1_epochs = 20
    n_past_steps = 5
    train_fraction = 0.5

    # Stage 2 config
    stage2_epochs = 100
    stage2_batch_size = 8

    # ---- Load data ----
    data_path = r"c:\Users\darsh\Documents\fyp\myfyp\time_in_encoders_only\sst\sst_2023_2024_combined.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    X = data["X"]; Y = data["Y"]; U = data["U"]; T = data["T"]
    print(f"U shape: {np.asarray(U).shape}")

    # ---- Load SINN class ----
    sinn = load_sinn_class(SINN_FILE)

    # ==================================================================
    # STAGE 1: Train bypass SINN
    # ==================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: Training bypass SINN")
    print(f"{'='*60}")

    solver = sinn(X, Y, U, T, debug=False)
    solver.split_train_test_timesteps(mode="sequential", train_frac=train_fraction)
    solver.standardise_u(time_indices=solver.train_time_indices)
    solver.split_interior_boundary(b_thick, include_t0, include_tT)
    solver.build_models(num_latentdim, num_units, num_layers, dropout, l2_reg, lr,
                        n_past_steps=n_past_steps)

    loss_history_s1 = solver.train(stage1_epochs, patch_dim, num_patches)
    print(f"\nStage 1 complete! Final loss: {loss_history_s1['total'][-1]:.6e}")
    solver.plot_training_history(loss_history_s1, save_path="training_loss_stage1.png", show=True)

    # ---- Generate Stage 1 predictions for ALL training timesteps ----
    print("\nGenerating Stage 1 predictions for training data...")
    train_results = []
    for i, t in enumerate(solver.train_time_indices):
        res = solver.reconstruct_field_at_timestep(int(t), use_fast_solver=True)
        train_results.append(res)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(solver.train_time_indices)} done")
    print(f"  Generated {len(train_results)} Stage 1 predictions")

    train_maes = [r["mae"] for r in train_results]
    print(f"  Stage 1 train MAE: {np.mean(train_maes):.4f}")

    # ==================================================================
    # STAGE 2: Train CNN residual corrector
    # ==================================================================
    obs_mask = np.zeros((solver.ny, solver.nx), dtype=bool)
    obs_mask[:b_thick, :] = True; obs_mask[-b_thick:, :] = True
    obs_mask[:, :b_thick] = True; obs_mask[:, -b_thick:] = True
    interior_mask = ~obs_mask

    corrector = ResidualCorrectorCNN(solver.ny, solver.nx, interior_mask)
    loss_history_s2 = corrector.train(
        train_results, solver.U_original,
        epochs=stage2_epochs, batch_size=stage2_batch_size
    )

    # ==================================================================
    # EVALUATE: Combined Stage 1 + Stage 2 on test timesteps
    # ==================================================================
    print(f"\n{'='*60}")
    print("EVALUATION: Combined Stage 1 + Stage 2")
    print(f"{'='*60}")

    test_results_combined = []
    test_results_stage1 = []

    for i, t in enumerate(solver.test_time_indices):
        # Stage 1
        res1 = solver.reconstruct_field_at_timestep(int(t), use_fast_solver=True)
        test_results_stage1.append(res1)

        u_pred_s1 = res1["u_pred"]
        u_true = res1["u_true"]
        bnd_mask = res1["boundary_mask"].astype(np.float32)
        bnd_vals = u_true * bnd_mask

        # Stage 2: correction
        correction = corrector.predict_correction(u_pred_s1, bnd_vals, bnd_mask)

        # Combined prediction
        u_pred_combined = u_pred_s1 + correction
        # Keep boundary values exact
        bnd_y, bnd_x = np.where(bnd_mask > 0.5)
        u_pred_combined[bnd_y, bnd_x] = u_true[bnd_y, bnd_x]

        # Metrics (interior only)
        int_y, int_x = np.where(interior_mask)
        err = np.abs(u_pred_combined - u_true)
        int_err = err[int_y, int_x]

        result = {
            "u_pred": u_pred_combined,
            "u_pred_stage1": u_pred_s1,
            "u_true": u_true,
            "u_error": err,
            "correction": correction,
            "boundary_mask": bnd_mask > 0.5,
            "interior_mask": interior_mask,
            "mse": float(np.mean(int_err**2)),
            "mae": float(np.mean(int_err)),
            "max_error": float(np.max(int_err)),
            "t_index": int(t),
            "mae_stage1": res1["mae"],
        }
        test_results_combined.append(result)

        if (i + 1) % 10 == 0 or i == 0 or i == len(solver.test_time_indices) - 1:
            print(f"  t={t:4d} | S1 MAE {res1['mae']:.4f} | "
                  f"Combined MAE {result['mae']:.4f} | "
                  f"Improvement {(1 - result['mae']/res1['mae'])*100:.1f}%")

    # Summary
    s1_maes = [r["mae_stage1"] for r in test_results_combined]
    combined_maes = [r["mae"] for r in test_results_combined]
    combined_mses = [r["mse"] for r in test_results_combined]

    print(f"\n  Stage 1 only  — Mean MAE: {np.mean(s1_maes):.4f}")
    print(f"  Combined S1+S2 — Mean MAE: {np.mean(combined_maes):.4f}")
    print(f"  Combined S1+S2 — Mean MSE: {np.mean(combined_mses):.4f}")
    print(f"  Improvement:     {(1 - np.mean(combined_maes)/np.mean(s1_maes))*100:.1f}%")

    # ==================================================================
    # PLOTS
    # ==================================================================
    import matplotlib.pyplot as plt

    # Training loss for Stage 2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(loss_history_s2)+1), loss_history_s2, 'k-', lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Stage 2 CNN Training Loss")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_loss_stage2.png", dpi=300, bbox_inches="tight")
    plt.show()

    # MAE over time comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    t_idx = [r["t_index"] for r in test_results_combined]
    ax.plot(t_idx, s1_maes, 'r--o', markersize=3, lw=1.5, label='Stage 1 (SINN)')
    ax.plot(t_idx, combined_maes, 'b-o', markersize=3, lw=1.5, label='Stage 1 + Stage 2')
    ax.set_xlabel("Timestep"); ax.set_ylabel("MAE")
    ax.set_title("Test MAE: SINN vs SINN + CNN correction")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("test_mae_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Field reconstructions: Stage 1 vs Combined
    for label, idx in [("early", 0), ("mid", len(test_results_combined)//2), ("late", -1)]:
        res = test_results_combined[idx]
        t = res["t_index"]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        vmin = min(np.percentile(res["u_true"], 2), np.percentile(res["u_pred"], 2))
        vmax = max(np.percentile(res["u_true"], 98), np.percentile(res["u_pred"], 98))

        # Row 1: Stage 1
        axes[0, 0].imshow(res["u_true"], cmap="viridis", origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
        axes[0, 0].set_title(f"True Field (t={t})")
        axes[0, 1].imshow(res["u_pred_stage1"], cmap="viridis", origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
        axes[0, 1].set_title(f"Stage 1 (MAE {res['mae_stage1']:.3f})")
        s1_err = np.abs(res["u_pred_stage1"] - res["u_true"])
        ev = np.percentile(s1_err, 95)
        axes[0, 2].imshow(s1_err, cmap="hot", origin="lower", vmin=0, vmax=ev, aspect="auto")
        axes[0, 2].set_title("Stage 1 Error")

        # Row 2: Combined
        axes[1, 0].imshow(res["correction"], cmap="RdBu_r", origin="lower", aspect="auto",
                          vmin=-np.percentile(np.abs(res["correction"]), 95),
                          vmax=np.percentile(np.abs(res["correction"]), 95))
        axes[1, 0].set_title("CNN Correction")
        axes[1, 1].imshow(res["u_pred"], cmap="viridis", origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
        axes[1, 1].set_title(f"Combined (MAE {res['mae']:.3f})")
        combined_err = np.abs(res["u_pred"] - res["u_true"])
        axes[1, 2].imshow(combined_err, cmap="hot", origin="lower", vmin=0, vmax=ev, aspect="auto")
        axes[1, 2].set_title("Combined Error")

        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([]); ax.set_yticks([])

        plt.suptitle(f"Two-Stage Reconstruction at t={t}", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(f"twostage_{label}_t{t}.png", dpi=300, bbox_inches="tight")
        plt.show()

    # Save results
    payload = {
        "test_time_indices": np.array(solver.test_time_indices, dtype=np.int32),
        "results": test_results_combined,
        "stage1_test_mae": float(np.mean(s1_maes)),
        "combined_test_mae": float(np.mean(combined_maes)),
        "combined_test_mse": float(np.mean(combined_mses)),
    }
    with open("test_results_twostage.pkl", "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved results to test_results_twostage.pkl")