"""
Stage 2 CNN residual corrector.

Lightweight U-Net that predicts the residual between the true field
and the Stage 1 SINN prediction.

Input:  (ny, nx, 3) — [stage1_pred, boundary_values, boundary_mask]
Output: (ny, nx, 1) — correction field

Extracted as a standalone module so it can be imported by run_twostage.py,
run_baselines.py (pure CNN), and run_multiseed.py without duplication.
"""

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate
from keras.models import Model


class ResidualCorrectorCNN:
    """
    Lightweight U-Net that predicts the residual between the true field
    and the Stage 1 SINN prediction.
    """

    def __init__(self, ny, nx, interior_mask):
        self.ny = ny
        self.nx = nx
        self.interior_mask = interior_mask.astype(np.float32)
        self.interior_mask_tf = tf.constant(
            self.interior_mask.reshape(1, ny, nx, 1), dtype=tf.float32
        )
        self.model = self._build_unet()
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        # Normalisation stats (set during training)
        self.x_mean_ch0 = 0.0
        self.x_std_ch0 = 1.0
        self.x_mean_ch1 = 0.0
        self.x_std_ch1 = 1.0
        self.y_mean = 0.0
        self.y_std = 1.0

    def _build_unet(self):
        """Small U-Net: 3 encoder levels, ~80k parameters."""
        inp = Input(shape=(self.ny, self.nx, 3))

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
        return Model(inp, out, name="residual_corrector")

    @tf.function
    def _train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            pred = self.model(x_batch, training=True)
            diff = (pred - y_batch) * self.interior_mask_tf
            n_interior = tf.reduce_sum(self.interior_mask_tf)
            loss = tf.reduce_sum(tf.square(diff)) / n_interior
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, stage1_results, U_original, epochs=200, batch_size=8):
        """
        Train on residuals from Stage 1 predictions.

        Args:
            stage1_results: list of dicts from SINN reconstruct_field_at_timestep
            U_original:     (nt, ny, nx) original physical field
            epochs:         number of epochs
            batch_size:     mini-batch size
        
        Returns:
            loss_history: list of per-epoch average loss values
        """
        print(f"\n{'='*60}")
        print(f"STAGE 2: Training CNN residual corrector")
        print(f"  Trainable params: {self.model.count_params():,}")
        print(f"  Training samples: {len(stage1_results)}")
        print(f"  Epochs: {epochs}")
        print(f"{'='*60}")

        # Precompute all training pairs
        X_all, Y_all = [], []
        for res in stage1_results:
            t = res["t_index"]
            u_true = U_original[t, :, :].astype(np.float32)
            u_pred = res["u_pred"].astype(np.float32)
            residual = u_true - u_pred

            bnd_mask = res["boundary_mask"].astype(np.float32)
            bnd_vals = u_true * bnd_mask

            inp = np.stack([u_pred, bnd_vals, bnd_mask], axis=-1)
            X_all.append(inp)
            Y_all.append(residual.reshape(self.ny, self.nx, 1))

        X_all = np.array(X_all, dtype=np.float32)
        Y_all = np.array(Y_all, dtype=np.float32)

        # Normalise
        eps = 1e-8
        self.x_mean_ch0 = float(np.mean(X_all[:, :, :, 0]))
        self.x_std_ch0 = float(np.std(X_all[:, :, :, 0])) + eps
        self.x_mean_ch1 = float(np.mean(X_all[:, :, :, 1]))
        self.x_std_ch1 = float(np.std(X_all[:, :, :, 1])) + eps
        self.y_mean = float(np.mean(Y_all))
        self.y_std = float(np.std(Y_all)) + eps

        X_all[:, :, :, 0] = (X_all[:, :, :, 0] - self.x_mean_ch0) / self.x_std_ch0
        X_all[:, :, :, 1] = (X_all[:, :, :, 1] - self.x_mean_ch1) / self.x_std_ch1
        Y_all = (Y_all - self.y_mean) / self.y_std

        print(f"  Input ch0 (pred):  mean={self.x_mean_ch0:.2f}, std={self.x_std_ch0:.2f}")
        print(f"  Input ch1 (bnd):   mean={self.x_mean_ch1:.2f}, std={self.x_std_ch1:.2f}")
        print(f"  Target (residual): mean={self.y_mean:.4f}, std={self.y_std:.4f}")

        N = X_all.shape[0]
        loss_history = []

        for epoch in range(epochs):
            perm = np.random.permutation(N)
            X_shuf = X_all[perm]
            Y_shuf = Y_all[perm]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N, batch_size):
                xb = tf.constant(X_shuf[i:i + batch_size])
                yb = tf.constant(Y_shuf[i:i + batch_size])
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
        x = np.stack([stage1_pred, boundary_values, boundary_mask], axis=-1)
        x = x.reshape(1, self.ny, self.nx, 3).astype(np.float32)

        x[0, :, :, 0] = (x[0, :, :, 0] - self.x_mean_ch0) / self.x_std_ch0
        x[0, :, :, 1] = (x[0, :, :, 1] - self.x_mean_ch1) / self.x_std_ch1

        correction_norm = self.model(tf.constant(x), training=False).numpy()
        correction = correction_norm * self.y_std + self.y_mean
        return correction.reshape(self.ny, self.nx)


class PureCNN:
    """
    Pure CNN baseline: same U-Net architecture but predicts the full field
    from boundary data, not a residual. No SINN involved.

    Input: (ny, nx, 3) — [boundary_values, boundary_mask, time_channel]
    Output: (ny, nx, 1) — predicted field
    """

    def __init__(self, ny, nx, interior_mask):
        self.ny = ny
        self.nx = nx
        self.interior_mask = interior_mask.astype(np.float32)
        self.interior_mask_tf = tf.constant(
            self.interior_mask.reshape(1, ny, nx, 1), dtype=tf.float32
        )
        self.model = self._build_unet()

    def _build_unet(self):
        inp = Input(shape=(self.ny, self.nx, 3))

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
        return Model(inp, out, name="pure_cnn")

    def interior_mse(self, y_true, y_pred):
        diff = (y_true - y_pred) * self.interior_mask_tf
        return tf.reduce_sum(diff ** 2) / tf.reduce_sum(self.interior_mask_tf)

    def train_and_evaluate(self, U, obs_mask, train_indices, test_indices,
                           epochs=200, batch_size=8):
        """
        Train pure CNN and evaluate on test set.

        Args:
            U: (nt, ny, nx) original SST array
            obs_mask: (ny, nx) boolean boundary mask
            train_indices: array of training timestep indices
            test_indices: array of test timestep indices

        Returns:
            test_maes: list of per-timestep MAE on test set
        """
        nt = U.shape[0]
        ny, nx = self.ny, self.nx
        int_y, int_x = np.where(~obs_mask)

        # Standardise using training data
        U_train = U[train_indices]
        U_mean = np.mean(U_train, axis=0)
        U_std = np.std(U_train, axis=0)
        U_std = np.maximum(U_std, 1e-8)
        U_norm = (U - U_mean) / U_std
        t_max = float(nt - 1)

        obs_f = obs_mask.astype(np.float32)

        def make_arrays(indices):
            N = len(indices)
            X = np.zeros((N, ny, nx, 3), dtype=np.float32)
            Y = np.zeros((N, ny, nx, 1), dtype=np.float32)
            for i, t in enumerate(indices):
                X[i, :, :, 0] = U_norm[t] * obs_f
                X[i, :, :, 1] = obs_f
                X[i, :, :, 2] = float(t) / t_max
                Y[i, :, :, 0] = U_norm[t]
            return X, Y

        print("  Building training data for pure CNN...")
        X_train, Y_train = make_arrays(train_indices)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=self.interior_mse,
        )
        print(f"  Pure CNN params: {self.model.count_params():,}")

        print("  Training pure CNN...")
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # Evaluate
        print("  Evaluating on test set...")
        X_test, _ = make_arrays(test_indices)
        preds = self.model.predict(X_test, batch_size=batch_size)

        maes = []
        for i, t in enumerate(test_indices):
            u_pred_norm = preds[i, :, :, 0]
            u_pred = u_pred_norm * U_std + U_mean
            u_true = U[t]
            mae = float(np.mean(np.abs(u_pred[int_y, int_x] - u_true[int_y, int_x])))
            maes.append(mae)

        return maes
