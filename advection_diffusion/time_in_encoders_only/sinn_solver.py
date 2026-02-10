import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Input
from keras.models import Model
import pickle


class sinn:
    """
    Structure-Informed Neural Network (SINN) solver for time-dependent fields.

    Current training strategy:
      - Sample spatio-temporal cuboid patches.
      - Encode patch boundary latents (boundary encoder on global boundary, interior encoder elsewhere).
      - For each time slice in the patch, solve an elliptic latent PDE with a shared A per patch.
      - Supervise latent consistency + reconstruction over the full spatial interior at each time slice.
    """

    def __init__(self, X, Y, U, T, debug: bool = False):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.U = np.asarray(U)
        self.T = np.asarray(T)
        self.nt, self.ny, self.nx = self.U.shape

        self.debug = bool(debug)

        # Precompute normalization factors for index-normalised coordinates
        self.nt_norm = float(self.nt - 1) if self.nt > 1 else 1.0
        self.ny_norm = float(self.ny - 1) if self.ny > 1 else 1.0
        self.nx_norm = float(self.nx - 1) if self.nx > 1 else 1.0

        # Filled later
        self.b_thick = 1
        self.num_latentdim = None

    # -----------------------------
    # Normalisation helpers
    # -----------------------------
    def standardise_u(self, eps: float = 1e-8):
        """
        Standardise U per (y,x) location across time: (U - mean_t) / std_t.

        eps prevents division by zero at points with (near) zero variance over time.
        """
        self.U_mean = np.mean(self.U, axis=0)
        self.U_std = np.std(self.U, axis=0)
        self.U_std = np.maximum(self.U_std, eps)
        self.U = (self.U - self.U_mean) / self.U_std

    def unstandardise_u(self, U_pred):
        return U_pred * self.U_std + self.U_mean

    # -----------------------------
    # Masks
    # -----------------------------
    def split_interior_boundary(self, b_thick: int, include_t0: bool = True, include_tT: bool = True):
        """
        Build:
          - self.mask_boundary: includes spatial boundary for all times, and optional t=0 and t=T faces.
          - self.mask_fd_safe_interior: interior points with +/-1 neighbors in (t,y,x) and not on boundary.
        """
        self.b_thick = int(b_thick)

        # Spatial boundary mask (2D)
        sp_int = np.zeros((self.ny, self.nx), dtype=bool)
        sp_int[self.b_thick:-self.b_thick, self.b_thick:-self.b_thick] = True
        sp_bnd = ~sp_int

        # Lift to 3D
        mask_bnd = np.zeros((self.nt, self.ny, self.nx), dtype=bool)
        mask_bnd[:, sp_bnd] = True  # spatial boundary for all times

        if include_t0:
            mask_bnd[0, :, :] = True
        if include_tT:
            mask_bnd[-1, :, :] = True

        mask_int = ~mask_bnd

        # FD-safe interior: must have neighbors in +/-1 along x,y,t
        safe = np.zeros((self.nt, self.ny, self.nx), dtype=bool)
        safe[1:-1, 1:-1, 1:-1] = True
        safe[:, :self.b_thick, :] = False
        safe[:, -self.b_thick:, :] = False
        safe[:, :, :self.b_thick] = False
        safe[:, :, -self.b_thick:] = False

        self.mask_boundary = mask_bnd
        self.mask_fd_safe_interior = safe & mask_int

    # -----------------------------
    # Models
    # -----------------------------
    @staticmethod
    def _build_coder(num_units, num_layers, input_shape, output_shape, name, dropout, l2_reg):
        """Build encoder/decoder network with optional dropout and L2 regularization."""
        inputs = Input(shape=(input_shape,))
        x = Dense(num_units, activation="tanh", kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
        for _ in range(num_layers - 1):
            x = Dense(num_units, activation="tanh", kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        if dropout and dropout > 0.0:
            x = layers.Dropout(dropout)(x)
        outputs = Dense(output_shape, activation=None)(x)
        return Model(inputs, outputs, name=name)

    def build_models(self, num_latentdim, num_units, num_layers, dropout, l2_reg, lr):
        self.num_latentdim = int(num_latentdim)

        self.interior_encoder = self._build_coder(
            num_units, num_layers, input_shape=4, output_shape=self.num_latentdim,
            name="interior_encoder", dropout=dropout, l2_reg=l2_reg
        )
        self.boundary_encoder = self._build_coder(
            num_units, num_layers, input_shape=4, output_shape=self.num_latentdim,
            name="boundary_encoder", dropout=dropout, l2_reg=l2_reg
        )
        self.decoder = self._build_coder(
            num_units, num_layers, input_shape=self.num_latentdim, output_shape=1,
            name="decoder", dropout=dropout, l2_reg=l2_reg
        )

        # PDE operator in latent space (learned)
        self.a_matrix = tf.Variable(
            np.eye(self.num_latentdim, dtype=np.float32),
            trainable=True,
            dtype=tf.float32,
            name="pde_operator",
        )

        self.trainable_vars = (
            self.interior_encoder.trainable_variables
            + self.boundary_encoder.trainable_variables
            + self.decoder.trainable_variables
            + [self.a_matrix]
        )

        print(f"Trainable variables: {len(self.trainable_vars)}")
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    # -----------------------------
    # Patch sampling (indices only)
    # -----------------------------
    def create_patch_centres_and_indices_and_values_and_boundary_splits(self, patch_dim, num_patches):
        """
        Create random spatio-temporal patches and store indices needed for training.

        - Stores indices only (no values).
        - Patch boundary is the outer shell of the cuboid (including t-faces).
        - Patch interior are points strictly inside that shell.
        """
        px, py, pt = int(patch_dim[0]), int(patch_dim[1]), int(patch_dim[2])

        # Offsets that keep the patch in-bounds.
        x_left, x_right = px // 2, (px - 1) // 2
        y_left, y_right = py // 2, (py - 1) // 2
        t_left, t_right = pt // 2, (pt - 1) // 2

        x_min, x_max = x_left, self.nx - x_right - 1
        y_min, y_max = y_left, self.ny - y_right - 1
        t_min, t_max = t_left, self.nt - t_right - 1

        if x_min > x_max or y_min > y_max or t_min > t_max:
            raise ValueError("Patch dimensions are too large for the grid.")

        # Centres
        self.patch_center_idx = np.zeros((num_patches, 3), dtype=np.int32)

        # Variable-length arrays per patch (indices only)
        self.patch_interior_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_global_boundary_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_global_interior_idx = np.empty(num_patches, dtype=object)

        # Local boundary mask for a (pt, py, px) cuboid
        patch_boundary_mask_local = np.zeros((pt, py, px), dtype=bool)
        patch_boundary_mask_local[0, :, :] = True
        patch_boundary_mask_local[-1, :, :] = True
        patch_boundary_mask_local[:, 0, :] = True
        patch_boundary_mask_local[:, -1, :] = True
        patch_boundary_mask_local[:, :, 0] = True
        patch_boundary_mask_local[:, :, -1] = True
        patch_interior_mask_local = ~patch_boundary_mask_local

        # Local offsets relative to centre
        t_offsets = np.arange(-t_left, t_right + 1, dtype=np.int32)
        y_offsets = np.arange(-y_left, y_right + 1, dtype=np.int32)
        x_offsets = np.arange(-x_left, x_right + 1, dtype=np.int32)
        TT, YY, XX = np.meshgrid(t_offsets, y_offsets, x_offsets, indexing="ij")

        local_offsets_flat = np.stack([TT.ravel(), YY.ravel(), XX.ravel()], axis=1)  # (pt*py*px, 3)
        boundary_mask_flat = patch_boundary_mask_local.ravel()
        interior_mask_flat = patch_interior_mask_local.ravel()

        rng = np.random.default_rng()

        for k in range(num_patches):
            ct = rng.integers(t_min, t_max + 1)
            cy = rng.integers(y_min, y_max + 1)
            cx = rng.integers(x_min, x_max + 1)
            self.patch_center_idx[k] = (ct, cy, cx)

            # Absolute indices for all points in the cuboid
            abs_t = ct + local_offsets_flat[:, 0]
            abs_y = cy + local_offsets_flat[:, 1]
            abs_x = cx + local_offsets_flat[:, 2]
            idx_all = np.stack([abs_t, abs_y, abs_x], axis=1).astype(np.int32)

            patch_boundary_idx = idx_all[boundary_mask_flat]
            patch_interior_idx = idx_all[interior_mask_flat]

            self.patch_boundary_idx[k] = patch_boundary_idx
            self.patch_interior_idx[k] = patch_interior_idx

            # Split patch boundary by *global* boundary membership
            if getattr(self, "mask_boundary", None) is not None:
                is_global_boundary = self.mask_boundary[
                    patch_boundary_idx[:, 0], patch_boundary_idx[:, 1], patch_boundary_idx[:, 2]
                ]
            else:
                is_global_boundary = np.zeros((patch_boundary_idx.shape[0],), dtype=bool)

            self.patch_boundary_global_boundary_idx[k] = patch_boundary_idx[is_global_boundary]
            self.patch_boundary_global_interior_idx[k] = patch_boundary_idx[~is_global_boundary]

    # -----------------------------
    # Feature stacking
    # -----------------------------
    def _stack_features_from_idx(self, idx_tyx):
        """Stack features [t_norm, y_norm, x_norm, u] for an (N,3) integer index array."""
        idx_tyx = np.asarray(idx_tyx, dtype=np.int32)
        if idx_tyx.size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        t = idx_tyx[:, 0]
        y = idx_tyx[:, 1]
        x = idx_tyx[:, 2]

        t_norm = t.astype(np.float32) / self.nt_norm
        y_norm = y.astype(np.float32) / self.ny_norm
        x_norm = x.astype(np.float32) / self.nx_norm
        u_val = self.U[t, y, x].astype(np.float32)

        return np.stack([t_norm, y_norm, x_norm, u_val], axis=1).astype(np.float32)

    # -----------------------------
    # Boundary latent encoding + alignment
    # -----------------------------
    def _encode_and_align_patch_boundary_latents(
        self,
        patch_boundary_idx_tyx: np.ndarray,
        patch_boundary_global_boundary_idx_tyx: np.ndarray,
        patch_boundary_global_interior_idx_tyx: np.ndarray,
        training: bool,
    ) -> tf.Tensor:
        """
        Return boundary latents aligned with patch_boundary_idx_tyx order.
        Uses:
          - boundary_encoder for points on global boundary
          - interior_encoder for patch boundary points not on global boundary
        """
        # Encode the two groups (may be empty)
        feats_bnd = self._stack_features_from_idx(patch_boundary_global_boundary_idx_tyx)
        feats_int = self._stack_features_from_idx(patch_boundary_global_interior_idx_tyx)

        lat_bnd = self.boundary_encoder(tf.convert_to_tensor(feats_bnd, dtype=tf.float32), training=training)
        lat_int = self.interior_encoder(tf.convert_to_tensor(feats_int, dtype=tf.float32), training=training)

        n_total = int(patch_boundary_idx_tyx.shape[0])
        if n_total == 0:
            return tf.zeros((0, self.num_latentdim), dtype=tf.float32)

        # Determine which entries in patch_boundary_idx correspond to global boundary
        tb = patch_boundary_idx_tyx[:, 0]
        yb = patch_boundary_idx_tyx[:, 1]
        xb = patch_boundary_idx_tyx[:, 2]
        global_boundary_mask_aligned = self.mask_boundary[tb, yb, xb]

        idx_bnd = np.nonzero(global_boundary_mask_aligned)[0].astype(np.int32)
        idx_int = np.nonzero(~global_boundary_mask_aligned)[0].astype(np.int32)

        out = tf.zeros((n_total, self.num_latentdim), dtype=tf.float32)

        if idx_bnd.size > 0:
            out = tf.tensor_scatter_nd_update(out, indices=tf.constant(idx_bnd)[:, None], updates=lat_bnd)

        if idx_int.size > 0:
            out = tf.tensor_scatter_nd_update(out, indices=tf.constant(idx_int)[:, None], updates=lat_int)

        return out

    # -----------------------------
    # PDE loss (full interior)
    # -----------------------------
    def compute_pde_loss(
        self,
        patch_center_spatial_y_index: int,
        patch_center_spatial_x_index: int,
        patch_time_indices_in_patch: np.ndarray,
        patch_boundary_idx_tyx: np.ndarray,
        latent_values_on_patch_boundary_aligned_with_patch_boundary_idx: tf.Tensor,
        rho_spd: float = 1e-3,
        alpha_recon: float = 1.0,
        spd_epsilon: float = 1e-6,
    ):
        """
        Compute PDE loss with FULL INTERIOR supervision (not just center).

        For each time slice in patch_time_indices_in_patch:
          - Solve (L ⊗ A) vec(L_int) = rhs(boundary latents)
          - Compare L_int to interior_encoder outputs at all strict interior points
          - Decode L_int and compare to true u at those interior points
        """
        patch_boundary_idx_np = np.asarray(patch_boundary_idx_tyx, dtype=np.int32)
        patch_time_indices_np = np.asarray(patch_time_indices_in_patch, dtype=np.int32)

        # Infer patch spatial bounds
        y_min = int(patch_boundary_idx_np[:, 1].min())
        y_max = int(patch_boundary_idx_np[:, 1].max())
        x_min = int(patch_boundary_idx_np[:, 2].min())
        x_max = int(patch_boundary_idx_np[:, 2].max())

        patch_spatial_height = (y_max - y_min + 1)
        patch_spatial_width = (x_max - x_min + 1)

        # Strict spatial interior mask inside patch bounding box
        spatial_strict_interior_mask = np.zeros((patch_spatial_height, patch_spatial_width), dtype=bool)
        if patch_spatial_height >= 3 and patch_spatial_width >= 3:
            spatial_strict_interior_mask[1:-1, 1:-1] = True

        interior_spatial_local_yx = np.argwhere(spatial_strict_interior_mask).astype(np.int32)
        num_interior_spatial_nodes = int(interior_spatial_local_yx.shape[0])
        if num_interior_spatial_nodes == 0:
            z = tf.constant(0.0, tf.float32)
            return z, z, z, z

        # Build row mapping for 2D patch interior
        interior_spatial_row_id = -np.ones((patch_spatial_height, patch_spatial_width), dtype=np.int32)
        for row_id, (ly, lx) in enumerate(interior_spatial_local_yx):
            interior_spatial_row_id[ly, lx] = row_id

        # Optional: sanity check (center should be strict interior for well-chosen patches)
        patch_center_local_y = int(patch_center_spatial_y_index - y_min)
        patch_center_local_x = int(patch_center_spatial_x_index - x_min)
        if interior_spatial_row_id[patch_center_local_y, patch_center_local_x] < 0:
            z = tf.constant(0.0, tf.float32)
            return z, z, z, z

        # Build Laplacian (dense) + boundary-neighbour lists
        laplacian_operator_matrix = np.zeros((num_interior_spatial_nodes, num_interior_spatial_nodes), dtype=np.float32)
        boundary_neighbour_local_yx_per_row = [[] for _ in range(num_interior_spatial_nodes)]
        neighbour_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for row_id, (ly, lx) in enumerate(interior_spatial_local_yx):
            laplacian_operator_matrix[row_id, row_id] += 4.0
            for dy, dx in neighbour_steps:
                nly = int(ly + dy)
                nlx = int(lx + dx)
                neighbour_row = int(interior_spatial_row_id[nly, nlx])
                if neighbour_row >= 0:
                    laplacian_operator_matrix[row_id, neighbour_row] += -1.0
                else:
                    boundary_neighbour_local_yx_per_row[row_id].append((nly, nlx))

        laplacian_operator_matrix_tf = tf.constant(laplacian_operator_matrix, dtype=tf.float32)

        # A matrix and SPD regularisation
        latent_operator_matrix = 0.5 * (self.a_matrix + tf.transpose(self.a_matrix))
        eigenvalues = tf.linalg.eigvalsh(latent_operator_matrix)
        spd_violation = tf.nn.relu(spd_epsilon - eigenvalues)
        spd_loss = rho_spd * tf.reduce_sum(spd_violation * spd_violation)

        # Build stiffness matrix ONCE for this patch spatial footprint
        stiffness_matrix_tf = tf.linalg.LinearOperatorKronecker(
            [
                tf.linalg.LinearOperatorFullMatrix(laplacian_operator_matrix_tf),
                tf.linalg.LinearOperatorFullMatrix(latent_operator_matrix),
            ]
        ).to_dense()

        # Pre-factor the matrix for faster repeated solves across time slices
        try:
            L_cholesky = tf.linalg.cholesky(stiffness_matrix_tf)
            use_cholesky = True
            lu = p = None
        except tf.errors.InvalidArgumentError:
            lu, p = tf.linalg.lu(stiffness_matrix_tf)
            use_cholesky = False
            L_cholesky = None

        latent_consistency_loss_accum = tf.constant(0.0, tf.float32)
        reconstruction_loss_accum = tf.constant(0.0, tf.float32)

        num_time_slices = int(patch_time_indices_np.shape[0])

        # Global coords for all strict interior points
        interior_spatial_global_y = interior_spatial_local_yx[:, 0] + y_min
        interior_spatial_global_x = interior_spatial_local_yx[:, 1] + x_min

        # Boundary arrays for quick time filtering
        patch_boundary_t = patch_boundary_idx_np[:, 0]
        patch_boundary_y = patch_boundary_idx_np[:, 1]
        patch_boundary_x = patch_boundary_idx_np[:, 2]
        boundary_latents_tf = latent_values_on_patch_boundary_aligned_with_patch_boundary_idx

        for t_n in patch_time_indices_np:
            boundary_mask_this_time = (patch_boundary_t == int(t_n))
            boundary_indices_this_time = np.nonzero(boundary_mask_this_time)[0].astype(np.int32)
            if boundary_indices_this_time.size == 0:
                continue

            boundary_y_this_time = patch_boundary_y[boundary_indices_this_time]
            boundary_x_this_time = patch_boundary_x[boundary_indices_this_time]
            boundary_latents_this_time = tf.gather(boundary_latents_tf, boundary_indices_this_time, axis=0)

            # Build boundary lookup (local y,x -> boundary row index) for this time slice
            boundary_lookup = -np.ones((patch_spatial_height, patch_spatial_width), dtype=np.int32)
            for j, (yy, xx) in enumerate(zip(boundary_y_this_time, boundary_x_this_time)):
                boundary_lookup[int(yy - y_min), int(xx - x_min)] = j

            # Assemble RHS blocks (python loop; correct but can be vectorized later)
            rhs_blocks = []
            for row_id in range(num_interior_spatial_nodes):
                rhs_r = tf.zeros((self.num_latentdim,), dtype=tf.float32)
                for (nly, nlx) in boundary_neighbour_local_yx_per_row[row_id]:
                    boundary_j = int(boundary_lookup[nly, nlx])
                    if boundary_j >= 0:
                        neighbour_latent = boundary_latents_this_time[boundary_j, :]
                        rhs_r = rhs_r + tf.linalg.matvec(latent_operator_matrix, neighbour_latent)
                rhs_blocks.append(rhs_r)

            rhs_vector_tf = tf.concat([tf.reshape(v, (self.num_latentdim, 1)) for v in rhs_blocks], axis=0)

            # Solve using pre-factored matrix
            if use_cholesky:
                latent_solution_vector_tf = tf.linalg.cholesky_solve(L_cholesky, rhs_vector_tf)
            else:
                latent_solution_vector_tf = tf.linalg.lu_solve(lu, p, rhs_vector_tf)

            latent_solution_vector_tf = tf.reshape(latent_solution_vector_tf, (num_interior_spatial_nodes, self.num_latentdim))

            # Build features for all interior points at this time
            interior_t = np.full((num_interior_spatial_nodes,), int(t_n), dtype=np.int32)
            interior_y = interior_spatial_global_y.astype(np.int32)
            interior_x = interior_spatial_global_x.astype(np.int32)

            t_norm = interior_t.astype(np.float32) / self.nt_norm
            y_norm = interior_y.astype(np.float32) / self.ny_norm
            x_norm = interior_x.astype(np.float32) / self.nx_norm

            u_val_all = self.U[interior_t, interior_y, interior_x].astype(np.float32)

            interior_features_all = np.stack([t_norm, y_norm, x_norm, u_val_all], axis=1).astype(np.float32)
            interior_features_all_tf = tf.constant(interior_features_all, dtype=tf.float32)

            latent_true_all_interior = self.interior_encoder(interior_features_all_tf, training=True)

            latent_consistency_loss_accum += tf.reduce_sum(tf.square(latent_solution_vector_tf - latent_true_all_interior))

            u_pred_all_interior = self.decoder(latent_solution_vector_tf, training=True)
            u_true_all_interior = tf.constant(u_val_all.reshape(-1, 1), dtype=tf.float32)

            reconstruction_loss_accum += tf.reduce_sum(tf.square(u_pred_all_interior - u_true_all_interior))

        total_points = tf.cast(num_time_slices * num_interior_spatial_nodes, tf.float32)
        latent_loss = latent_consistency_loss_accum / total_points
        recon_loss = reconstruction_loss_accum / total_points

        total_loss = latent_loss + alpha_recon * recon_loss + spd_loss
        return total_loss, latent_loss, recon_loss, spd_loss

    # -----------------------------
    # Training
    # -----------------------------
    def train(self, epochs, patch_dim, num_patches, clip_norm: float = 1.0):
        loss_history = {"total": [], "latent": [], "recon": [], "spd": []}

        for epoch in range(epochs):
            # Build patches once per epoch
            self.create_patch_centres_and_indices_and_values_and_boundary_splits(
                patch_dim=patch_dim, num_patches=num_patches
            )

            epoch_total_loss = 0.0
            epoch_latent_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_spd_loss = 0.0

            for patch_k in range(num_patches):
                patch_center_idx_tyx = self.patch_center_idx[patch_k]
                patch_boundary_idx_tyx = self.patch_boundary_idx[patch_k]
                patch_boundary_global_boundary_idx_tyx = self.patch_boundary_global_boundary_idx[patch_k]
                patch_boundary_global_interior_idx_tyx = self.patch_boundary_global_interior_idx[patch_k]

                with tf.GradientTape() as tape:
                    latent_on_patch_boundary_aligned = self._encode_and_align_patch_boundary_latents(
                        patch_boundary_idx_tyx=patch_boundary_idx_tyx,
                        patch_boundary_global_boundary_idx_tyx=patch_boundary_global_boundary_idx_tyx,
                        patch_boundary_global_interior_idx_tyx=patch_boundary_global_interior_idx_tyx,
                        training=True,
                    )

                    patch_time_indices_in_patch = np.unique(patch_boundary_idx_tyx[:, 0])

                    total_loss, latent_loss, recon_loss, spd_loss = self.compute_pde_loss(
                        patch_center_spatial_y_index=int(patch_center_idx_tyx[1]),
                        patch_center_spatial_x_index=int(patch_center_idx_tyx[2]),
                        patch_time_indices_in_patch=patch_time_indices_in_patch,
                        patch_boundary_idx_tyx=patch_boundary_idx_tyx,
                        latent_values_on_patch_boundary_aligned_with_patch_boundary_idx=latent_on_patch_boundary_aligned,
                        rho_spd=1e-3,
                        alpha_recon=1.0,
                    )

                grads = tape.gradient(total_loss, self.trainable_vars)
                grads, _ = tf.clip_by_global_norm(grads, clip_norm)
                self.optimizer.apply_gradients(zip(grads, self.trainable_vars))

                epoch_total_loss += float(total_loss.numpy())
                epoch_latent_loss += float(latent_loss.numpy())
                epoch_recon_loss += float(recon_loss.numpy())
                epoch_spd_loss += float(spd_loss.numpy())

            # Average over patches
            inv = 1.0 / float(num_patches)
            epoch_total_loss *= inv
            epoch_latent_loss *= inv
            epoch_recon_loss *= inv
            epoch_spd_loss *= inv

            loss_history["total"].append(epoch_total_loss)
            loss_history["latent"].append(epoch_latent_loss)
            loss_history["recon"].append(epoch_recon_loss)
            loss_history["spd"].append(epoch_spd_loss)

            print(
                f"epoch {epoch+1:04d} | total {epoch_total_loss:.6e} | "
                f"latent {epoch_latent_loss:.6e} | recon {epoch_recon_loss:.6e} | "
                f"spd {epoch_spd_loss:.6e}"
            )

        return loss_history

    # -----------------------------
    # Plotting
    # -----------------------------
    def plot_training_history(self, loss_history, save_path=None, show=True):
        import matplotlib.pyplot as plt

        epochs = range(1, len(loss_history["total"]) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, loss_history["total"], "k-", linewidth=2, label="Total Loss", marker="o")
        ax.plot(epochs, loss_history["latent"], "b--", linewidth=1.5, label="Latent Loss", marker="s")
        ax.plot(epochs, loss_history["recon"], "r--", linewidth=1.5, label="Reconstruction Loss", marker="^")
        ax.plot(epochs, loss_history["spd"], "g--", linewidth=1.5, label="SPD Loss", marker="d")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Loss History", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_yscale("log")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        if show:
            plt.show()

        return fig

    # -----------------------------
    # Reconstruction
    # -----------------------------
    def reconstruct_field_at_timestep(self, t_index):
        """
        Reconstruct the entire field at a given timestep using the trained model.

        Uses a spatial (y,x) elliptic solve in latent space for that timestep.
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import spsolve

        t = int(t_index)
        print(f"\nReconstructing field at timestep t={t}...")

        b_thick = int(getattr(self, "b_thick", 1))

        # Spatial interior mask (2D)
        spatial_interior_mask = np.zeros((self.ny, self.nx), dtype=bool)
        spatial_interior_mask[b_thick:-b_thick, b_thick:-b_thick] = True
        spatial_boundary_mask = ~spatial_interior_mask

        interior_indices = np.argwhere(spatial_interior_mask)  # (N_interior, 2)
        boundary_indices = np.argwhere(spatial_boundary_mask)  # (N_boundary, 2)

        num_interior = interior_indices.shape[0]
        num_boundary = boundary_indices.shape[0]

        print(f"  Interior points: {num_interior}")
        print(f"  Boundary points: {num_boundary}")

        if num_interior == 0:
            raise ValueError(
                f"No interior points! Check boundary thickness b_thick={b_thick}. Grid is ({self.ny},{self.nx})."
            )

        # Step 1: Encode boundary conditions
        print("  Step 1: Encoding boundary conditions...")

        boundary_y = boundary_indices[:, 0]
        boundary_x = boundary_indices[:, 1]
        boundary_t = np.full(num_boundary, t, dtype=np.int32)

        t_norm = boundary_t.astype(np.float32) / self.nt_norm
        y_norm = boundary_y.astype(np.float32) / self.ny_norm
        x_norm = boundary_x.astype(np.float32) / self.nx_norm
        u_boundary = self.U[boundary_t, boundary_y, boundary_x].astype(np.float32)

        boundary_features = np.stack([t_norm, y_norm, x_norm, u_boundary], axis=1).astype(np.float32)
        boundary_features_tf = tf.constant(boundary_features, dtype=tf.float32)

        boundary_latents = self.boundary_encoder(boundary_features_tf, training=False).numpy()
        latent_dim = boundary_latents.shape[1]
        print(f"  Latent dimension: {latent_dim}")

        # Step 2: Build spatial Laplacian operator
        print("  Step 2: Building Laplacian operator...")

        interior_row_map = -np.ones((self.ny, self.nx), dtype=np.int32)
        for row_id, (y, x) in enumerate(interior_indices):
            interior_row_map[y, x] = row_id

        row_indices, col_indices, values = [], [], []
        neighbour_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        boundary_contributions = [[] for _ in range(num_interior)]

        for row_id, (y, x) in enumerate(interior_indices):
            row_indices.append(row_id); col_indices.append(row_id); values.append(4.0)
            for dy, dx in neighbour_steps:
                ny, nx = y + dy, x + dx
                neighbour_row = interior_row_map[ny, nx]
                if neighbour_row >= 0:
                    row_indices.append(row_id); col_indices.append(neighbour_row); values.append(-1.0)
                else:
                    # boundary neighbour: find its boundary index once (slow but OK for now)
                    bidx = np.where((boundary_indices[:, 0] == ny) & (boundary_indices[:, 1] == nx))[0]
                    if len(bidx) > 0:
                        boundary_contributions[row_id].append(int(bidx[0]))

        laplacian_sparse = sp.csr_matrix((values, (row_indices, col_indices)), shape=(num_interior, num_interior))

        # Step 3: Get A matrix
        print("  Step 3: Getting PDE operator (A matrix)...")
        A_matrix_np = (0.5 * (self.a_matrix + tf.transpose(self.a_matrix))).numpy()

        stiffness = sp.kron(laplacian_sparse, A_matrix_np, format="csr")
        print(f"  Stiffness matrix size: {stiffness.shape}")

        # Step 4: RHS
        print("  Step 4: Building right-hand side...")
        rhs = np.zeros((num_interior, latent_dim), dtype=np.float32)
        for row_id in range(num_interior):
            for bidx in boundary_contributions[row_id]:
                rhs[row_id, :] += A_matrix_np @ boundary_latents[bidx, :]

        rhs_flat = rhs.reshape(-1)

        # Step 5: Solve
        print("  Step 5: Solving linear system...")
        latent_interior_flat = spsolve(stiffness, rhs_flat)
        latent_interior = latent_interior_flat.reshape((num_interior, latent_dim)).astype(np.float32)
        latent_interior_tf = tf.constant(latent_interior, dtype=tf.float32)

        # Step 6: Decode
        print("  Step 6: Decoding latents to physical field...")
        u_interior_pred = self.decoder(latent_interior_tf, training=False).numpy().reshape(-1)

        # Step 7: Assemble
        print("  Step 7: Assembling full field...")
        u_pred_full = np.zeros((self.ny, self.nx), dtype=np.float32)
        u_true_full = self.U[t, :, :].astype(np.float32)

        u_pred_full[boundary_y, boundary_x] = u_boundary
        interior_y = interior_indices[:, 0]
        interior_x = interior_indices[:, 1]
        u_pred_full[interior_y, interior_x] = u_interior_pred

        u_error = np.abs(u_pred_full - u_true_full)

        interior_error = u_error[interior_y, interior_x]
        mse = float(np.mean(interior_error ** 2))
        mae = float(np.mean(interior_error))
        max_error = float(np.max(interior_error))

        print("\n  Reconstruction Statistics (interior only):")
        print(f"    MSE: {mse:.6e}")
        print(f"    MAE: {mae:.6e}")
        print(f"    Max Error: {max_error:.6e}")
        print("  Reconstruction complete!\n")

        return {
            "u_pred": u_pred_full,
            "u_true": u_true_full,
            "u_error": u_error,
            "boundary_mask": spatial_boundary_mask,
            "interior_mask": spatial_interior_mask,
            "mse": mse,
            "mae": mae,
            "max_error": max_error,
            "t_index": t,
        }

    def plot_field_reconstruction(self, results, save_path=None, show=True):
        import matplotlib.pyplot as plt

        u_true = results["u_true"]
        u_pred = results["u_pred"]
        u_error = results["u_error"]
        boundary_mask = results["boundary_mask"]
        t_index = results["t_index"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        vmin = min(u_true.min(), u_pred.min())
        vmax = max(u_true.max(), u_pred.max())

        boundary_coords = np.argwhere(boundary_mask)

        ax = axes[0]
        im1 = ax.imshow(u_true, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"True Field (t={t_index})", fontsize=13, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.scatter(boundary_coords[:, 1], boundary_coords[:, 0], c="red", s=1, alpha=0.3)
        plt.colorbar(im1, ax=ax).set_label("u")

        ax = axes[1]
        im2 = ax.imshow(u_pred, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title("Predicted Field (Reconstructed)", fontsize=13, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.scatter(boundary_coords[:, 1], boundary_coords[:, 0], c="red", s=1, alpha=0.3)
        plt.colorbar(im2, ax=ax).set_label("u")

        ax = axes[2]
        im3 = ax.imshow(u_error, cmap="hot", origin="lower", aspect="auto")
        ax.set_title("Absolute Error", fontsize=13, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.scatter(boundary_coords[:, 1], boundary_coords[:, 0], c="cyan", s=1, alpha=0.5)
        plt.colorbar(im3, ax=ax).set_label("|error|")

        ax.text(
            0.02, 0.98,
            f"MSE: {results['mse']:.4e}\nMAE: {results['mae']:.4e}\nMax: {results['max_error']:.4e}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.suptitle(f"Field Reconstruction at t={t_index}", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Field reconstruction plot saved to {save_path}")
        if show:
            plt.show()

        return fig


if __name__ == "__main__":
    # Hyperparameters / config
    b_thick = 1
    include_t0 = True
    include_tT = True
    num_latentdim = 10
    num_units = 128
    num_layers = 3
    dropout = 0.0
    l2_reg = 1e-5
    lr = 1e-3
    patch_dim = [10, 10, 10]  # (x, y, t)
    num_patches = 100
    epochs = 5

    with open(r"c:\Users\darsh\Documents\fyp\myfyp\advection_diffusion\time_in_encoders_only\numerical_data.pkl", "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    Y = data["Y"]
    U = data["U"]
    T = data["T"]

    solver = sinn(X, Y, U, T, debug=False)
    solver.standardise_u()
    solver.split_interior_boundary(b_thick, include_t0, include_tT)
    solver.build_models(num_latentdim, num_units, num_layers, dropout, l2_reg, lr)

    if solver.debug:
        print("\nBOUNDARY MASK CHECK:")
        t_mid = solver.nt // 2
        print(f"Grid shape: {solver.U.shape}")
        print(f"Boundary mask shape: {solver.mask_boundary.shape}")
        print(f"At t={t_mid}:")
        print(f"  Boundary points: {np.sum(solver.mask_boundary[t_mid])}")
        print(f"  Interior points: {np.sum(~solver.mask_boundary[t_mid])}")

    loss_history = solver.train(epochs, patch_dim, num_patches)

    print("\nTraining complete!")
    print(f"Final total loss: {loss_history['total'][-1]:.6e}")
    print(f"Final latent loss: {loss_history['latent'][-1]:.6e}")
    print(f"Final recon loss: {loss_history['recon'][-1]:.6e}")
    print(f"Final spd loss: {loss_history['spd'][-1]:.6e}")

    solver.plot_training_history(loss_history, save_path="training_loss.png", show=True)

    results = solver.reconstruct_field_at_timestep(t_index=50)
    solver.plot_field_reconstruction(results, save_path="field_t50.png", show=True)
    print(f"Reconstruction MAE: {results['mae']:.6e}")
