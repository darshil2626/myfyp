import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense, Input
from keras.models import Model
import pickle
from typing import Tuple, List


class sinn():
    def __init__(self, X, Y, U, T):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.U = np.asarray(U)
        self.T = np.asarray(T)
        self.nt, self.ny, self.nx = self.U.shape
        
        # Precompute normalization factors
        self.nt_norm = float(self.nt - 1) if self.nt > 1 else 1.0
        self.ny_norm = float(self.ny - 1) if self.ny > 1 else 1.0
        self.nx_norm = float(self.nx - 1) if self.nx > 1 else 1.0
        
    def standardise_u(self):
        self.U_mean = np.mean(self.U, axis=0)
        self.U_std = np.std(self.U, axis=0)
        self.U = (self.U - self.U_mean) / self.U_std
    
    def unstandardise_u(self, U_pred):
        return U_pred * self.U_std + self.U_mean
        
    def split_interior_boundary(self, b_thick, include_t0=True, include_tT=True):
        # Spatial boundary mask (2D)
        sp_int = np.zeros((self.ny, self.nx), dtype=bool)
        sp_int[b_thick:-b_thick, b_thick:-b_thick] = True
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
        safe[:, :b_thick, :] = False
        safe[:, -b_thick:, :] = False
        safe[:, :, :b_thick] = False
        safe[:, :, -b_thick:] = False

        self.mask_boundary = mask_bnd
        self.mask_fd_safe_interior = safe & mask_int
    
    @staticmethod
    def _build_coder(num_units, num_layers, input_shape, output_shape, name, dropout, l2_reg):
        """Build encoder/decoder network with optional dropout and L2 regularization."""
        inputs = Input(shape=(input_shape,))
        x = Dense(num_units, activation='tanh', kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
        for _ in range(num_layers - 1):
            x = Dense(num_units, activation='tanh', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        if dropout and dropout > 0.0:
            x = layers.Dropout(dropout)(x)
        outputs = Dense(output_shape, activation=None)(x)
        return Model(inputs, outputs, name=name)
    
    def build_models(self, num_latentdim, num_units, num_layers, dropout, l2_reg, lr):
        self.interior_encoder = self._build_coder(num_units, num_layers, input_shape=4, 
                                           output_shape=num_latentdim, name='interior encoder', 
                                           dropout=dropout, l2_reg=l2_reg)
        self.boundary_encoder = self._build_coder(num_units, num_layers, input_shape=4, 
                                           output_shape=num_latentdim, name='boundary encoder', 
                                           dropout=dropout, l2_reg=l2_reg)  
        self.decoder = self._build_coder(num_units, num_layers, input_shape=num_latentdim, 
                                   output_shape=1, name='decoder', 
                                   dropout=dropout, l2_reg=l2_reg)
        
        # Initialize A_matrix
        self.a_matrix = tf.Variable(np.eye(num_latentdim, dtype=np.float32), 
                                    trainable=True, dtype=tf.float32, name='pde_operator')
        
        self.trainable_vars = (self.interior_encoder.trainable_variables + 
                              self.boundary_encoder.trainable_variables + 
                              self.decoder.trainable_variables + 
                              [self.a_matrix])
        
        print(f"Trainable variables: {len(self.trainable_vars)}")
        
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        
    def create_patch_centres_and_indices_and_values_and_boundary_splits(self, patch_dim, num_patches):
        """
        Optimized patch creation with precomputed masks and vectorized operations.
        Works for both odd and even patch dimensions.
        """
        px, py, pt = int(patch_dim[0]), int(patch_dim[1]), int(patch_dim[2])
        
        # Calculate left and right offsets for even/odd dimensions
        # For odd (e.g., 5): left=2, right=2 -> 5 points centered
        # For even (e.g., 6): left=3, right=2 -> 6 points (slightly left-biased)
        x_left, x_right = px // 2, (px - 1) // 2
        y_left, y_right = py // 2, (py - 1) // 2
        t_left, t_right = pt // 2, (pt - 1) // 2

        # Centres so patch box is in-bounds
        x_min, x_max = x_left, self.nx - x_right - 1
        y_min, y_max = y_left, self.ny - y_right - 1
        t_min, t_max = t_left, self.nt - t_right - 1

        # Preallocate as numpy arrays for better memory efficiency
        self.patch_center_idx = np.zeros((num_patches, 3), dtype=np.int32)
        
        # Use lists for variable-length arrays (converted to object arrays at end)
        self.patch_interior_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_idx = np.empty(num_patches, dtype=object)
        self.patch_interior_values = np.empty(num_patches, dtype=object)
        self.patch_boundary_values = np.empty(num_patches, dtype=object)
        self.patch_boundary_global_boundary_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_global_interior_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_global_boundary_values = np.empty(num_patches, dtype=object)
        self.patch_boundary_global_interior_values = np.empty(num_patches, dtype=object)

        # Precompute local masks (constant for all patches)
        patch_boundary_mask_local = np.zeros((pt, py, px), dtype=bool)
        patch_boundary_mask_local[0, :, :] = True
        patch_boundary_mask_local[-1, :, :] = True
        patch_boundary_mask_local[:, 0, :] = True
        patch_boundary_mask_local[:, -1, :] = True
        patch_boundary_mask_local[:, :, 0] = True
        patch_boundary_mask_local[:, :, -1] = True
        patch_strict_interior_mask_local = ~patch_boundary_mask_local
        
        # Precompute boundary indices in local coordinates (same for all patches)
        patch_boundary_local_indices_tyx = np.argwhere(patch_boundary_mask_local).astype(np.int32)

        # Vectorized random sampling of patch centers
        centers_x = np.random.randint(x_min, x_max + 1, size=num_patches)
        centers_y = np.random.randint(y_min, y_max + 1, size=num_patches)
        centers_t = np.random.randint(t_min, t_max + 1, size=num_patches)
        
        for k in range(num_patches):
            cx, cy, ct = centers_x[k], centers_y[k], centers_t[k]
            self.patch_center_idx[k] = [ct, cy, cx]

            # Patch bounds (works for both odd and even dimensions)
            x_start, x_end = cx - x_left, cx + x_right + 1
            y_start, y_end = cy - y_left, cy + y_right + 1
            t_start, t_end = ct - t_left, ct + t_right + 1

            # Slice masks to patch subvolume
            mask_fd_safe_interior_subvolume = self.mask_fd_safe_interior[t_start:t_end, y_start:y_end, x_start:x_end]
            mask_global_boundary_subvolume = self.mask_boundary[t_start:t_end, y_start:y_end, x_start:x_end]

            # Patch boundary indices (vectorized offset computation)
            patch_boundary_idx_tyx = patch_boundary_local_indices_tyx.copy()
            patch_boundary_idx_tyx[:, 0] += t_start
            patch_boundary_idx_tyx[:, 1] += y_start
            patch_boundary_idx_tyx[:, 2] += x_start

            # Vectorized value extraction
            patch_boundary_values = self.U[
                patch_boundary_idx_tyx[:, 0],
                patch_boundary_idx_tyx[:, 1],
                patch_boundary_idx_tyx[:, 2],
            ].astype(np.float32)

            # Strict patch interior indices
            mask_patch_interior_and_fd_safe = patch_strict_interior_mask_local & mask_fd_safe_interior_subvolume
            patch_interior_local_indices_tyx = np.argwhere(mask_patch_interior_and_fd_safe).astype(np.int32)

            patch_interior_idx_tyx = patch_interior_local_indices_tyx.copy()
            patch_interior_idx_tyx[:, 0] += t_start
            patch_interior_idx_tyx[:, 1] += y_start
            patch_interior_idx_tyx[:, 2] += x_start

            if patch_interior_idx_tyx.shape[0] > 0:
                patch_interior_values = self.U[
                    patch_interior_idx_tyx[:, 0],
                    patch_interior_idx_tyx[:, 1],
                    patch_interior_idx_tyx[:, 2],
                ].astype(np.float32)
            else:
                patch_interior_values = np.zeros((0,), dtype=np.float32)

            # Split patch boundary into global-boundary vs global-interior (vectorized)
            boundary_local_t = patch_boundary_local_indices_tyx[:, 0]
            boundary_local_y = patch_boundary_local_indices_tyx[:, 1]
            boundary_local_x = patch_boundary_local_indices_tyx[:, 2]

            global_boundary_mask_on_patch_boundary = mask_global_boundary_subvolume[
                boundary_local_t, boundary_local_y, boundary_local_x
            ]

            patch_boundary_global_boundary_idx_tyx = patch_boundary_idx_tyx[global_boundary_mask_on_patch_boundary]
            patch_boundary_global_interior_idx_tyx = patch_boundary_idx_tyx[~global_boundary_mask_on_patch_boundary]

            patch_boundary_global_boundary_values = patch_boundary_values[global_boundary_mask_on_patch_boundary]
            patch_boundary_global_interior_values = patch_boundary_values[~global_boundary_mask_on_patch_boundary]

            # Store (using object arrays for variable-length data)
            self.patch_boundary_idx[k] = patch_boundary_idx_tyx
            self.patch_boundary_values[k] = patch_boundary_values
            self.patch_interior_idx[k] = patch_interior_idx_tyx
            self.patch_interior_values[k] = patch_interior_values
            self.patch_boundary_global_boundary_idx[k] = patch_boundary_global_boundary_idx_tyx
            self.patch_boundary_global_boundary_values[k] = patch_boundary_global_boundary_values
            self.patch_boundary_global_interior_idx[k] = patch_boundary_global_interior_idx_tyx
            self.patch_boundary_global_interior_values[k] = patch_boundary_global_interior_values
    
    def _stack_features_from_idx(self, idx_tyx):
        """Optimized feature stacking with precomputed normalization."""
        idx_tyx = np.asarray(idx_tyx, dtype=np.int32)
        if idx_tyx.size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        t = idx_tyx[:, 0]
        y = idx_tyx[:, 1]
        x = idx_tyx[:, 2]

        # Use precomputed normalization factors
        t_norm = t.astype(np.float32) / self.nt_norm
        y_norm = y.astype(np.float32) / self.ny_norm
        x_norm = x.astype(np.float32) / self.nx_norm

        u_val = self.U[t, y, x].astype(np.float32)

        return np.stack([t_norm, y_norm, x_norm, u_val], axis=1).astype(np.float32)
    
    def stack_features_from_idx_batch(self, idx_list):
        """Batch version of stack_features_from_idx for multiple patches."""
        return [self._stack_features_from_idx(idx) for idx in idx_list]
    
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
        Optimized PDE loss computation.
        Note: @tf.function removed due to numpy() calls and dynamic control flow.
        Performance is still improved through vectorization and precomputation.
        """
        # Inputs are already numpy arrays
        patch_boundary_idx_np = np.asarray(patch_boundary_idx_tyx, dtype=np.int32)
        patch_time_indices_np = np.asarray(patch_time_indices_in_patch, dtype=np.int32)
        
        # Infer patch spatial bounds
        y_min = int(patch_boundary_idx_np[:, 1].min())
        y_max = int(patch_boundary_idx_np[:, 1].max())
        x_min = int(patch_boundary_idx_np[:, 2].min())
        x_max = int(patch_boundary_idx_np[:, 2].max())

        patch_spatial_height = (y_max - y_min + 1)
        patch_spatial_width = (x_max - x_min + 1)

        # Local interior mask
        spatial_strict_interior_mask = np.zeros((patch_spatial_height, patch_spatial_width), dtype=bool)
        if patch_spatial_height >= 3 and patch_spatial_width >= 3:
            spatial_strict_interior_mask[1:-1, 1:-1] = True

        interior_spatial_local_yx = np.argwhere(spatial_strict_interior_mask).astype(np.int32)
        num_interior_spatial_nodes = int(interior_spatial_local_yx.shape[0])

        if num_interior_spatial_nodes == 0:
            return (tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32), 
                   tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32))

        # Build row mapping
        interior_spatial_row_id = -np.ones((patch_spatial_height, patch_spatial_width), dtype=np.int32)
        for row_id, (ly, lx) in enumerate(interior_spatial_local_yx):
            interior_spatial_row_id[ly, lx] = row_id

        patch_center_local_y = int(patch_center_spatial_y_index - y_min)
        patch_center_local_x = int(patch_center_spatial_x_index - x_min)

        centre_row_id = int(interior_spatial_row_id[patch_center_local_y, patch_center_local_x])
        if centre_row_id < 0:
            return (tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32), 
                   tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32))

        # Build Laplacian matrix (vectorized where possible)
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
        latent_dim = int(latent_operator_matrix.shape[0])

        eigenvalues = tf.linalg.eigvalsh(latent_operator_matrix)
        spd_violation = tf.nn.relu(spd_epsilon - eigenvalues)
        spd_loss = rho_spd * tf.reduce_sum(spd_violation * spd_violation)

        # Kronecker product for stiffness matrix
        stiffness_matrix_tf = tf.linalg.LinearOperatorKronecker([
            tf.linalg.LinearOperatorFullMatrix(laplacian_operator_matrix_tf),
            tf.linalg.LinearOperatorFullMatrix(latent_operator_matrix),
        ]).to_dense()

        # Loss accumulators
        latent_consistency_loss_accum = tf.constant(0.0, tf.float32)
        reconstruction_loss_accum = tf.constant(0.0, tf.float32)

        num_time_slices = int(patch_time_indices_np.shape[0])

        # Get center features
        patch_centre_time_yx_indices = np.stack([
            patch_time_indices_np,
            np.full((num_time_slices,), patch_center_spatial_y_index, dtype=np.int32),
            np.full((num_time_slices,), patch_center_spatial_x_index, dtype=np.int32)],
            axis=1
        )

        t = patch_centre_time_yx_indices[:, 0]
        y = patch_centre_time_yx_indices[:, 1]
        x = patch_centre_time_yx_indices[:, 2]
        t_norm = t.astype(np.float32) / self.nt_norm
        y_norm = y.astype(np.float32) / self.ny_norm
        x_norm = x.astype(np.float32) / self.nx_norm
        u_val = self.U[t, y, x].astype(np.float32)

        centre_features = np.stack([t_norm, y_norm, x_norm, u_val], axis=1).astype(np.float32)
        centre_features_tf = tf.constant(centre_features, dtype=tf.float32)

        latent_true_at_patch_centre_all_times = self.interior_encoder(centre_features_tf, training=True)
        u_true_at_patch_centre_all_times = tf.constant(u_val.reshape(-1, 1), dtype=tf.float32)

        # Per-time slice solve
        patch_boundary_t = patch_boundary_idx_np[:, 0]
        patch_boundary_y = patch_boundary_idx_np[:, 1]
        patch_boundary_x = patch_boundary_idx_np[:, 2]
        boundary_latents_tf = latent_values_on_patch_boundary_aligned_with_patch_boundary_idx

        for time_list_index, t_n in enumerate(patch_time_indices_np):
            boundary_mask_this_time = (patch_boundary_t == int(t_n))
            boundary_indices_this_time = np.nonzero(boundary_mask_this_time)[0].astype(np.int32)

            if boundary_indices_this_time.size == 0:
                continue

            boundary_y_this_time = patch_boundary_y[boundary_indices_this_time]
            boundary_x_this_time = patch_boundary_x[boundary_indices_this_time]
            boundary_latents_this_time = tf.gather(boundary_latents_tf, boundary_indices_this_time, axis=0)

            # Build boundary lookup
            boundary_lookup = -np.ones((patch_spatial_height, patch_spatial_width), dtype=np.int32)
            for j, (yy, xx) in enumerate(zip(boundary_y_this_time, boundary_x_this_time)):
                boundary_lookup[int(yy - y_min), int(xx - x_min)] = j

            # Assemble RHS
            rhs_blocks = []
            for row_id, (ly, lx) in enumerate(interior_spatial_local_yx):
                rhs_r = tf.zeros((latent_dim,), dtype=tf.float32)

                for (nly, nlx) in boundary_neighbour_local_yx_per_row[row_id]:
                    boundary_j = int(boundary_lookup[nly, nlx])
                    if boundary_j >= 0:
                        neighbour_latent = boundary_latents_this_time[boundary_j, :]
                        rhs_r = rhs_r + tf.linalg.matvec(latent_operator_matrix, neighbour_latent)

                rhs_blocks.append(rhs_r)

            rhs_vector_tf = tf.concat([tf.reshape(v, (latent_dim, 1)) for v in rhs_blocks], axis=0)

            # Solve system
            latent_solution_vector_tf = tf.linalg.solve(stiffness_matrix_tf, rhs_vector_tf)
            latent_solution_vector_tf = tf.reshape(latent_solution_vector_tf, (num_interior_spatial_nodes, latent_dim))

            latent_pred_at_centre_this_time = latent_solution_vector_tf[centre_row_id, :]
            latent_true_at_centre_this_time = latent_true_at_patch_centre_all_times[time_list_index, :]
            
            latent_consistency_loss_accum += tf.reduce_sum(
                tf.square(latent_pred_at_centre_this_time - latent_true_at_centre_this_time)
            )

            u_pred_at_centre_this_time = self.decoder(
                tf.reshape(latent_pred_at_centre_this_time, (1, latent_dim)),
                training=True
            )
            u_true_at_centre_this_time = tf.reshape(u_true_at_patch_centre_all_times[time_list_index, :], (1, 1))

            reconstruction_loss_accum += tf.reduce_sum(
                tf.square(u_pred_at_centre_this_time - u_true_at_centre_this_time)
            )

        denom = tf.cast(tf.maximum(1, num_time_slices), tf.float32)
        latent_loss = latent_consistency_loss_accum / denom
        recon_loss = reconstruction_loss_accum / denom

        total_loss = latent_loss + alpha_recon * recon_loss + spd_loss
        return total_loss, latent_loss, recon_loss, spd_loss

    def train(self, epochs, patch_dim, num_patches, use_mixed_precision=False):
        """
        Optimized training loop with optional mixed precision training.
        
        Args:
            epochs: Number of training epochs
            patch_dim: Patch dimensions [x, y, t]
            num_patches: Number of patches per epoch
            use_mixed_precision: Enable mixed precision training for better GPU utilization
        """
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision training enabled")

        loss_history = []

        for epoch in range(epochs):
            # Build patches once per epoch
            self.create_patch_centres_and_indices_and_values_and_boundary_splits(
                patch_dim=patch_dim,
                num_patches=num_patches
            )

            epoch_total_loss_value = 0.0

            # Process patches
            for patch_k in range(num_patches):
                # Fetch precomputed data
                patch_center_idx_tyx = self.patch_center_idx[patch_k]
                patch_interior_idx_tyx = self.patch_interior_idx[patch_k]
                patch_boundary_idx_tyx = self.patch_boundary_idx[patch_k]
                patch_boundary_global_boundary_idx_tyx = self.patch_boundary_global_boundary_idx[patch_k]
                patch_boundary_global_interior_idx_tyx = self.patch_boundary_global_interior_idx[patch_k]

                # Build encoder inputs
                patch_center_features = self._stack_features_from_idx(patch_center_idx_tyx[None, :])
                patch_boundary_features_global_boundary = self._stack_features_from_idx(patch_boundary_global_boundary_idx_tyx)
                patch_boundary_features_global_interior = self._stack_features_from_idx(patch_boundary_global_interior_idx_tyx)

                with tf.GradientTape() as tape:
                    # Interior encoder for patch center
                    latent_at_patch_center_from_interior_encoder = self.interior_encoder(
                        tf.convert_to_tensor(patch_center_features, dtype=tf.float32),
                        training=True
                    )

                    # Boundary and interior encoders for patch boundary
                    latent_on_patch_boundary_from_boundary_encoder = self.boundary_encoder(
                        tf.convert_to_tensor(patch_boundary_features_global_boundary, dtype=tf.float32),
                        training=True
                    )

                    latent_on_patch_boundary_from_interior_encoder = self.interior_encoder(
                        tf.convert_to_tensor(patch_boundary_features_global_interior, dtype=tf.float32),
                        training=True
                    )

                    # Stitch boundary latents
                    if patch_boundary_idx_tyx.shape[0] > 0:
                        tb = patch_boundary_idx_tyx[:, 0]
                        yb = patch_boundary_idx_tyx[:, 1]
                        xb = patch_boundary_idx_tyx[:, 2]
                        global_boundary_mask_aligned = self.mask_boundary[tb, yb, xb]

                        indices_boundary = np.nonzero(global_boundary_mask_aligned)[0].astype(np.int32)
                        indices_interior = np.nonzero(~global_boundary_mask_aligned)[0].astype(np.int32)

                        if int(latent_on_patch_boundary_from_boundary_encoder.shape[0]) > 0:
                            latent_dim = int(latent_on_patch_boundary_from_boundary_encoder.shape[-1])
                        else:
                            latent_dim = int(latent_on_patch_boundary_from_interior_encoder.shape[-1])

                        latent_on_patch_boundary_aligned = tf.zeros(
                            (patch_boundary_idx_tyx.shape[0], latent_dim),
                            dtype=tf.float32
                        )

                        if indices_boundary.size > 0:
                            latent_on_patch_boundary_aligned = tf.tensor_scatter_nd_update(
                                latent_on_patch_boundary_aligned,
                                indices=tf.constant(indices_boundary)[:, None],
                                updates=latent_on_patch_boundary_from_boundary_encoder
                            )

                        if indices_interior.size > 0:
                            latent_on_patch_boundary_aligned = tf.tensor_scatter_nd_update(
                                latent_on_patch_boundary_aligned,
                                indices=tf.constant(indices_interior)[:, None],
                                updates=latent_on_patch_boundary_from_interior_encoder
                            )
                    else:
                        latent_on_patch_boundary_aligned = tf.zeros((0, int(self.a_matrix.shape[0])), dtype=tf.float32)

                    # Compute PDE loss
                    patch_time_indices_in_patch = np.unique(self.patch_boundary_idx[patch_k][:, 0])

                    total_loss, latent_loss, recon_loss, spd_loss = self.compute_pde_loss(
                        patch_center_spatial_y_index=int(patch_center_idx_tyx[1]),
                        patch_center_spatial_x_index=int(patch_center_idx_tyx[2]),
                        patch_time_indices_in_patch=patch_time_indices_in_patch,
                        patch_boundary_idx_tyx=patch_boundary_idx_tyx,
                        latent_values_on_patch_boundary_aligned_with_patch_boundary_idx=latent_on_patch_boundary_aligned,
                        rho_spd=1e-3,
                        alpha_recon=1.0,
                    )

                # Apply gradients
                grads = tape.gradient(total_loss, self.trainable_vars)
                
                # Gradient clipping for stability
                grads, _ = tf.clip_by_global_norm(grads, 1.0)
                
                self.optimizer.apply_gradients(zip(grads, self.trainable_vars))

                epoch_total_loss_value += float(total_loss.numpy())

            epoch_total_loss_value /= float(num_patches)
            loss_history.append(epoch_total_loss_value)

            print(f"epoch {epoch+1:04d} | loss {epoch_total_loss_value:.6e}")

        return loss_history


if __name__ == "__main__":
    # Constants and hyperparameters
    b_thick = 1
    include_t0 = True
    include_tT = True
    num_latentdim = 3
    num_units = 64
    num_layers = 3
    dropout = 0.0
    l2_reg = 1e-5
    lr = 1e-3
    patch_dim = [4, 4, 4]  # (x, y, t)
    num_patches = 50
    epochs = 3
    
    # Load variables from pickle file
    with open(r'c:\Users\darsh\Documents\fyp\myfyp\advection_diffusion\time_in_encoders_only\numerical_data.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    Y = data['Y']
    U = data['U']
    T = data['T']

    # Initialize and train
    solver = sinn(X, Y, U, T)
    solver.standardise_u()
    solver.split_interior_boundary(b_thick, include_t0, include_tT)
    solver.build_models(num_latentdim, num_units, num_layers, dropout, l2_reg, lr)
    
    # Train with optional mixed precision (set to True for better GPU performance)
    loss_history = solver.train(epochs, patch_dim, num_patches, use_mixed_precision=False)
    
    print("\nTraining complete!")
    print(f"Final loss: {loss_history[-1]:.6e}")