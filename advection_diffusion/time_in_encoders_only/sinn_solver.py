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
    
    # Modified compute_pde_loss method for FULL INTERIOR loss
    # Replace your existing compute_pde_loss method with this one

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
        
        This version compares the PDE solution to encoder predictions at ALL
        interior points, not just the patch center. This provides stronger
        supervision and better data efficiency.
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

        # NOTE: We no longer need centre_row_id for center-only loss
        # But keep it for potential debugging/comparison
        patch_center_local_y = int(patch_center_spatial_y_index - y_min)
        patch_center_local_x = int(patch_center_spatial_x_index - x_min)
        centre_row_id = int(interior_spatial_row_id[patch_center_local_y, patch_center_local_x])
        
        if centre_row_id < 0:
            # Center not in interior (shouldn't happen for proper patches)
            return (tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32), 
                tf.constant(0.0, tf.float32), tf.constant(0.0, tf.float32))

        # Build Laplacian matrix
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

        # Build stiffness matrix ONCE
        stiffness_matrix_tf = tf.linalg.LinearOperatorKronecker([
            tf.linalg.LinearOperatorFullMatrix(laplacian_operator_matrix_tf),
            tf.linalg.LinearOperatorFullMatrix(latent_operator_matrix),
        ]).to_dense()
        
        # Pre-factor the matrix for faster solves
        try:
            L_cholesky = tf.linalg.cholesky(stiffness_matrix_tf)
            use_cholesky = True
        except tf.errors.InvalidArgumentError:
            lu, p = tf.linalg.lu(stiffness_matrix_tf)
            use_cholesky = False

        # Loss accumulators
        latent_consistency_loss_accum = tf.constant(0.0, tf.float32)
        reconstruction_loss_accum = tf.constant(0.0, tf.float32)

        num_time_slices = int(patch_time_indices_np.shape[0])

        # ============================================================================
        # NEW: Prepare global coordinates for ALL interior points (not just center)
        # ============================================================================
        interior_spatial_global_y = interior_spatial_local_yx[:, 0] + y_min
        interior_spatial_global_x = interior_spatial_local_yx[:, 1] + x_min
        # Shape: (num_interior_spatial_nodes,) each

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

            # Solve using pre-factored matrix
            if use_cholesky:
                latent_solution_vector_tf = tf.linalg.cholesky_solve(L_cholesky, rhs_vector_tf)
            else:
                latent_solution_vector_tf = tf.linalg.lu_solve(lu, p, rhs_vector_tf)
            
            latent_solution_vector_tf = tf.reshape(latent_solution_vector_tf, (num_interior_spatial_nodes, latent_dim))
            # Shape: (num_interior_spatial_nodes, latent_dim)

            # ========================================================================
            # NEW: Get features for ALL interior points at this time slice
            # ========================================================================
            
            # Create arrays for all interior points at this time
            interior_t = np.full((num_interior_spatial_nodes,), int(t_n), dtype=np.int32)
            interior_y = interior_spatial_global_y.astype(np.int32)
            interior_x = interior_spatial_global_x.astype(np.int32)
            
            # Normalize coordinates
            t_norm = interior_t.astype(np.float32) / self.nt_norm
            y_norm = interior_y.astype(np.float32) / self.ny_norm
            x_norm = interior_x.astype(np.float32) / self.nx_norm
            
            # Get true values at all interior points
            u_val_all = self.U[interior_t, interior_y, interior_x].astype(np.float32)
            
            # Stack features for all interior points
            interior_features_all = np.stack([t_norm, y_norm, x_norm, u_val_all], axis=1).astype(np.float32)
            interior_features_all_tf = tf.constant(interior_features_all, dtype=tf.float32)
            # Shape: (num_interior_spatial_nodes, 4)
            
            # ========================================================================
            # NEW: Encode ALL interior points (not just center)
            # ========================================================================
            latent_true_all_interior = self.interior_encoder(interior_features_all_tf, training=True)
            # Shape: (num_interior_spatial_nodes, latent_dim)
            
            # ========================================================================
            # NEW: Latent loss over ALL interior points
            # ========================================================================
            latent_consistency_loss_accum += tf.reduce_sum(
                tf.square(latent_solution_vector_tf - latent_true_all_interior)
            )
            # This sums over ALL (num_interior_spatial_nodes * latent_dim) elements
            
            # ========================================================================
            # NEW: Reconstruction loss over ALL interior points
            # ========================================================================
            u_pred_all_interior = self.decoder(latent_solution_vector_tf, training=True)
            # Shape: (num_interior_spatial_nodes, 1)
            
            u_true_all_interior = tf.constant(u_val_all.reshape(-1, 1), dtype=tf.float32)
            # Shape: (num_interior_spatial_nodes, 1)
            
            reconstruction_loss_accum += tf.reduce_sum(
                tf.square(u_pred_all_interior - u_true_all_interior)
            )
            # This sums over ALL num_interior_spatial_nodes elements

        # ============================================================================
        # NEW: Normalize by total number of points (time slices × interior nodes)
        # ============================================================================
        total_points = tf.cast(num_time_slices * num_interior_spatial_nodes, tf.float32)
        latent_loss = latent_consistency_loss_accum / total_points
        recon_loss = reconstruction_loss_accum / total_points

        total_loss = latent_loss + alpha_recon * recon_loss + spd_loss
        return total_loss, latent_loss, recon_loss, spd_loss


        # ============================================================================
        # SUMMARY OF CHANGES:
        # ============================================================================
        # 
        # OLD (Center-only):
        # - Compute PDE solution for N interior points
        # - Compare only 1 point (center) per time slice
        # - Call encoder 1× per time slice
        # - Call decoder 1× per time slice
        # - Normalize loss by number of time slices
        # 
        # NEW (Full interior):
        # - Compute PDE solution for N interior points
        # - Compare ALL N points per time slice  ✓
        # - Call encoder 1× per time slice (batch all N points)  ✓
        # - Call decoder 1× per time slice (batch all N points)  ✓
        # - Normalize loss by (time slices × interior nodes)  ✓
        # 
        # Benefits:
        # - 100% data efficiency (use all computed solutions)
        # - N× stronger gradient signal
        # - Likely faster convergence (fewer epochs needed)
        # - More robust learning
        # 
        # Cost:
        # - ~N× more memory in gradient tape (still manageable for small patches)
        # - Slightly slower per epoch but FASTER overall
        # ============================================================================

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

        loss_history = {
            'total': [],
            'latent': [],
            'recon': [],
            'spd': []
        }

        for epoch in range(epochs):
            # Build patches once per epoch
            self.create_patch_centres_and_indices_and_values_and_boundary_splits(
                patch_dim=patch_dim,
                num_patches=num_patches
            )

            epoch_total_loss = 0.0
            epoch_latent_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_spd_loss = 0.0

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

                # Accumulate all loss components
                epoch_total_loss += float(total_loss.numpy())
                epoch_latent_loss += float(latent_loss.numpy())
                epoch_recon_loss += float(recon_loss.numpy())
                epoch_spd_loss += float(spd_loss.numpy())

            # Average losses over patches
            epoch_total_loss /= float(num_patches)
            epoch_latent_loss /= float(num_patches)
            epoch_recon_loss /= float(num_patches)
            epoch_spd_loss /= float(num_patches)
            
            # Store in history
            loss_history['total'].append(epoch_total_loss)
            loss_history['latent'].append(epoch_latent_loss)
            loss_history['recon'].append(epoch_recon_loss)
            loss_history['spd'].append(epoch_spd_loss)

            print(f"epoch {epoch+1:04d} | total {epoch_total_loss:.6e} | "
                  f"latent {epoch_latent_loss:.6e} | recon {epoch_recon_loss:.6e} | "
                  f"spd {epoch_spd_loss:.6e}")

        return loss_history
    
    def plot_training_history(self, loss_history, save_path=None, show=True):
        """
        Plot training loss history showing all loss components.
        
        Args:
            loss_history: Dictionary with keys 'total', 'latent', 'recon', 'spd'
                         containing lists of loss values per epoch
            save_path: Optional path to save the figure (e.g., 'loss_plot.png')
            show: Whether to display the plot (default True)
        
        Returns:
            matplotlib figure object
        """
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(loss_history['total']) + 1)
        
        # Create figure with good size
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each loss component
        ax.plot(epochs, loss_history['total'], 'k-', linewidth=2, label='Total Loss', marker='o')
        ax.plot(epochs, loss_history['latent'], 'b--', linewidth=1.5, label='Latent Loss', marker='s')
        ax.plot(epochs, loss_history['recon'], 'r--', linewidth=1.5, label='Reconstruction Loss', marker='^')
        ax.plot(epochs, loss_history['spd'], 'g--', linewidth=1.5, label='SPD Loss', marker='d')
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss History', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')  # Log scale often better for loss visualization
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        return fig
        
    # FIXED reconstruct_field_at_timestep method
    # The issue: self.mask_boundary[0, :, :] is marking ALL points as boundary
    # Fix: Use proper spatial interior mask that doesn't include time boundaries

    def reconstruct_field_at_timestep(self, t_index):
        """
        Reconstruct the entire field at a given timestep using the trained model.
        
        FIXED: Properly handles spatial vs temporal boundaries
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import spsolve
        
        t = int(t_index)
        
        print(f"\nReconstructing field at timestep t={t}...")
        
        # ========================================================================
        # FIX: Build spatial boundary mask correctly
        # The issue: self.mask_boundary includes temporal boundaries (t=0, t=T)
        # We only want SPATIAL boundaries for a single time slice
        # ========================================================================
        
        # Get the spatial boundary thickness from split_interior_boundary
        # Default is typically b_thick = 1
        b_thick = 1  # Assuming this was your setting
        
        # Build spatial interior mask (2D - no time dimension)
        spatial_interior_mask = np.zeros((self.ny, self.nx), dtype=bool)
        spatial_interior_mask[b_thick:-b_thick, b_thick:-b_thick] = True
        
        spatial_boundary_mask = ~spatial_interior_mask
        
        print(f"  DEBUG: Grid shape: ({self.ny}, {self.nx})")
        print(f"  DEBUG: Boundary thickness: {b_thick}")
        print(f"  DEBUG: mask_boundary shape: {self.mask_boundary.shape}")
        print(f"  DEBUG: mask_boundary[{t}] has {np.sum(self.mask_boundary[t])} boundary points")
        
        # Get interior and boundary indices
        interior_indices = np.argwhere(spatial_interior_mask)  # (N_interior, 2)
        boundary_indices = np.argwhere(spatial_boundary_mask)  # (N_boundary, 2)
        
        num_interior = interior_indices.shape[0]
        num_boundary = boundary_indices.shape[0]
        
        print(f"  Interior points: {num_interior}")
        print(f"  Boundary points: {num_boundary}")
        
        if num_interior == 0:
            raise ValueError(f"No interior points! Check your boundary thickness (b_thick={b_thick}). "
                            f"Grid size is ({self.ny}, {self.nx}), which may be too small for b_thick={b_thick}")
        
        # ========================================================================
        # STEP 1: Encode boundary conditions
        # ========================================================================
        print("  Step 1: Encoding boundary conditions...")
        
        # Get boundary features
        boundary_y = boundary_indices[:, 0]
        boundary_x = boundary_indices[:, 1]
        boundary_t = np.full(num_boundary, t, dtype=np.int32)
        
        t_norm = boundary_t.astype(np.float32) / self.nt_norm
        y_norm = boundary_y.astype(np.float32) / self.ny_norm
        x_norm = boundary_x.astype(np.float32) / self.nx_norm
        u_boundary = self.U[boundary_t, boundary_y, boundary_x].astype(np.float32)
        
        boundary_features = np.stack([t_norm, y_norm, x_norm, u_boundary], axis=1)
        boundary_features_tf = tf.constant(boundary_features, dtype=tf.float32)
        
        # Encode boundary
        boundary_latents = self.boundary_encoder(boundary_features_tf, training=False)
        boundary_latents_np = boundary_latents.numpy()  # (N_boundary, latent_dim)
        
        latent_dim = boundary_latents_np.shape[1]
        print(f"  Latent dimension: {latent_dim}")
        
        # ========================================================================
        # STEP 2: Build spatial Laplacian operator
        # ========================================================================
        print("  Step 2: Building Laplacian operator...")
        
        # Create mapping from (y,x) to row index
        interior_row_map = -np.ones((self.ny, self.nx), dtype=np.int32)
        for row_id, (y, x) in enumerate(interior_indices):
            interior_row_map[y, x] = row_id
        
        # Build sparse Laplacian matrix
        row_indices = []
        col_indices = []
        values = []
        
        neighbour_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        # Track which interior nodes have boundary neighbors
        boundary_contributions = [[] for _ in range(num_interior)]
        
        for row_id, (y, x) in enumerate(interior_indices):
            # Diagonal (self)
            row_indices.append(row_id)
            col_indices.append(row_id)
            values.append(4.0)
            
            # Neighbors
            for dy, dx in neighbour_steps:
                ny, nx = y + dy, x + dx
                
                if 0 <= ny < self.ny and 0 <= nx < self.nx:
                    neighbour_row = interior_row_map[ny, nx]
                    
                    if neighbour_row >= 0:
                        # Interior neighbor
                        row_indices.append(row_id)
                        col_indices.append(neighbour_row)
                        values.append(-1.0)
                    else:
                        # Boundary neighbor - store for RHS
                        # Find index in boundary_indices
                        boundary_idx = np.where((boundary_indices[:, 0] == ny) & 
                                            (boundary_indices[:, 1] == nx))[0]
                        if len(boundary_idx) > 0:
                            boundary_contributions[row_id].append(int(boundary_idx[0]))
        
        # Create sparse Laplacian
        laplacian_sparse = sp.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(num_interior, num_interior)
        )
        
        # ========================================================================
        # STEP 3: Get A matrix (PDE operator in latent space)
        # ========================================================================
        print("  Step 3: Getting PDE operator (A matrix)...")
        
        A_matrix = 0.5 * (self.a_matrix + tf.transpose(self.a_matrix))
        A_matrix_np = A_matrix.numpy()  # (latent_dim, latent_dim)
        
        # Build stiffness matrix: K = Laplacian ⊗ A
        stiffness = sp.kron(laplacian_sparse, A_matrix_np, format='csr')
        
        print(f"  Stiffness matrix size: {stiffness.shape}")
        
        # ========================================================================
        # STEP 4: Build RHS from boundary conditions
        # ========================================================================
        print("  Step 4: Building right-hand side...")
        
        # RHS is -Laplacian @ boundary contributions
        rhs = np.zeros((num_interior, latent_dim), dtype=np.float32)
        
        for row_id in range(num_interior):
            for boundary_idx in boundary_contributions[row_id]:
                # Contribution from this boundary node
                rhs[row_id, :] += A_matrix_np @ boundary_latents_np[boundary_idx, :]
        
        rhs_flat = rhs.flatten()
        
        # ========================================================================
        # STEP 5: Solve linear system K @ latent_interior = rhs
        # ========================================================================
        print("  Step 5: Solving linear system...")
        
        latent_interior_flat = spsolve(stiffness, rhs_flat)
        latent_interior = latent_interior_flat.reshape((num_interior, latent_dim))
        latent_interior_tf = tf.constant(latent_interior, dtype=tf.float32)
        
        # ========================================================================
        # STEP 6: Decode latents to get interior field
        # ========================================================================
        print("  Step 6: Decoding latents to physical field...")
        
        u_interior_pred = self.decoder(latent_interior_tf, training=False)
        u_interior_pred_np = u_interior_pred.numpy().flatten()
        
        # ========================================================================
        # STEP 7: Assemble full field
        # ========================================================================
        print("  Step 7: Assembling full field...")
        
        # Initialize
        u_pred_full = np.zeros((self.ny, self.nx), dtype=np.float32)
        u_true_full = self.U[t, :, :].astype(np.float32)
        
        # Fill boundary values
        u_pred_full[boundary_y, boundary_x] = u_boundary
        
        # Fill interior values
        interior_y = interior_indices[:, 0]
        interior_x = interior_indices[:, 1]
        u_pred_full[interior_y, interior_x] = u_interior_pred_np
        
        # Compute error
        u_error = np.abs(u_pred_full - u_true_full)
        
        # Compute statistics (interior only)
        interior_error = u_error[interior_y, interior_x]
        mse = np.mean(interior_error ** 2)
        mae = np.mean(interior_error)
        max_error = np.max(interior_error)
        
        print(f"\n  Reconstruction Statistics (interior only):")
        print(f"    MSE: {mse:.6e}")
        print(f"    MAE: {mae:.6e}")
        print(f"    Max Error: {max_error:.6e}")
        
        results = {
            'u_pred': u_pred_full,
            'u_true': u_true_full,
            'u_error': u_error,
            'boundary_mask': spatial_boundary_mask,
            'interior_mask': spatial_interior_mask,
            'mse': mse,
            'mae': mae,
            'max_error': max_error,
            't_index': t
        }
        
        print(f"  Reconstruction complete!\n")
        
        return results
    
    def plot_field_reconstruction(self, results, save_path=None, show=True):
        """
        Plot original field, predicted field, and error field side-by-side.
        
        Args:
            results: Dictionary returned from reconstruct_field_at_timestep()
            save_path: Optional path to save the figure
            show: Whether to display the plot
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        u_true = results['u_true']
        u_pred = results['u_pred']
        u_error = results['u_error']
        boundary_mask = results['boundary_mask']
        t_index = results['t_index']
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Determine common color scale for true and predicted
        vmin = min(u_true.min(), u_pred.min())
        vmax = max(u_true.max(), u_pred.max())
        
        # ========================================================================
        # Plot 1: True field
        # ========================================================================
        ax = axes[0]
        im1 = ax.imshow(u_true, cmap='viridis', origin='lower', 
                        vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'True Field (t={t_index})', fontsize=13, fontweight='bold')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        
        # Mark boundary
        boundary_coords = np.argwhere(boundary_mask)
        ax.scatter(boundary_coords[:, 1], boundary_coords[:, 0], 
                c='red', s=1, alpha=0.3, label='Boundary')
        
        cbar1 = plt.colorbar(im1, ax=ax)
        cbar1.set_label('u', fontsize=11)
        
        # ========================================================================
        # Plot 2: Predicted field
        # ========================================================================
        ax = axes[1]
        im2 = ax.imshow(u_pred, cmap='viridis', origin='lower',
                        vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'Predicted Field (Reconstructed)', fontsize=13, fontweight='bold')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        
        # Mark boundary
        ax.scatter(boundary_coords[:, 1], boundary_coords[:, 0], 
                c='red', s=1, alpha=0.3, label='Boundary')
        
        cbar2 = plt.colorbar(im2, ax=ax)
        cbar2.set_label('u', fontsize=11)
        
        # ========================================================================
        # Plot 3: Error field
        # ========================================================================
        ax = axes[2]
        im3 = ax.imshow(u_error, cmap='hot', origin='lower', aspect='auto')
        ax.set_title(f'Absolute Error', fontsize=13, fontweight='bold')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        
        # Mark boundary
        ax.scatter(boundary_coords[:, 1], boundary_coords[:, 0], 
                c='cyan', s=1, alpha=0.5, label='Boundary')
        
        cbar3 = plt.colorbar(im3, ax=ax)
        cbar3.set_label('|error|', fontsize=11)
        
        # Add error statistics
        ax.text(0.02, 0.98, 
                f"MSE: {results['mse']:.4e}\nMAE: {results['mae']:.4e}\nMax: {results['max_error']:.4e}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ========================================================================
        # Overall title
        # ========================================================================
        plt.suptitle(f'Field Reconstruction at t={t_index}', 
                    fontsize=15, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Field reconstruction plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig


        # ============================================================================
        # EXAMPLE USAGE
        # ============================================================================
        """
        # After training is complete:

        # Reconstruct field at timestep 50
        results = solver.reconstruct_field_at_timestep(t_index=50)

        # Plot the reconstruction
        solver.plot_field_reconstruction(results, 
                                        save_path='field_reconstruction_t50.png',
                                        show=True)

        # Access the results
        print(f"Prediction MSE: {results['mse']:.6e}")
        u_pred = results['u_pred']  # Predicted field (ny, nx)
        u_true = results['u_true']  # True field (ny, nx)
        u_error = results['u_error']  # Error field (ny, nx)
        """


if __name__ == "__main__":
    # Constants and hyperparameters
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
    epochs = 50
    
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
    
    # After solver.build_models(...)
    print("\nBOUNDARY MASK CHECK:")
    print(f"Grid shape: {solver.U.shape}")
    print(f"Boundary mask shape: {solver.mask_boundary.shape}")

    t_mid = solver.nt // 2
    print(f"At t={t_mid}:")
    print(f"  Boundary points: {np.sum(solver.mask_boundary[t_mid])}")
    print(f"  Interior points: {np.sum(~solver.mask_boundary[t_mid])}")
    
    # Train with optional mixed precision (set to True for better GPU performance)
    loss_history = solver.train(epochs, patch_dim, num_patches, use_mixed_precision=False)
    
    print("\nTraining complete!")
    print(f"Final total loss: {loss_history['total'][-1]:.6e}")
    print(f"Final latent loss: {loss_history['latent'][-1]:.6e}")
    print(f"Final recon loss: {loss_history['recon'][-1]:.6e}")
    print(f"Final spd loss: {loss_history['spd'][-1]:.6e}")
    
    # Plot training history
    solver.plot_training_history(loss_history, save_path='training_loss.png', show=True)
    
    # Reconstruct field at timestep 50
    results = solver.reconstruct_field_at_timestep(t_index=50)

    # Visualize
    solver.plot_field_reconstruction(results, 
                                    save_path='field_t50.png',
                                    show=True)

    # Check quality
    print(f"Reconstruction MAE: {results['mae']:.6e}")