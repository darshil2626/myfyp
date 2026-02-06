import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.layers import Dense, Input
from keras.models import Model

import pickle


class sinn():
    def __init__(self, X, Y, U, T):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.U = np.asarray(U)
        self.T = np.asarray(T)
        self.nt, self.ny, self.nx = self.U.shape
        
    def standardise_u(self):
        self.U_mean = np.mean(self.U, axis=0)
        self.U_std = np.std(self.U, axis=0)
        self.U = (self.U - self.U_mean) / self.U_std
    
    def unstandardise_u(self, U_pred):
        return U_pred * self.U_std + self.U_mean
        
    def split_interior_boundary(self, b_thick, include_t0 = True, include_tT = True):
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
        self.mask_interior = mask_int
        self.mask_fd_safe_interior = safe & mask_int

        self.boundary_idx = np.argwhere(self.mask_boundary)          # (Nb,3)
        self.interior_idx = np.argwhere(self.mask_interior)          # (Ni,3)
        self.fd_safe_interior_idx = np.argwhere(self.mask_fd_safe_interior)  # (Nsafe,3)
    
    def build_models(self, num_latentdim, num_units, num_layers, dropout, l2_reg, lr):
        self.interior_encoder = build_coder(num_units, num_layers, input_shape=4, output_shape=num_latentdim, name='interior encoder', dropout=dropout, l2_reg=l2_reg)
        self.boundary_encoder = build_coder(num_units, num_layers, input_shape=4, output_shape=num_latentdim, name='boundary encoder', dropout=dropout, l2_reg=l2_reg)  
        self.decoder = build_coder(num_units, num_layers, input_shape=num_latentdim, output_shape=1, name='decoder', dropout=dropout, l2_reg=l2_reg)
        
        # Initialize A_matrix
        self.a_matrix = tf.Variable(np.eye(num_latentdim, dtype=np.float32), trainable=True, dtype=tf.float32, name='pde_operator')
        
        self.trainable_vars = (self.interior_encoder.trainable_variables + self.boundary_encoder.trainable_variables + self.decoder.trainable_variables + [self.a_matrix])
        
        print(f"Trainable variables: {len(self.trainable_vars)}")
        
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        
    def create_patch_centres_and_indices_and_values_and_boundary_splits(self, patch_dim, num_patches):
        px, py, pt = int(patch_dim[0]), int(patch_dim[1]), int(patch_dim[2])
        x_half, y_half, t_half = px // 2, py // 2, pt // 2

        # Centres so patch box is in-bounds
        x_min, x_max = x_half, self.nx - x_half - 1
        y_min, y_max = y_half, self.ny - y_half - 1
        t_min, t_max = t_half, self.nt - t_half - 1

        # Preallocate python lists (append is also fine, but this is tidy)
        self.patch_center_idx = [None] * num_patches

        self.patch_interior_idx = [None] * num_patches
        self.patch_boundary_idx = [None] * num_patches

        self.patch_interior_values = [None] * num_patches
        self.patch_boundary_values = [None] * num_patches

        self.patch_boundary_global_boundary_idx = [None] * num_patches
        self.patch_boundary_global_interior_idx = [None] * num_patches

        self.patch_boundary_global_boundary_values = [None] * num_patches
        self.patch_boundary_global_interior_values = [None] * num_patches

        # Small local mask for "boundary of patch volume" in (t,y,x) local coordinates
        # This is constant for all patches of same pt,py,px, so build it once.
        patch_boundary_mask_local = np.zeros((pt, py, px), dtype=bool)
        patch_boundary_mask_local[0, :, :] = True
        patch_boundary_mask_local[-1, :, :] = True
        patch_boundary_mask_local[:, 0, :] = True
        patch_boundary_mask_local[:, -1, :] = True
        patch_boundary_mask_local[:, :, 0] = True
        patch_boundary_mask_local[:, :, -1] = True

        # Local interior-of-patch mask (strict interior)
        patch_strict_interior_mask_local = ~patch_boundary_mask_local

        for k in range(num_patches):
            # sample patch centre
            cx = np.random.randint(x_min, x_max + 1)
            cy = np.random.randint(y_min, y_max + 1)
            ct = np.random.randint(t_min, t_max + 1)

            patch_center_idx_tyx = np.array([ct, cy, cx], dtype=np.int32)
            self.patch_center_idx[k] = patch_center_idx_tyx

            # ---------------------------
            # 2) patch bounds (half-open)
            # ---------------------------
            x_start, x_end = cx - x_half, cx + x_half + 1
            y_start, y_end = cy - y_half, cy + y_half + 1
            t_start, t_end = ct - t_half, ct + t_half + 1

            # ---------------------------
            # 3) slice masks to patch subvolume
            # ---------------------------
            # FD-safe interior mask restricted to this patch volume
            mask_fd_safe_interior_subvolume = self.mask_fd_safe_interior[t_start:t_end, y_start:y_end, x_start:x_end]
            # Global boundary mask restricted to this patch volume
            mask_global_boundary_subvolume = self.mask_boundary[t_start:t_end, y_start:y_end, x_start:x_end]

            # ---------------------------
            # 4) patch boundary indices (∂Q) from local mask -> global indices
            # ---------------------------
            patch_boundary_local_indices_tyx = np.argwhere(patch_boundary_mask_local).astype(np.int32)  # (Nb,3) local (t,y,x)

            patch_boundary_idx_tyx = patch_boundary_local_indices_tyx.copy()
            patch_boundary_idx_tyx[:, 0] += t_start
            patch_boundary_idx_tyx[:, 1] += y_start
            patch_boundary_idx_tyx[:, 2] += x_start

            patch_boundary_values = self.U[
                patch_boundary_idx_tyx[:, 0],
                patch_boundary_idx_tyx[:, 1],
                patch_boundary_idx_tyx[:, 2],
            ].astype(np.float32)

            # ---------------------------
            # 5) strict patch interior indices (inside box, exclude ∂Q) AND FD-safe
            # ---------------------------
            # interior points are those that are strictly inside patch AND FD-safe interior
            mask_patch_interior_and_fd_safe = patch_strict_interior_mask_local & mask_fd_safe_interior_subvolume
            patch_interior_local_indices_tyx = np.argwhere(mask_patch_interior_and_fd_safe).astype(np.int32)  # (Ni,3)

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

            # ---------------------------
            # 6) split patch boundary into global-boundary vs global-interior
            # ---------------------------
            # Use the sliced global-boundary mask, but only on patch boundary points.
            boundary_local_t = patch_boundary_local_indices_tyx[:, 0]
            boundary_local_y = patch_boundary_local_indices_tyx[:, 1]
            boundary_local_x = patch_boundary_local_indices_tyx[:, 2]

            global_boundary_mask_on_patch_boundary = mask_global_boundary_subvolume[
                boundary_local_t, boundary_local_y, boundary_local_x
            ]  # (Nb,) bool

            patch_boundary_global_boundary_idx_tyx = patch_boundary_idx_tyx[global_boundary_mask_on_patch_boundary]
            patch_boundary_global_interior_idx_tyx = patch_boundary_idx_tyx[~global_boundary_mask_on_patch_boundary]

            patch_boundary_global_boundary_values = patch_boundary_values[global_boundary_mask_on_patch_boundary]
            patch_boundary_global_interior_values = patch_boundary_values[~global_boundary_mask_on_patch_boundary]

            # ---------------------------
            # 7) store
            # ---------------------------
            self.patch_boundary_idx[k] = patch_boundary_idx_tyx
            self.patch_boundary_values[k] = patch_boundary_values

            self.patch_interior_idx[k] = patch_interior_idx_tyx
            self.patch_interior_values[k] = patch_interior_values

            self.patch_boundary_global_boundary_idx[k] = patch_boundary_global_boundary_idx_tyx
            self.patch_boundary_global_boundary_values[k] = patch_boundary_global_boundary_values

            self.patch_boundary_global_interior_idx[k] = patch_boundary_global_interior_idx_tyx
            self.patch_boundary_global_interior_values[k] = patch_boundary_global_interior_values


            
    def stack_features_from_idx(self, idx_tyx):
        """
        idx_tyx: (N,3) int array [t,y,x]
        returns: (N,4) float32 -> matches build_models(input_shape=4) :contentReference[oaicite:2]{index=2}

        features = [t_norm, y_norm, x_norm, u(t,y,x)]
        """
        idx_tyx = np.asarray(idx_tyx, dtype=np.int32)
        if idx_tyx.size == 0:
            return np.zeros((0, 4), dtype=np.float32)

        t = idx_tyx[:, 0]
        y = idx_tyx[:, 1]
        x = idx_tyx[:, 2]

        t_norm = t.astype(np.float32) / (self.nt - 1) if self.nt > 1 else t.astype(np.float32)
        y_norm = y.astype(np.float32) / (self.ny - 1) if self.ny > 1 else y.astype(np.float32)
        x_norm = x.astype(np.float32) / (self.nx - 1) if self.nx > 1 else x.astype(np.float32)

        u_val = self.U[t, y, x].astype(np.float32)

        return np.stack([t_norm, y_norm, x_norm, u_val], axis=1).astype(np.float32)
    
    def compute_pde_loss(
        self,
        patch_center_spatial_y_index: int,
        patch_center_spatial_x_index: int,
        patch_time_indices_in_patch: np.ndarray,  # (Nt_patch,) sorted unique t indices
        patch_boundary_idx_tyx: np.ndarray,       # (Nb,3) [t,y,x] patch boundary points
        latent_values_on_patch_boundary_aligned_with_patch_boundary_idx: tf.Tensor,  # (Nb,r)
        rho_spd: float = 1e-3,
        alpha_recon: float = 1.0,
        spd_epsilon: float = 1e-6,
    ):
        """
        Implements pseudocode steps 4-7:
        - per-time-slice elliptic solve on spatial patch using Dirichlet latent BCs from ∂Qk
        - latent consistency loss at patch centre for all time slices
        - decode predicted latent at centre and compare to true u at centre
        - SPD regularisation on A

        Returns:
        total_loss, latent_loss, recon_loss, spd_loss (all tf.Tensors)
        """

        # -----------------------------
        # Basic numpy hygiene
        # -----------------------------
        patch_boundary_idx_tyx = np.asarray(patch_boundary_idx_tyx, dtype=np.int32)
        patch_time_indices_in_patch = np.asarray(patch_time_indices_in_patch, dtype=np.int32)

        # Infer patch spatial bounds from boundary indices (works because boundary covers all faces)
        y_min = int(patch_boundary_idx_tyx[:, 1].min())
        y_max = int(patch_boundary_idx_tyx[:, 1].max())
        x_min = int(patch_boundary_idx_tyx[:, 2].min())
        x_max = int(patch_boundary_idx_tyx[:, 2].max())

        patch_spatial_height = (y_max - y_min + 1)
        patch_spatial_width  = (x_max - x_min + 1)

        # Local (y,x) interior mask for the spatial patch: strict interior (exclude patch spatial boundary)
        # This is the spatial analogue of ∂P_k / interior nodes in pseudocode step 4A. :contentReference[oaicite:7]{index=7}
        spatial_strict_interior_mask = np.zeros((patch_spatial_height, patch_spatial_width), dtype=bool)
        if patch_spatial_height >= 3 and patch_spatial_width >= 3:
            spatial_strict_interior_mask[1:-1, 1:-1] = True

        # Build list of interior spatial nodes (local coords) and a mapping (local_y,local_x) -> row id
        interior_spatial_local_yx = np.argwhere(spatial_strict_interior_mask).astype(np.int32)  # (Nint,2)
        num_interior_spatial_nodes = int(interior_spatial_local_yx.shape[0])

        # If patch is too small to have interior nodes, return something that won’t crash training
        if num_interior_spatial_nodes == 0:
            latent_loss = tf.constant(0.0, tf.float32)
            recon_loss  = tf.constant(0.0, tf.float32)
            spd_loss    = tf.constant(0.0, tf.float32)
            total_loss  = latent_loss + alpha_recon * recon_loss + spd_loss
            return total_loss, latent_loss, recon_loss, spd_loss

        interior_spatial_row_id = -np.ones((patch_spatial_height, patch_spatial_width), dtype=np.int32)
        for row_id, (ly, lx) in enumerate(interior_spatial_local_yx):
            interior_spatial_row_id[ly, lx] = row_id

        # Identify which interior row corresponds to the patch centre (must be a strict interior node)
        patch_center_local_y = int(patch_center_spatial_y_index - y_min)
        patch_center_local_x = int(patch_center_spatial_x_index - x_min)

        centre_row_id = int(interior_spatial_row_id[patch_center_local_y, patch_center_local_x])
        if centre_row_id < 0:
            # centre lies on patch spatial boundary (shouldn't happen if you choose centres correctly)
            latent_loss = tf.constant(0.0, tf.float32)
            recon_loss  = tf.constant(0.0, tf.float32)
            spd_loss    = tf.constant(0.0, tf.float32)
            total_loss  = latent_loss + alpha_recon * recon_loss + spd_loss
            return total_loss, latent_loss, recon_loss, spd_loss

        # -----------------------------
        # Assemble the spatial Laplacian stencil matrix once (Step 4A) :contentReference[oaicite:8]{index=8}
        # Using operator: 4*ell(i) - sum(ell(neighbours)) = sum(boundary neighbours)
        # -----------------------------
        # Build dense (Nint x Nint) for simplicity (patch sizes small). You can sparsify later.
        laplacian_operator_matrix = np.zeros((num_interior_spatial_nodes, num_interior_spatial_nodes), dtype=np.float32)

        # Precompute for each interior node: which neighbour locations are boundary vs interior
        # We'll also store the neighbour (local_y,local_x) pairs for boundary contributions.
        boundary_neighbour_local_yx_per_row = [[] for _ in range(num_interior_spatial_nodes)]

        # 4-neighbour connectivity
        neighbour_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for row_id, (ly, lx) in enumerate(interior_spatial_local_yx):
            # Diagonal coefficient
            laplacian_operator_matrix[row_id, row_id] += 4.0

            for dy, dx in neighbour_steps:
                nly = int(ly + dy)
                nlx = int(lx + dx)

                # neighbour is inside patch spatial box by construction
                neighbour_row = int(interior_spatial_row_id[nly, nlx])

                if neighbour_row >= 0:
                    # neighbour is an interior unknown -> move to LHS
                    laplacian_operator_matrix[row_id, neighbour_row] += -1.0
                else:
                    # neighbour is on patch spatial boundary -> contributes to RHS via Dirichlet value
                    boundary_neighbour_local_yx_per_row[row_id].append((nly, nlx))

        laplacian_operator_matrix_tf = tf.convert_to_tensor(laplacian_operator_matrix, dtype=tf.float32)

        # -----------------------------
        # A matrix (latent coupling) and SPD regularisation (Step 7) :contentReference[oaicite:9]{index=9}
        # -----------------------------
        # Symmetrise for safety
        latent_operator_matrix = 0.5 * (self.a_matrix + tf.transpose(self.a_matrix))
        latent_dim = int(latent_operator_matrix.shape[0])

        # SPD penalty via eigenvalues (stable; avoids NaNs from logdet if not SPD)
        eigenvalues = tf.linalg.eigvalsh(latent_operator_matrix)
        spd_violation = tf.nn.relu(spd_epsilon - eigenvalues)
        spd_loss = rho_spd * tf.reduce_sum(spd_violation * spd_violation)

        # Build Kronecker-style block system: K = (Laplacian) ⊗ A
        # Dense kron is fine for small patches.
        stiffness_matrix_tf = tf.linalg.LinearOperatorKronecker([
            tf.linalg.LinearOperatorFullMatrix(laplacian_operator_matrix_tf),
            tf.linalg.LinearOperatorFullMatrix(latent_operator_matrix),
        ]).to_dense()  # shape (Nint*r, Nint*r)

        # -----------------------------
        # Step 5 & 6 accumulators over time slices :contentReference[oaicite:10]{index=10}
        # -----------------------------
        latent_consistency_loss_accum = tf.constant(0.0, tf.float32)
        reconstruction_loss_accum = tf.constant(0.0, tf.float32)

        num_time_slices = int(patch_time_indices_in_patch.shape[0])

        # We'll compare at the spatial patch centre for each time t_n:
        # ℓ_true(t_n) from interior encoder at (t_n, cy, cx)
        # ℓ_pde(t_n) from PDE solve at the same point.
        patch_centre_time_yx_indices = np.stack(
            [patch_time_indices_in_patch,
            np.full((num_time_slices,), patch_center_spatial_y_index, dtype=np.int32),
            np.full((num_time_slices,), patch_center_spatial_x_index, dtype=np.int32)],
            axis=1
        )  # (Nt_patch, 3) [t,y,x]

        # Build your current 4D features [t_norm, y_norm, x_norm, u] for these centre points
        # (Uses your standard “input_shape=4” scheme.) :contentReference[oaicite:11]{index=11}
        t = patch_centre_time_yx_indices[:, 0]
        y = patch_centre_time_yx_indices[:, 1]
        x = patch_centre_time_yx_indices[:, 2]
        t_norm = t.astype(np.float32) / (self.nt - 1) if self.nt > 1 else t.astype(np.float32)
        y_norm = y.astype(np.float32) / (self.ny - 1) if self.ny > 1 else y.astype(np.float32)
        x_norm = x.astype(np.float32) / (self.nx - 1) if self.nx > 1 else x.astype(np.float32)
        u_val  = self.U[t, y, x].astype(np.float32)

        centre_features = np.stack([t_norm, y_norm, x_norm, u_val], axis=1).astype(np.float32)  # (Nt_patch,4)
        centre_features_tf = tf.convert_to_tensor(centre_features, dtype=tf.float32)

        latent_true_at_patch_centre_all_times = self.interior_encoder(centre_features_tf, training=True)  # (Nt_patch, r)

        # True physical centre values (for recon loss)
        u_true_at_patch_centre_all_times = tf.convert_to_tensor(u_val.reshape(-1, 1), dtype=tf.float32)  # (Nt_patch,1)

        # -----------------------------
        # Step 4B: Solve per time slice :contentReference[oaicite:12]{index=12}
        # -----------------------------
        # We'll need to pick out boundary latent Dirichlet values for each time slice.
        patch_boundary_t = patch_boundary_idx_tyx[:, 0]
        patch_boundary_y = patch_boundary_idx_tyx[:, 1]
        patch_boundary_x = patch_boundary_idx_tyx[:, 2]

        # Make sure boundary latents are aligned with patch_boundary_idx_tyx (caller guarantees this)
        boundary_latents_tf = latent_values_on_patch_boundary_aligned_with_patch_boundary_idx  # (Nb, r)

        for time_list_index, t_n in enumerate(patch_time_indices_in_patch):
            # Select boundary points that are on this time slice
            boundary_mask_this_time = (patch_boundary_t == int(t_n))
            boundary_indices_this_time = np.nonzero(boundary_mask_this_time)[0].astype(np.int32)

            # If there are no boundary nodes on this time slice (shouldn't happen for a box), skip
            if boundary_indices_this_time.size == 0:
                continue

            boundary_y_this_time = patch_boundary_y[boundary_indices_this_time]
            boundary_x_this_time = patch_boundary_x[boundary_indices_this_time]
            boundary_latents_this_time = tf.gather(boundary_latents_tf, boundary_indices_this_time, axis=0)  # (Nb_t, r)

            # Build a quick lookup from (y,x) -> latent vector for this time slice boundary nodes
            # Since patch boundary nodes cover full spatial boundary, this is safe.
            # We'll do it in numpy for indices, and tf.gather for values.
            # Map local boundary coords to row in boundary_indices_this_time
            boundary_lookup = -np.ones((patch_spatial_height, patch_spatial_width), dtype=np.int32)
            for j, (yy, xx) in enumerate(zip(boundary_y_this_time, boundary_x_this_time)):
                boundary_lookup[int(yy - y_min), int(xx - x_min)] = j

            # Assemble RHS: rhs is (Nint * r, 1)
            rhs_blocks = []

            for row_id, (ly, lx) in enumerate(interior_spatial_local_yx):
                rhs_r = tf.zeros((latent_dim,), dtype=tf.float32)

                # For each boundary neighbour of this interior node, add A @ ℓ_bdy(neighbour)
                for (nly, nlx) in boundary_neighbour_local_yx_per_row[row_id]:
                    boundary_j = int(boundary_lookup[nly, nlx])
                    # If boundary_j == -1, that neighbour isn’t on this time slice boundary list (shouldn't happen)
                    if boundary_j >= 0:
                        neighbour_latent = boundary_latents_this_time[boundary_j, :]  # (r,)
                        rhs_r = rhs_r + tf.linalg.matvec(latent_operator_matrix, neighbour_latent)  # A * ℓ_bdy

                rhs_blocks.append(rhs_r)

            rhs_vector_tf = tf.concat([tf.reshape(v, (latent_dim, 1)) for v in rhs_blocks], axis=0)  # (Nint*r,1)

            # Solve linear system: K * L = rhs  (Step 4B) :contentReference[oaicite:13]{index=13}
            latent_solution_vector_tf = tf.linalg.solve(stiffness_matrix_tf, rhs_vector_tf)  # (Nint*r,1)
            latent_solution_vector_tf = tf.reshape(latent_solution_vector_tf, (num_interior_spatial_nodes, latent_dim))  # (Nint, r)

            # Extract PDE-predicted latent at spatial patch centre (local interior row)
            latent_pred_at_centre_this_time = latent_solution_vector_tf[centre_row_id, :]  # (r,)

            # Compare vs encoder "true" latent at that same centre/time (Step 5) :contentReference[oaicite:14]{index=14}
            latent_true_at_centre_this_time = latent_true_at_patch_centre_all_times[time_list_index, :]  # (r,)
            latent_consistency_loss_accum += tf.reduce_sum(
                tf.square(latent_pred_at_centre_this_time - latent_true_at_centre_this_time)
            )

            # Decode predicted latent to reconstruct u at centre (Step 6) :contentReference[oaicite:15]{index=15}
            u_pred_at_centre_this_time = self.decoder(
                tf.reshape(latent_pred_at_centre_this_time, (1, latent_dim)),
                training=True
            )  # (1,1)
            u_true_at_centre_this_time = tf.reshape(u_true_at_patch_centre_all_times[time_list_index, :], (1, 1))

            reconstruction_loss_accum += tf.reduce_sum(
                tf.square(u_pred_at_centre_this_time - u_true_at_centre_this_time)
            )

        # Average over number of time slices that actually contributed
        denom = tf.cast(tf.maximum(1, num_time_slices), tf.float32)
        latent_loss = latent_consistency_loss_accum / denom
        recon_loss  = reconstruction_loss_accum / denom

        total_loss = latent_loss + alpha_recon * recon_loss + spd_loss
        return total_loss, latent_loss, recon_loss, spd_loss
    

    def train(self, epochs, patch_dim, num_patches):
        """
        Train loop using:
        - create_patch_centres_and_indices_and_values_and_boundary_splits(...)
        - routing patch-boundary points by global-boundary membership:
                global boundary -> boundary_encoder
                global interior -> interior_encoder
        """

        loss_history = []

        for epoch in range(epochs):

            # Build a fresh set of random patch centres + indices + values + boundary splits
            self.create_patch_centres_and_indices_and_values_and_boundary_splits(
                patch_dim=patch_dim,
                num_patches=num_patches
            )

            epoch_total_loss_value = 0.0

            for patch_k in range(num_patches):

                # ---- fetch this patch’s precomputed data
                patch_center_idx_tyx = self.patch_center_idx[patch_k]                       # (3,)
                patch_interior_idx_tyx = self.patch_interior_idx[patch_k]                   # (Ni,3)
                patch_boundary_idx_tyx = self.patch_boundary_idx[patch_k]                   # (Nb,3)

                patch_boundary_global_boundary_idx_tyx = self.patch_boundary_global_boundary_idx[patch_k]  # (Nb_gb,3)
                patch_boundary_global_interior_idx_tyx = self.patch_boundary_global_interior_idx[patch_k]  # (Nb_gi,3)

                # (Optional) you still have values lists too:
                patch_interior_values = self.patch_interior_values[patch_k]
                patch_boundary_values = self.patch_boundary_values[patch_k]

                # ---- build encoder inputs (input_shape=4) from indices + gathered U
                patch_center_features = self.stack_features_from_idx(patch_center_idx_tyx[None, :])   # (1,4)
                patch_interior_features = self.stack_features_from_idx(patch_interior_idx_tyx)        # (Ni,4)

                patch_boundary_features_all = self.stack_features_from_idx(patch_boundary_idx_tyx)    # (Nb,4)
                patch_boundary_features_global_boundary = self.stack_features_from_idx(patch_boundary_global_boundary_idx_tyx)  # (Nb_gb,4)
                patch_boundary_features_global_interior = self.stack_features_from_idx(patch_boundary_global_interior_idx_tyx)  # (Nb_gi,4)

                with tf.GradientTape() as tape:

                    # ============================================================
                    # 1) "True" latent at patch centre (interior encoder)
                    # ============================================================
                    latent_at_patch_center_from_interior_encoder = self.interior_encoder(
                        tf.convert_to_tensor(patch_center_features, dtype=tf.float32),
                        training=True
                    )  # (1,r)

                    # ============================================================
                    # 2) Latents on the PATCH boundary ∂Q, routed by GLOBAL boundary
                    #    - global boundary points -> boundary_encoder
                    #    - global interior points -> interior_encoder
                    # ============================================================
                    latent_on_patch_boundary_from_boundary_encoder = self.boundary_encoder(
                        tf.convert_to_tensor(patch_boundary_features_global_boundary, dtype=tf.float32),
                        training=True
                    )  # (Nb_gb, r)

                    latent_on_patch_boundary_from_interior_encoder = self.interior_encoder(
                        tf.convert_to_tensor(patch_boundary_features_global_interior, dtype=tf.float32),
                        training=True
                    )  # (Nb_gi, r)

                    # Stitch the two routed outputs back into a single tensor aligned with patch_boundary_idx_tyx
                    # (We re-compute the boolean mask aligned to patch_boundary_idx_tyx; cheap and robust.)
                    if patch_boundary_idx_tyx.shape[0] > 0:
                        tb = patch_boundary_idx_tyx[:, 0]
                        yb = patch_boundary_idx_tyx[:, 1]
                        xb = patch_boundary_idx_tyx[:, 2]
                        global_boundary_mask_aligned_with_patch_boundary = self.mask_boundary[tb, yb, xb]  # (Nb,) bool :contentReference[oaicite:3]{index=3}

                        indices_in_patch_boundary_list_routed_to_boundary_encoder = np.nonzero(
                            global_boundary_mask_aligned_with_patch_boundary
                        )[0].astype(np.int32)

                        indices_in_patch_boundary_list_routed_to_interior_encoder = np.nonzero(
                            ~global_boundary_mask_aligned_with_patch_boundary
                        )[0].astype(np.int32)

                        # Determine latent dimension r
                        if int(latent_on_patch_boundary_from_boundary_encoder.shape[0]) > 0:
                            latent_dim = int(latent_on_patch_boundary_from_boundary_encoder.shape[-1])
                        else:
                            latent_dim = int(latent_on_patch_boundary_from_interior_encoder.shape[-1])

                        latent_on_patch_boundary_aligned_with_patch_boundary_idx = tf.zeros(
                            (patch_boundary_idx_tyx.shape[0], latent_dim),
                            dtype=tf.float32
                        )

                        if indices_in_patch_boundary_list_routed_to_boundary_encoder.size > 0:
                            latent_on_patch_boundary_aligned_with_patch_boundary_idx = tf.tensor_scatter_nd_update(
                                latent_on_patch_boundary_aligned_with_patch_boundary_idx,
                                indices=tf.constant(indices_in_patch_boundary_list_routed_to_boundary_encoder)[:, None],
                                updates=latent_on_patch_boundary_from_boundary_encoder
                            )

                        if indices_in_patch_boundary_list_routed_to_interior_encoder.size > 0:
                            latent_on_patch_boundary_aligned_with_patch_boundary_idx = tf.tensor_scatter_nd_update(
                                latent_on_patch_boundary_aligned_with_patch_boundary_idx,
                                indices=tf.constant(indices_in_patch_boundary_list_routed_to_interior_encoder)[:, None],
                                updates=latent_on_patch_boundary_from_interior_encoder
                            )
                    else:
                        # empty boundary (shouldn't happen for a proper box)
                        latent_on_patch_boundary_aligned_with_patch_boundary_idx = tf.zeros((0, int(self.a_matrix.shape[0])), dtype=tf.float32)

                    # ============================================================
                    # 3) Latent PDE solve + losses
                    # ============================================================
                    # NOTE: Your current compute_pde_loss(...) call in sinn_solver.py :contentReference[oaicite:4]{index=4}
                    # is still placeholder-ish for the real SINN patch elliptic solve.
                    #
                    # For now, keep the same pattern, but pass the *latents* and the
                    # *features* you’re using (not raw 1D values).
                    #
                    # You will likely update compute_pde_loss to:
                    #   - build/solve the latent elliptic system on the patch (per time slice),
                    #   - compare predicted centre latent to latent_at_patch_center_from_interior_encoder,
                    #   - optionally decode predicted latent & compare reconstruction.
                    #
                    # patch_center_idx_tyx = np.array([ct, cy, cx])
                    patch_time_indices_in_patch = np.unique(self.patch_boundary_idx[patch_k][:, 0])  # all t in this patch box

                    total_loss, latent_loss, recon_loss, spd_loss = self.compute_pde_loss(
                        patch_center_spatial_y_index=int(patch_center_idx_tyx[1]),
                        patch_center_spatial_x_index=int(patch_center_idx_tyx[2]),
                        patch_time_indices_in_patch=patch_time_indices_in_patch,
                        patch_boundary_idx_tyx=patch_boundary_idx_tyx,
                        latent_values_on_patch_boundary_aligned_with_patch_boundary_idx=latent_on_patch_boundary_aligned_with_patch_boundary_idx,
                        rho_spd=1e-3,
                        alpha_recon=1.0,
                    )


                grads = tape.gradient(total_loss, self.trainable_vars)
                self.optimizer.apply_gradients(zip(grads, self.trainable_vars))

                epoch_total_loss_value += float(total_loss.numpy())

            epoch_total_loss_value /= float(num_patches)
            loss_history.append(epoch_total_loss_value)

            print(f"epoch {epoch+1:04d} | loss {epoch_total_loss_value:.6e}")

        return loss_history

            
            

    
    
        
def build_coder(num_units, num_layers, input_shape, output_shape, name, dropout, l2_reg):
        inputs = Input(shape=(input_shape,))
        x = Dense(num_units, activation='tanh', kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
        for _ in range(num_layers - 1):
            x = Dense(num_units, activation='tanh', kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        if dropout and dropout > 0.0:
            x = layers.Dropout(dropout)(x)
        outputs = Dense(output_shape, activation=None)(x)
        return Model(inputs, outputs, name=name)
    
if __name__ == "__main__":
    # constants and hyperparameters
    b_thick = 1
    include_t0 = True
    include_tT = True
    num_latentdim = 3
    num_units = 64
    num_layers = 3
    dropout = 0.0
    l2_reg = 1e-5
    lr = 1e-3
    patch_dim = [5, 5, 5]  # (x, y, t)
    num_patches = 100
    epochs = 5
    
    # Load variables from pickle file
    with open(r'c:\Users\darsh\Documents\fyp\myfyp\advection_diffusion\time_in_encoders_only\numerical_data.pkl', 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    Y = data['Y']
    U = data['U']
    T = data['T']

    solver = sinn(X, Y, U, T)
    solver.standardise_u()
    solver.split_interior_boundary(b_thick, include_t0, include_tT)
    solver.build_models(num_latentdim, num_units, num_layers, dropout, l2_reg, lr)
    solver.train(epochs, patch_dim, num_patches)
    
    test = 0
