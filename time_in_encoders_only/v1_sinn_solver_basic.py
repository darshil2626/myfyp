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
        self.U_original = self.U.copy()  # Store original before any standardisation
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
        
        # ---- Mask cloud (Option A: patch vector) ----
        self.mask_radius = 1   # r=1 => 3x3 window. Try 1 then 2 (5x5)
        self.mask_pad_value = 0.0  # value used for out-of-bounds padding
        self.encoder_input_dim = None


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
        # ---- Mask-cloud encoder input dimension ----
        k = 2 * self.mask_radius + 1
        self.encoder_input_dim = 3 + 2 * (k * k)
        self.num_latentdim = int(num_latentdim)

        self.interior_encoder = self._build_coder(
            num_units, num_layers, input_shape=self.encoder_input_dim, output_shape=self.num_latentdim,
            name="interior_encoder", dropout=dropout, l2_reg=l2_reg
        )
        self.boundary_encoder = self._build_coder(
            num_units, num_layers, input_shape=self.encoder_input_dim, output_shape=self.num_latentdim,
            name="boundary_encoder", dropout=dropout, l2_reg=l2_reg
        )
        self.decoder = self._build_coder(
            num_units, num_layers, input_shape=self.num_latentdim, output_shape=1,
            name="decoder", dropout=dropout, l2_reg=l2_reg
        )

        # ---- SPD latent operator via Cholesky: A = L L^T ----
        self._chol_eps = tf.constant(1e-6, dtype=tf.float32)  # numerical safety

        # Unconstrained parameter (we'll take its lower triangle as L_raw)
        # Start near identity by setting B ~ I.
        init_B = np.eye(self.num_latentdim, dtype=np.float32)
        self.a_chol_raw = tf.Variable(init_B, trainable=True, dtype=tf.float32, name="a_chol_raw")

        self.trainable_vars = (
            self.interior_encoder.trainable_variables
            + self.boundary_encoder.trainable_variables
            + self.decoder.trainable_variables
            + [self.a_chol_raw]
        )

        print(f"Trainable variables: {len(self.trainable_vars)}")
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    # -----------------------------
    # Patch sampling (indices only)
    # -----------------------------
    # def create_patch_centres_and_indices_and_values_and_boundary_splits(self, patch_dim, num_patches):
    #     """
    #     Create random spatio-temporal patches and store indices needed for training.

    #     - Stores indices only (no values).
    #     - Patch boundary is the outer shell of the cuboid (including t-faces).
    #     - Patch interior are points strictly inside that shell.
    #     """
    #     px, py, pt = int(patch_dim[0]), int(patch_dim[1]), int(patch_dim[2])

    #     # Offsets that keep the patch in-bounds.
    #     x_left, x_right = px // 2, (px - 1) // 2
    #     y_left, y_right = py // 2, (py - 1) // 2
    #     t_left, t_right = pt // 2, (pt - 1) // 2

    #     x_min, x_max = x_left, self.nx - x_right - 1
    #     y_min, y_max = y_left, self.ny - y_right - 1
    #     t_min, t_max = t_left, self.nt - t_right - 1

    #     if x_min > x_max or y_min > y_max or t_min > t_max:
    #         raise ValueError("Patch dimensions are too large for the grid.")

    #     # Centres
    #     self.patch_center_idx = np.zeros((num_patches, 3), dtype=np.int32)

    #     # Variable-length arrays per patch (indices only)
    #     self.patch_interior_idx = np.empty(num_patches, dtype=object)
    #     self.patch_boundary_idx = np.empty(num_patches, dtype=object)
    #     self.patch_boundary_global_boundary_idx = np.empty(num_patches, dtype=object)
    #     self.patch_boundary_global_interior_idx = np.empty(num_patches, dtype=object)

    #     # Local boundary mask for a (pt, py, px) cuboid
    #     patch_boundary_mask_local = np.zeros((pt, py, px), dtype=bool)
    #     patch_boundary_mask_local[0, :, :] = True
    #     patch_boundary_mask_local[-1, :, :] = True
    #     patch_boundary_mask_local[:, 0, :] = True
    #     patch_boundary_mask_local[:, -1, :] = True
    #     patch_boundary_mask_local[:, :, 0] = True
    #     patch_boundary_mask_local[:, :, -1] = True
    #     patch_interior_mask_local = ~patch_boundary_mask_local

    #     # Local offsets relative to centre
    #     t_offsets = np.arange(-t_left, t_right + 1, dtype=np.int32)
    #     y_offsets = np.arange(-y_left, y_right + 1, dtype=np.int32)
    #     x_offsets = np.arange(-x_left, x_right + 1, dtype=np.int32)
    #     TT, YY, XX = np.meshgrid(t_offsets, y_offsets, x_offsets, indexing="ij")

    #     local_offsets_flat = np.stack([TT.ravel(), YY.ravel(), XX.ravel()], axis=1)  # (pt*py*px, 3)
    #     boundary_mask_flat = patch_boundary_mask_local.ravel()
    #     interior_mask_flat = patch_interior_mask_local.ravel()

    #     rng = np.random.default_rng()

    #     for k in range(num_patches):
    #         ct = rng.integers(t_min, t_max + 1)
    #         cy = rng.integers(y_min, y_max + 1)
    #         cx = rng.integers(x_min, x_max + 1)
    #         self.patch_center_idx[k] = (ct, cy, cx)

    #         # Absolute indices for all points in the cuboid
    #         abs_t = ct + local_offsets_flat[:, 0]
    #         abs_y = cy + local_offsets_flat[:, 1]
    #         abs_x = cx + local_offsets_flat[:, 2]
    #         idx_all = np.stack([abs_t, abs_y, abs_x], axis=1).astype(np.int32)

    #         patch_boundary_idx = idx_all[boundary_mask_flat]
    #         patch_interior_idx = idx_all[interior_mask_flat]

    #         self.patch_boundary_idx[k] = patch_boundary_idx
    #         self.patch_interior_idx[k] = patch_interior_idx

    #         # Split patch boundary by *global* boundary membership
    #         if getattr(self, "mask_boundary", None) is not None:
    #             is_global_boundary = self.mask_boundary[
    #                 patch_boundary_idx[:, 0], patch_boundary_idx[:, 1], patch_boundary_idx[:, 2]
    #             ]
    #         else:
    #             is_global_boundary = np.zeros((patch_boundary_idx.shape[0],), dtype=bool)

    #         self.patch_boundary_global_boundary_idx[k] = patch_boundary_idx[is_global_boundary]
    #         self.patch_boundary_global_interior_idx[k] = patch_boundary_idx[~is_global_boundary]
    
    def create_patch_centres_and_indices_and_values_and_boundary_splits(self, patch_dim, num_patches, boundary_fraction=0.5):
        """
        Create spatio-temporal patches with bias toward global boundaries.
        
        Args:
            patch_dim: [px, py, pt] - patch dimensions
            num_patches: Total number of patches to create
            boundary_fraction: Fraction of patches guaranteed to touch global boundary (default: 0.6)
        
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

        # Split patches into boundary-touching and interior
        num_boundary_patches = int(num_patches * boundary_fraction)
        num_interior_patches = num_patches - num_boundary_patches

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

        offsets = np.stack([TT.ravel(), YY.ravel(), XX.ravel()], axis=1)
        boundary_offsets = offsets[patch_boundary_mask_local.ravel()]
        interior_offsets = offsets[patch_interior_mask_local.ravel()]

        # Sample boundary-touching patches
        for p in range(num_boundary_patches):
            # Randomly choose which face(s) to touch
            face = np.random.choice(['t0', 'tT', 'x_left', 'x_right', 'y_bottom', 'y_top'])
            
            if face == 't0':
                # Touch t=0 face
                t_c = t_left
                y_c = np.random.randint(y_min, y_max + 1)
                x_c = np.random.randint(x_min, x_max + 1)
            elif face == 'tT':
                # Touch t=T face
                t_c = self.nt - t_right - 1
                y_c = np.random.randint(y_min, y_max + 1)
                x_c = np.random.randint(x_min, x_max + 1)
            elif face == 'x_left':
                # Touch left spatial boundary
                x_c = x_left
                y_c = np.random.randint(y_min, y_max + 1)
                t_c = np.random.randint(t_min, t_max + 1)
            elif face == 'x_right':
                # Touch right spatial boundary
                x_c = self.nx - x_right - 1
                y_c = np.random.randint(y_min, y_max + 1)
                t_c = np.random.randint(t_min, t_max + 1)
            elif face == 'y_bottom':
                # Touch bottom spatial boundary
                y_c = y_left
                x_c = np.random.randint(x_min, x_max + 1)
                t_c = np.random.randint(t_min, t_max + 1)
            else:  # y_top
                # Touch top spatial boundary
                y_c = self.ny - y_right - 1
                x_c = np.random.randint(x_min, x_max + 1)
                t_c = np.random.randint(t_min, t_max + 1)

            self.patch_center_idx[p, :] = [t_c, y_c, x_c]

            # Global indices for patch boundary and interior
            patch_bnd_global = np.array([t_c, y_c, x_c], dtype=np.int32) + boundary_offsets
            patch_int_global = np.array([t_c, y_c, x_c], dtype=np.int32) + interior_offsets

            # Split patch boundary into global boundary and global interior
            is_global_boundary = self.mask_boundary[
                patch_bnd_global[:, 0], patch_bnd_global[:, 1], patch_bnd_global[:, 2]
            ]

            patch_bnd_on_glob_bnd = patch_bnd_global[is_global_boundary]
            patch_bnd_on_glob_int = patch_bnd_global[~is_global_boundary]

            self.patch_interior_idx[p] = patch_int_global
            self.patch_boundary_idx[p] = patch_bnd_global
            self.patch_boundary_global_boundary_idx[p] = patch_bnd_on_glob_bnd
            self.patch_boundary_global_interior_idx[p] = patch_bnd_on_glob_int

        # Sample remaining interior patches randomly
        for p in range(num_boundary_patches, num_patches):
            t_c = np.random.randint(t_min, t_max + 1)
            y_c = np.random.randint(y_min, y_max + 1)
            x_c = np.random.randint(x_min, x_max + 1)

            self.patch_center_idx[p, :] = [t_c, y_c, x_c]

            # Global indices for patch boundary and interior
            patch_bnd_global = np.array([t_c, y_c, x_c], dtype=np.int32) + boundary_offsets
            patch_int_global = np.array([t_c, y_c, x_c], dtype=np.int32) + interior_offsets

            # Split patch boundary into global boundary and global interior
            is_global_boundary = self.mask_boundary[
                patch_bnd_global[:, 0], patch_bnd_global[:, 1], patch_bnd_global[:, 2]
            ]

            patch_bnd_on_glob_bnd = patch_bnd_global[is_global_boundary]
            patch_bnd_on_glob_int = patch_bnd_global[~is_global_boundary]

            self.patch_interior_idx[p] = patch_int_global
            self.patch_boundary_idx[p] = patch_bnd_global
            self.patch_boundary_global_boundary_idx[p] = patch_bnd_on_glob_bnd
            self.patch_boundary_global_interior_idx[p] = patch_bnd_on_glob_int

        # Count how many patches actually touch the global boundary
        num_touching = sum(1 for p in range(num_patches) 
                        if len(self.patch_boundary_global_boundary_idx[p]) > 0)

        print(f"  - {num_touching} total patches touch global boundary ({100*num_touching/num_patches:.1f}%)")

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
    
    def _stack_mask_patch_features_from_idx(self, idx_tyx, radius=None):
        """
        Option A mask-cloud encoder features.

        For each (t,y,x), build:
          [t_norm, y_norm, x_norm,  vec(U_window), vec(mask_window)]

        - U_window is (2r+1)x(2r+1) around (y,x) at fixed t, pulled from self.U (standardised).
        - mask_window is 1 where in-bounds, 0 where padded.

        Returns float32 array of shape (N, 3 + 2*k*k), k=(2r+1).
        """
        idx_tyx = np.asarray(idx_tyx, dtype=np.int32)
        if idx_tyx.size == 0:
            r = self.mask_radius if radius is None else int(radius)
            k = 2 * r + 1
            d = 3 + 2 * (k * k)
            return np.zeros((0, d), dtype=np.float32)


        r = self.mask_radius if radius is None else int(radius)
        k = 2 * r + 1
        kk = k * k

        t = idx_tyx[:, 0]
        y = idx_tyx[:, 1]
        x = idx_tyx[:, 2]
        N = idx_tyx.shape[0]

        # Normalised coords
        t_norm = t.astype(np.float32) / self.nt_norm
        y_norm = y.astype(np.float32) / self.ny_norm
        x_norm = x.astype(np.float32) / self.nx_norm

        # Allocate
        u_win = np.full((N, kk), self.mask_pad_value, dtype=np.float32)
        m_win = np.zeros((N, kk), dtype=np.float32)

        # Fill windows (python loops are fine for now; you can vectorise later)
        for i in range(N):
            ti = int(t[i])
            yi = int(y[i])
            xi = int(x[i])

            ptr = 0
            for dy in range(-r, r + 1):
                yy = yi + dy
                for dx in range(-r, r + 1):
                    xx = xi + dx
                    if (0 <= yy < self.ny) and (0 <= xx < self.nx):
                        u_win[i, ptr] = float(self.U[ti, yy, xx])  # self.U is standardised already
                        m_win[i, ptr] = 1.0
                    # else: keep pad value and mask=0
                    ptr += 1

        feats = np.concatenate(
            [
                np.stack([t_norm, y_norm, x_norm], axis=1).astype(np.float32),
                u_win,
                m_win,
            ],
            axis=1,
        ).astype(np.float32)

        return feats


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
        feats_bnd = self._stack_mask_patch_features_from_idx(patch_boundary_global_boundary_idx_tyx)
        feats_int = self._stack_mask_patch_features_from_idx(patch_boundary_global_interior_idx_tyx)

        if feats_bnd.shape[0] > 0:
            lat_bnd = self.boundary_encoder(tf.convert_to_tensor(feats_bnd, dtype=tf.float32), training=training)
        else:
            lat_bnd = tf.zeros((0, self.num_latentdim), dtype=tf.float32)

        if feats_int.shape[0] > 0:
            lat_int = self.interior_encoder(tf.convert_to_tensor(feats_int, dtype=tf.float32), training=training)
        else:
            lat_int = tf.zeros((0, self.num_latentdim), dtype=tf.float32)


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
    
    def get_latent_operator_matrix(self) -> tf.Tensor:
        """
        Return SPD latent operator A = L L^T via Cholesky parameterisation.

        L is lower triangular, with positive diagonal enforced by softplus.
        """
        B = self.a_chol_raw

        # Lower triangular part
        L = tf.linalg.band_part(B, -1, 0)

        # Positive diagonal
        diag = tf.linalg.diag_part(L)
        diag_pos = tf.nn.softplus(diag) + self._chol_eps
        L = tf.linalg.set_diag(L, diag_pos)

        A = tf.matmul(L, L, transpose_b=True)
        return A


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
        alpha_recon: float = 1.0,
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
            print("Warning: Patch center is not in strict interior. No interior supervision for this patch.")
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

        # SPD A via Cholesky (no penalty needed)
        latent_operator_matrix = self.get_latent_operator_matrix()
        spd_loss = tf.constant(0.0, tf.float32)


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

            # t_norm = interior_t.astype(np.float32) / self.nt_norm
            # y_norm = interior_y.astype(np.float32) / self.ny_norm
            # x_norm = interior_x.astype(np.float32) / self.nx_norm

            idx_all = np.stack([interior_t, interior_y, interior_x], axis=1).astype(np.int32)
            interior_features_all = self._stack_mask_patch_features_from_idx(idx_all)
            interior_features_all_tf = tf.constant(interior_features_all, dtype=tf.float32)

            latent_true_all_interior = self.interior_encoder(interior_features_all_tf, training=True)

            latent_consistency_loss_accum += tf.reduce_sum(tf.square(latent_solution_vector_tf - latent_true_all_interior))

            u_pred_all_interior = self.decoder(latent_solution_vector_tf, training=True)
            # interior_t, interior_y, interior_x are 1D arrays of indices
            u_true_all_interior = self.U[interior_t, interior_y, interior_x].astype(np.float32).reshape(-1, 1)
            u_true_all_interior_tf = tf.constant(u_true_all_interior, dtype=tf.float32)

            reconstruction_loss_accum += tf.reduce_sum(tf.square(u_pred_all_interior - u_true_all_interior_tf))


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
                        alpha_recon=1.0,
                    )

                grads = tape.gradient(total_loss, self.trainable_vars)
                pairs = [(g, v) for (g, v) in zip(grads, self.trainable_vars) if g is not None]
                if not pairs:
                    continue
                grads, vars_ = zip(*pairs)
                grads, _ = tf.clip_by_global_norm(grads, clip_norm)
                self.optimizer.apply_gradients(zip(grads, vars_))


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
        Boundary-only reconstruction (correct evaluation):

        - Uses ONLY observed boundary values from U_original at time t.
        - Boundary encoder receives mask-cloud features where any unobserved
        window entries are zeroed and masked out.
        - No interior data leakage through feature windows.
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import spsolve

        t = int(t_index)
        print(f"\n[Boundary-only] Reconstructing field at timestep t={t}...")

        b_thick = int(getattr(self, "b_thick", 1))

        # -----------------------------
        # 0) Define observed set OBS
        # -----------------------------
        # Here: rectangular boundary of thickness b_thick is "observed"
        obs_mask = np.zeros((self.ny, self.nx), dtype=bool)
        obs_mask[:b_thick, :] = True
        obs_mask[-b_thick:, :] = True
        obs_mask[:, :b_thick] = True
        obs_mask[:, -b_thick:] = True

        # Define interior as NOT observed (for evaluation)
        spatial_interior_mask = ~obs_mask
        spatial_boundary_mask = obs_mask

        interior_indices = np.argwhere(spatial_interior_mask)  # (N_int, 2)
        boundary_indices = np.argwhere(spatial_boundary_mask)  # (N_bnd, 2)

        num_interior = interior_indices.shape[0]
        num_boundary = boundary_indices.shape[0]

        print(f"  Interior points: {num_interior}")
        print(f"  Observed boundary points: {num_boundary}")

        if num_interior == 0:
            raise ValueError(
                f"No interior points! Check b_thick={b_thick} vs grid ({self.ny},{self.nx})."
            )

        # ---------------------------------------------------------------------
        # Helper: build mask-cloud features but ONLY using observed u-values
        # ---------------------------------------------------------------------
        # This matches your training feature layout:
        # [t_norm, y_norm, x_norm, u_win(flat), mask_win(flat)]
        # but with u_win entries only filled where obs_mask is True.
        def stack_mask_patch_features_observed_only(idx_tyx: np.ndarray) -> np.ndarray:
            idx_tyx = np.asarray(idx_tyx, dtype=np.int32)
            N = idx_tyx.shape[0]

            r = int(getattr(self, "mask_radius", 1))
            win = 2 * r + 1
            win_sz = win * win

            feats = np.zeros((N, 3 + 2 * win_sz), dtype=np.float32)

            # coords
            t_norm = idx_tyx[:, 0].astype(np.float32) / max(self.nt - 1, 1)
            y_norm = idx_tyx[:, 1].astype(np.float32) / max(self.ny - 1, 1)
            x_norm = idx_tyx[:, 2].astype(np.float32) / max(self.nx - 1, 1)
            feats[:, 0] = t_norm
            feats[:, 1] = y_norm
            feats[:, 2] = x_norm

            # window entries
            for i in range(N):
                ti, yi, xi = idx_tyx[i]
                ptr = 0
                for dy in range(-r, r + 1):
                    yy = yi + dy
                    for dx in range(-r, r + 1):
                        xx = xi + dx

                        in_bounds = (0 <= yy < self.ny) and (0 <= xx < self.nx)
                        if in_bounds and obs_mask[yy, xx]:
                            # observed ⇒ use STANDARDISED value from self.U (already standardised)
                            # IMPORTANT: self.U is standardised version of U_original.
                            feats[i, 3 + ptr] = float(self.U[ti, yy, xx])
                            feats[i, 3 + win_sz + ptr] = 1.0
                        else:
                            # unobserved/out-of-bounds ⇒ masked out
                            feats[i, 3 + ptr] = 0.0
                            feats[i, 3 + win_sz + ptr] = 0.0
                        ptr += 1

            return feats

        # -----------------------------
        # 1) Encode boundary conditions
        # -----------------------------
        print("  Step 1: Encoding observed boundary conditions...")

        boundary_y = boundary_indices[:, 0].astype(np.int32)
        boundary_x = boundary_indices[:, 1].astype(np.int32)
        boundary_t = np.full(num_boundary, t, dtype=np.int32)

        # raw boundary values for later assembly
        u_boundary_raw = self.U_original[boundary_t, boundary_y, boundary_x].astype(np.float32)

        idx_bnd = np.stack([boundary_t, boundary_y, boundary_x], axis=1).astype(np.int32)

        # boundary features with *no interior leakage*
        boundary_features = stack_mask_patch_features_observed_only(idx_bnd)
        boundary_latents = self.boundary_encoder(
            tf.constant(boundary_features, dtype=tf.float32),
            training=False
        ).numpy()

        latent_dim = boundary_latents.shape[1]
        print(f"  Latent dimension: {latent_dim}")

        # --------------------------------
        # 2) Build spatial Laplacian
        # --------------------------------
        print("  Step 2: Building Laplacian operator...")

        interior_row_map = -np.ones((self.ny, self.nx), dtype=np.int32)
        for row_id, (y, x) in enumerate(interior_indices):
            interior_row_map[y, x] = row_id

        # speed: map boundary (y,x) -> boundary row index
        bnd_map = { (int(y), int(x)): i for i, (y, x) in enumerate(boundary_indices) }

        row_indices, col_indices, values = [], [], []
        neighbour_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        boundary_contributions = [[] for _ in range(num_interior)]

        for row_id, (y, x) in enumerate(interior_indices):
            row_indices.append(row_id); col_indices.append(row_id); values.append(4.0)
            for dy, dx in neighbour_steps:
                nyy, nxx = int(y + dy), int(x + dx)
                nbr_row = interior_row_map[nyy, nxx]
                if nbr_row >= 0:
                    row_indices.append(row_id); col_indices.append(int(nbr_row)); values.append(-1.0)
                else:
                    # neighbour is boundary (should be in bnd_map)
                    bidx = bnd_map.get((nyy, nxx), None)
                    if bidx is not None:
                        boundary_contributions[row_id].append(bidx)

        laplacian_sparse = sp.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(num_interior, num_interior),
            dtype=np.float32
        )

        # --------------------------------
        # 3) Get A and prepare eigenbasis
        # --------------------------------
        print("  Step 3: Getting PDE operator (A matrix)...")
        A_matrix_np = self.get_latent_operator_matrix().numpy().astype(np.float64)

        # Defensive symmetrisation (helps eigh + stability)
        A_matrix_np = 0.5 * (A_matrix_np + A_matrix_np.T)

        # Eigen-decomposition: A = Q diag(lam) Q^T
        eigvals, Q = np.linalg.eigh(A_matrix_np)  # eigvals (d,), Q (d,d)

        # Sanity check: elliptic solve needs SPD A (positive eigenvalues)
        min_eig = float(np.min(eigvals))
        if min_eig <= 1e-10:
            raise ValueError(f"A is not SPD enough for elliptic solve. min eigenvalue = {min_eig:.3e}")

        # --------------------------------
        # 4) RHS from boundary latents (same as you have, but float64)
        # --------------------------------
        print("  Step 4: Building right-hand side...")
        rhs = np.zeros((num_interior, latent_dim), dtype=np.float64)
        for row_id in range(num_interior):
            for bidx in boundary_contributions[row_id]:
                rhs[row_id, :] += A_matrix_np @ boundary_latents[bidx, :].astype(np.float64)

        # Rotate RHS into eigenbasis: rhs_tilde = rhs * Q
        rhs_tilde = rhs @ Q  # (Nint, d)

        # --------------------------------
        # 5) Solve d Laplacian systems instead of 1 big kron system
        # --------------------------------
        print("  Step 5: Solving linear systems in eigenbasis (CG)...")
        from scipy.sparse.linalg import cg

        K = laplacian_sparse.tocsr()

        latent_tilde = np.zeros_like(rhs_tilde, dtype=np.float64)

        tol = 1e-6
        maxiter = 2000

        for k in range(latent_dim):
            lam = float(eigvals[k])
            b = rhs_tilde[:, k] / lam

            xk, info = cg(K, b, tol=tol, maxiter=maxiter)
            if info != 0:
                print(f"    [cg] dim {k}: did not fully converge (info={info}). Consider preconditioning.")
            latent_tilde[:, k] = xk

        # Rotate back: latent = latent_tilde * Q^T
        latent_interior = (latent_tilde @ Q.T).astype(np.float32)

        # --------------------------------
        # 6) Decode
        # --------------------------------
        print("  Step 6: Decoding latents to physical field...")
        u_interior_pred_standardised = self.decoder(
            tf.constant(latent_interior, dtype=tf.float32),
            training=False
        ).numpy().reshape(-1)

        # --------------------------------
        # 7) Unstandardise
        # --------------------------------
        print("  Step 7: Unstandardising predictions...")
        interior_y = interior_indices[:, 0]
        interior_x = interior_indices[:, 1]
        u_interior_pred = (
            u_interior_pred_standardised * self.U_std[interior_y, interior_x]
            + self.U_mean[interior_y, interior_x]
        )

        # --------------------------------
        # 8) Assemble + stats
        # --------------------------------
        print("  Step 8: Assembling full field...")
        u_pred_full = np.zeros((self.ny, self.nx), dtype=np.float32)
        u_true_full = self.U_original[t, :, :].astype(np.float32)

        # boundary = observed true values
        u_pred_full[boundary_y, boundary_x] = u_boundary_raw
        # interior = predicted
        u_pred_full[interior_y, interior_x] = u_interior_pred

        u_error = np.abs(u_pred_full - u_true_full)

        interior_error = u_error[interior_y, interior_x]
        mse = float(np.mean(interior_error ** 2))
        mae = float(np.mean(interior_error))
        max_error = float(np.max(interior_error))

        print("\n  [Boundary-only] Reconstruction Statistics (interior only):")
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
        u_error = np.abs(results["u_error"]) * 100.0 / (np.abs(u_true) + 1e-8)  # Avoid division by zero
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
        ax.set_title("Percentage Error", fontsize=13, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.scatter(boundary_coords[:, 1], boundary_coords[:, 0], c="cyan", s=1, alpha=0.5)
        plt.colorbar(im3, ax=ax).set_label("% error")

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
    patch_dim = [10, 10, 10]  # [px, py, pt] - spatial_x, spatial_y, time dimensions
    num_patches = 100
    epochs = 5
    

    with open(r"c:\Users\darsh\Documents\fyp\myfyp\advection_diffusion\time_in_encoders_only\numerical_data_sst.pkl", "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    Y = data["Y"]
    U = data["U"]
    T = data["T"]

    solver = sinn(X, Y, U, T, debug=False)
    solver.standardise_u()  # Now uncommented - CRITICAL for proper training!
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

    results = solver.reconstruct_field_at_timestep(t_index=177)
    solver.plot_field_reconstruction(results, save_path="field_t2.25.png", show=True)
    print(f"Reconstruction MAE (with PDE): {results['mae']:.6e}")
    
    A_final = solver.get_latent_operator_matrix().numpy()
    print("A matrix (first 3x3):")
    print(A_final[:3, :3])
    print(f"\nFrobenius distance from identity: {np.linalg.norm(A_final - np.eye(solver.num_latentdim)):.4f}")
    print(f"Eigenvalues: {np.linalg.eigvalsh(A_final)}")