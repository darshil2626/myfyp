import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Dense, Input
from keras.models import Model
import pickle
import matplotlib
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.sparse import diags

matplotlib.use("Agg")

from config import (
    EPS_STD, EPS_CHOL, EPS_EIG, NAN_CLAMP, GRAD_CLIP_NORM,
    CG_TOL_FAST, CG_MAXITER_FAST, CG_TOL_ACCURATE, CG_MAXITER_ACCURATE,
    MASK_RADIUS,
)


class sinn:
    """
    Time-conditioned SINN with decoder bypass (skip connection).

    Homogeneous elliptic PDE: div(A grad ell) = 0.
    The decoder receives both the latent field AND a harmonic interpolation
    of boundary physical values, so it learns to predict the residual
    between what boundary interpolation gives and the true field.
    This bypasses the elliptic solve's smoothing for fine-scale structure.
    """

    def __init__(self, X, Y, U, T, debug: bool = False):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.U = np.asarray(U)
        self.U_original = self.U.copy()
        self.T = np.asarray(T)
        self.nt, self.ny, self.nx = self.U.shape
        self.debug = bool(debug)

        self.nt_norm = float(self.nt - 1) if self.nt > 1 else 1.0
        self.ny_norm = float(self.ny - 1) if self.ny > 1 else 1.0
        self.nx_norm = float(self.nx - 1) if self.nx > 1 else 1.0

        self.b_thick = 1
        self.num_latentdim = None
        self.train_time_indices = np.arange(self.nt, dtype=np.int32)
        self.test_time_indices = np.array([], dtype=np.int32)

        self.mask_radius = MASK_RADIUS
        self.mask_pad_value = 0.0
        self.encoder_input_dim = None
        self.n_past_steps = 0

        # Raw coordinates (no Fourier embedding)
        self.coord_embed_dim = 3

    # =========================================================================
    # Coordinate embedding (raw — no Fourier features)
    # =========================================================================
    def _fourier_embed_coords(self, t_norm, y_norm, x_norm):
        return np.stack([t_norm, y_norm, x_norm], axis=1).astype(np.float32)

    # =========================================================================
    # Normalisation
    # =========================================================================
    def standardise_u(self, eps: float = EPS_STD, time_indices=None):
        if getattr(self, "_is_standardised", False):
            raise RuntimeError(
                "standardise_u() called twice. self.U is already standardised. "
                "Create a fresh sinn instance or reload the original data."
            )
        if time_indices is None:
            U_ref = self.U
        else:
            time_indices = np.asarray(time_indices, dtype=np.int32)
            U_ref = self.U[time_indices, :, :]
        n_nan = np.isnan(U_ref).sum()
        n_inf = np.isinf(U_ref).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"[standardise_u] Warning: {n_nan} NaN, {n_inf} Inf")
            U_ref = np.nan_to_num(U_ref, nan=0.0, posinf=1e6, neginf=-1e6)
            self.U = np.nan_to_num(self.U, nan=0.0, posinf=1e6, neginf=-1e6)
        self.U_mean = np.mean(U_ref, axis=0)
        self.U_std = np.std(U_ref, axis=0)
        self.U_std = np.maximum(self.U_std, eps)
        self.U = (self.U - self.U_mean) / self.U_std
        if np.isnan(self.U).sum() > 0:
            self.U = np.nan_to_num(self.U, nan=0.0)
        self._is_standardised = True

    def unstandardise_u(self, U_pred):
        return U_pred * self.U_std + self.U_mean

    # =========================================================================
    # Masks
    # =========================================================================
    def split_interior_boundary(self, b_thick: int, include_t0=True, include_tT=True):
        self.b_thick = int(b_thick)
        sp_int = np.zeros((self.ny, self.nx), dtype=bool)
        sp_int[self.b_thick:-self.b_thick, self.b_thick:-self.b_thick] = True
        sp_bnd = ~sp_int
        # Spatial-only obs mask — what the boundary encoder ACTUALLY sees at
        # inference. Stored so training can apply the same masking and avoid
        # the train/test feature distribution shift.
        self.obs_mask_spatial = sp_bnd.copy()
        mask_bnd = np.zeros((self.nt, self.ny, self.nx), dtype=bool)
        mask_bnd[:, sp_bnd] = True
        if include_t0:
            mask_bnd[0, :, :] = True
        if include_tT:
            mask_bnd[-1, :, :] = True
        self.mask_boundary = mask_bnd
        safe = np.zeros((self.nt, self.ny, self.nx), dtype=bool)
        safe[1:-1, 1:-1, 1:-1] = True
        safe[:, :self.b_thick, :] = False
        safe[:, -self.b_thick:, :] = False
        safe[:, :, :self.b_thick] = False
        safe[:, :, -self.b_thick:] = False
        self.mask_fd_safe_interior = safe & ~mask_bnd

        # ---- Cache spatial Laplacian (built once, reused every timestep) ----
        interior_indices = np.argwhere(~sp_bnd).astype(np.int32)   # (n_int, 2)
        boundary_indices = np.argwhere(sp_bnd).astype(np.int32)    # (n_bnd, 2)
        n_int = interior_indices.shape[0]

        interior_row_map = np.full((self.ny, self.nx), -1, dtype=np.int32)
        interior_row_map[interior_indices[:, 0], interior_indices[:, 1]] = np.arange(n_int, dtype=np.int32)

        bnd_map = {(int(y), int(x)): i for i, (y, x) in enumerate(boundary_indices)}

        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        rows_l, cols_l, vals_l = [], [], []
        bnd_contribs = [[] for _ in range(n_int)]
        for rid in range(n_int):
            y, x = int(interior_indices[rid, 0]), int(interior_indices[rid, 1])
            rows_l.append(rid); cols_l.append(rid); vals_l.append(4.0)
            for dy, dx in steps:
                ny_, nx_ = y + dy, x + dx
                if ny_ < 0 or ny_ >= self.ny or nx_ < 0 or nx_ >= self.nx:
                    continue
                nr = int(interior_row_map[ny_, nx_])
                if nr >= 0:
                    rows_l.append(rid); cols_l.append(nr); vals_l.append(-1.0)
                else:
                    bi = bnd_map.get((ny_, nx_))
                    if bi is not None:
                        bnd_contribs[rid].append(bi)

        self._interior_indices = interior_indices
        self._boundary_indices = boundary_indices
        self._interior_row_map = interior_row_map
        self._bnd_map = bnd_map
        self._bnd_contribs = bnd_contribs
        self._K_sparse = sp.csr_matrix(
            (vals_l, (rows_l, cols_l)), shape=(n_int, n_int), dtype=np.float64
        )
        diag_K = np.array(self._K_sparse.sum(axis=1)).ravel()
        diag_K[diag_K == 0] = 1.0
        self._M_pre = diags(1.0 / diag_K, dtype=np.float64, format="csr")
        print(f"[laplacian] Cached K ({n_int}×{n_int}), {len(boundary_indices)} boundary pts")

    # =========================================================================
    # Models
    # =========================================================================
    @staticmethod
    def _build_coder(num_units, num_layers, input_shape, output_shape, name, dropout, l2_reg):
        inputs = Input(shape=(input_shape,))
        x = Dense(num_units, activation="tanh", kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
        for _ in range(num_layers - 1):
            x = Dense(num_units, activation="tanh", kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        if dropout and dropout > 0.0:
            x = layers.Dropout(dropout)(x)
        outputs = Dense(output_shape, activation=None)(x)
        return Model(inputs, outputs, name=name)

    def build_models(self, num_latentdim, num_units, num_layers, dropout, l2_reg, lr, n_past_steps=0):
        self.n_past_steps = int(n_past_steps)
        k = 2 * self.mask_radius + 1
        n_time_slices = 1 + self.n_past_steps
        self.encoder_input_dim = self.coord_embed_dim + 2 * (k * k) * n_time_slices
        self.num_latentdim = int(num_latentdim)

        self.interior_encoder = self._build_coder(
            num_units, num_layers, input_shape=self.encoder_input_dim,
            output_shape=self.num_latentdim,
            name="interior_encoder", dropout=dropout, l2_reg=l2_reg,
        )
        self.boundary_encoder = self._build_coder(
            num_units, num_layers, input_shape=self.encoder_input_dim,
            output_shape=self.num_latentdim,
            name="boundary_encoder", dropout=dropout, l2_reg=l2_reg,
        )
        # Decoder receives latent field + harmonic interpolation of boundary values
        # The +1 is the skip connection: boundary-interpolated physical value at each point
        self.decoder = self._build_coder(
            num_units, num_layers, input_shape=self.num_latentdim + 1,
            output_shape=1, name="decoder", dropout=dropout, l2_reg=l2_reg,
        )

        self._chol_eps = tf.constant(EPS_CHOL, dtype=tf.float32)
        init_B = np.eye(self.num_latentdim, dtype=np.float32)
        self.a_chol_raw = tf.Variable(init_B, trainable=True, dtype=tf.float32, name="a_chol_raw")

        self.trainable_vars = (
            self.interior_encoder.trainable_variables
            + self.boundary_encoder.trainable_variables
            + self.decoder.trainable_variables
            + [self.a_chol_raw]
        )
        print(f"Trainable variables: {len(self.trainable_vars)}")
        print(f"  Encoder input dim: {self.encoder_input_dim}")
        print(f"  Decoder input dim: {self.num_latentdim + 1} (latent + bypass)")
        print(f"  Coordinates: raw (t, y, x) — no Fourier embedding")
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    # =========================================================================
    # Train / test split
    # =========================================================================
    def split_train_test_timesteps(self, mode: str = "sequential", train_frac: float = 0.8):
        all_t = np.arange(self.nt, dtype=np.int32)
        if mode == "sequential":
            n_train = max(1, min(int(self.nt * train_frac), self.nt - 1))
            self.train_time_indices = all_t[:n_train]
            self.test_time_indices = all_t[n_train:]
        elif mode == "alternate":
            self.train_time_indices = all_t[0::2]
            self.test_time_indices = all_t[1::2]
        elif mode == "all":
            self.train_time_indices = all_t
            self.test_time_indices = np.array([], dtype=np.int32)
        else:
            raise ValueError(f"Unknown mode '{mode}'.")
        print(f"[split] mode='{mode}'")
        print(f"  Train: {len(self.train_time_indices)} steps "
              f"(t={self.train_time_indices[0]}..{self.train_time_indices[-1]})")
        if len(self.test_time_indices):
            print(f"  Test:  {len(self.test_time_indices)} steps "
                  f"(t={self.test_time_indices[0]}..{self.test_time_indices[-1]})")

    # =========================================================================
    # Patch sampling
    # =========================================================================
    def create_patches(self, patch_dim, num_patches, boundary_fraction=0.5):
        px, py, pt = int(patch_dim[0]), int(patch_dim[1]), int(patch_dim[2])
        x_left, x_right = px // 2, (px - 1) // 2
        y_left, y_right = py // 2, (py - 1) // 2
        t_left, t_right = pt // 2, (pt - 1) // 2
        x_min, x_max = x_left, self.nx - x_right - 1
        y_min, y_max = y_left, self.ny - y_right - 1
        t_min, t_max = t_left, self.nt - t_right - 1
        if x_min > x_max or y_min > y_max or t_min > t_max:
            raise ValueError("Patch dimensions too large.")
        valid_train_t = self.train_time_indices[
            (self.train_time_indices >= t_min) & (self.train_time_indices <= t_max)]
        if len(valid_train_t) == 0:
            raise ValueError("No valid training time indices for patch half-width.")
        num_bnd = int(num_patches * boundary_fraction)

        self.patch_center_idx = np.zeros((num_patches, 3), dtype=np.int32)
        self.patch_interior_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_global_boundary_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_global_interior_idx = np.empty(num_patches, dtype=object)

        pbm = np.zeros((pt, py, px), dtype=bool)
        pbm[0, :, :] = True; pbm[-1, :, :] = True
        pbm[:, 0, :] = True; pbm[:, -1, :] = True
        pbm[:, :, 0] = True; pbm[:, :, -1] = True

        t_off = np.arange(-t_left, t_right + 1, dtype=np.int32)
        y_off = np.arange(-y_left, y_right + 1, dtype=np.int32)
        x_off = np.arange(-x_left, x_right + 1, dtype=np.int32)
        TT, YY, XX = np.meshgrid(t_off, y_off, x_off, indexing="ij")
        offsets = np.stack([TT.ravel(), YY.ravel(), XX.ravel()], axis=1)
        bnd_offsets = offsets[pbm.ravel()]
        int_offsets = offsets[~pbm.ravel()]

        faces = ['t0', 'tT', 'x_left', 'x_right', 'y_bottom', 'y_top']
        for p in range(num_patches):
            if p < num_bnd:
                face = np.random.choice(faces)
                if face == 't0':
                    t_c, y_c, x_c = int(valid_train_t[0]), np.random.randint(y_min, y_max+1), np.random.randint(x_min, x_max+1)
                elif face == 'tT':
                    t_c, y_c, x_c = int(valid_train_t[-1]), np.random.randint(y_min, y_max+1), np.random.randint(x_min, x_max+1)
                elif face == 'x_left':
                    t_c, y_c, x_c = int(np.random.choice(valid_train_t)), np.random.randint(y_min, y_max+1), x_left
                elif face == 'x_right':
                    t_c, y_c, x_c = int(np.random.choice(valid_train_t)), np.random.randint(y_min, y_max+1), self.nx - x_right - 1
                elif face == 'y_bottom':
                    t_c, y_c, x_c = int(np.random.choice(valid_train_t)), y_left, np.random.randint(x_min, x_max+1)
                else:
                    t_c, y_c, x_c = int(np.random.choice(valid_train_t)), self.ny - y_right - 1, np.random.randint(x_min, x_max+1)
            else:
                t_c = int(np.random.choice(valid_train_t))
                y_c = np.random.randint(y_min, y_max + 1)
                x_c = np.random.randint(x_min, x_max + 1)
            self.patch_center_idx[p, :] = [t_c, y_c, x_c]
            centre = np.array([t_c, y_c, x_c], dtype=np.int32)
            pb = centre + bnd_offsets
            pi = centre + int_offsets
            is_gb = self.mask_boundary[pb[:, 0], pb[:, 1], pb[:, 2]]
            self.patch_interior_idx[p] = pi
            self.patch_boundary_idx[p] = pb
            self.patch_boundary_global_boundary_idx[p] = pb[is_gb]
            self.patch_boundary_global_interior_idx[p] = pb[~is_gb]

    # =========================================================================
    # Feature stacking (vectorised)
    # =========================================================================
    def _stack_mask_patch_features_from_idx(self, idx_tyx, radius=None,
                                              apply_obs_mask=False):
        """Build (N, coord_embed_dim + 2*kk*n_time_slices) features.

        When apply_obs_mask=True, neighborhood positions that are NOT on the
        global spatial-boundary are zeroed out and their validity mask set to
        zero. This mirrors what stack_observed_only does at inference time
        and is required for boundary-encoder training to match the inference
        feature distribution.
        """
        idx_tyx = np.asarray(idx_tyx, dtype=np.int32)
        r = self.mask_radius if radius is None else int(radius)
        k = 2 * r + 1
        kk = k * k
        n_time_slices = 1 + self.n_past_steps

        if idx_tyx.size == 0:
            return np.zeros((0, self.coord_embed_dim + 2 * kk * n_time_slices), dtype=np.float32)

        t = idx_tyx[:, 0]; y = idx_tyx[:, 1]; x = idx_tyx[:, 2]
        N = idx_tyx.shape[0]

        t_norm = t.astype(np.float32) / self.nt_norm
        y_norm = y.astype(np.float32) / self.ny_norm
        x_norm = x.astype(np.float32) / self.nx_norm
        coord_feats = self._fourier_embed_coords(t_norm, y_norm, x_norm)

        dy_off = np.repeat(np.arange(-r, r + 1), k).astype(np.int32)
        dx_off = np.tile(np.arange(-r, r + 1), k).astype(np.int32)
        y_all = y[:, None] + dy_off[None, :]
        x_all = x[:, None] + dx_off[None, :]
        spatial_valid = (y_all >= 0) & (y_all < self.ny) & (x_all >= 0) & (x_all < self.nx)
        y_safe = np.clip(y_all, 0, self.ny - 1)
        x_safe = np.clip(x_all, 0, self.nx - 1)

        if apply_obs_mask:
            # Match inference: only positions on the global spatial obs mask
            # contribute observed values; everything else is padded zero.
            obs_at_pos = self.obs_mask_spatial[y_safe, x_safe]
            spatial_valid = spatial_valid & obs_at_pos

        u_win = np.full((N, kk * n_time_slices), self.mask_pad_value, dtype=np.float32)
        m_win = np.zeros((N, kk * n_time_slices), dtype=np.float32)

        for p in range(n_time_slices):
            t_slice = t - p
            t_valid = t_slice >= 0
            t_safe = np.clip(t_slice, 0, self.nt - 1)
            valid = spatial_valid & t_valid[:, None]
            vals = self.U[t_safe[:, None], y_safe, x_safe]
            off = p * kk
            u_win[:, off:off + kk] = np.where(valid, vals, self.mask_pad_value)
            m_win[:, off:off + kk] = valid.astype(np.float32)

        return np.concatenate([coord_feats, u_win, m_win], axis=1).astype(np.float32)

    # =========================================================================
    # Boundary latent encoding
    # =========================================================================
    def _encode_and_align_patch_boundary_latents(self, patch_boundary_idx_tyx,
                                                  patch_boundary_global_boundary_idx_tyx,
                                                  patch_boundary_global_interior_idx_tyx,
                                                  training):
        # Boundary-encoder features must match the inference distribution
        # (neighborhood values masked outside the global spatial boundary).
        # Interior-encoder features keep the full neighborhood since the
        # interior encoder is only ever called during training.
        feats_bnd = self._stack_mask_patch_features_from_idx(
            patch_boundary_global_boundary_idx_tyx, apply_obs_mask=True)
        feats_int = self._stack_mask_patch_features_from_idx(
            patch_boundary_global_interior_idx_tyx, apply_obs_mask=False)
        lat_bnd = (self.boundary_encoder(tf.constant(feats_bnd, tf.float32), training=training)
                   if feats_bnd.shape[0] > 0 else tf.zeros((0, self.num_latentdim), tf.float32))
        lat_int = (self.interior_encoder(tf.constant(feats_int, tf.float32), training=training)
                   if feats_int.shape[0] > 0 else tf.zeros((0, self.num_latentdim), tf.float32))
        n_total = int(patch_boundary_idx_tyx.shape[0])
        if n_total == 0:
            return tf.zeros((0, self.num_latentdim), tf.float32)
        gb_mask = self.mask_boundary[
            patch_boundary_idx_tyx[:, 0], patch_boundary_idx_tyx[:, 1], patch_boundary_idx_tyx[:, 2]]
        idx_b = np.nonzero(gb_mask)[0].astype(np.int32)
        idx_i = np.nonzero(~gb_mask)[0].astype(np.int32)
        out = tf.zeros((n_total, self.num_latentdim), tf.float32)
        if idx_b.size > 0:
            out = tf.tensor_scatter_nd_update(out, tf.constant(idx_b)[:, None], lat_bnd)
        if idx_i.size > 0:
            out = tf.tensor_scatter_nd_update(out, tf.constant(idx_i)[:, None], lat_int)
        return out

    def get_latent_operator_matrix(self):
        B = self.a_chol_raw
        L = tf.linalg.band_part(B, -1, 0)
        diag = tf.linalg.diag_part(L)
        diag_pos = tf.nn.softplus(diag) + self._chol_eps
        L = tf.linalg.set_diag(L, diag_pos)
        return tf.matmul(L, L, transpose_b=True)

    # =========================================================================
    # PDE loss (homogeneous elliptic, no source term)
    # =========================================================================
    def compute_pde_loss(self, patch_center_spatial_y_index, patch_center_spatial_x_index,
                         patch_time_indices_in_patch, patch_boundary_idx_tyx,
                         latent_values_on_patch_boundary_aligned, alpha_recon=1.0):
        patch_boundary_idx_np = np.asarray(patch_boundary_idx_tyx, dtype=np.int32)
        patch_time_indices_np = np.asarray(patch_time_indices_in_patch, dtype=np.int32)

        y_min = int(patch_boundary_idx_np[:, 1].min())
        y_max = int(patch_boundary_idx_np[:, 1].max())
        x_min = int(patch_boundary_idx_np[:, 2].min())
        x_max = int(patch_boundary_idx_np[:, 2].max())
        ph = y_max - y_min + 1
        pw = x_max - x_min + 1

        spatial_int_mask = np.zeros((ph, pw), dtype=bool)
        if ph >= 3 and pw >= 3:
            spatial_int_mask[1:-1, 1:-1] = True
        int_local_yx = np.argwhere(spatial_int_mask).astype(np.int32)
        n_int = int(int_local_yx.shape[0])
        if n_int == 0:
            z = tf.constant(0.0, tf.float32)
            return z, z, z, z

        int_row_id = -np.ones((ph, pw), dtype=np.int32)
        for rid, (ly, lx) in enumerate(int_local_yx):
            int_row_id[ly, lx] = rid

        if int_row_id[int(patch_center_spatial_y_index - y_min),
                       int(patch_center_spatial_x_index - x_min)] < 0:
            z = tf.constant(0.0, tf.float32)
            return z, z, z, z

        # Build the patch Laplacian L_mat AND a flat list of (rid, nly, nlx)
        # boundary neighbours so we can vectorise the per-timestep RHS build.
        L_mat = np.zeros((n_int, n_int), dtype=np.float32)
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nbr_rid_list = []
        nbr_nly_list = []
        nbr_nlx_list = []
        for rid, (ly, lx) in enumerate(int_local_yx):
            L_mat[rid, rid] += 4.0
            for dy, dx in steps:
                nly, nlx = int(ly + dy), int(lx + dx)
                nr = int(int_row_id[nly, nlx])
                if nr >= 0:
                    L_mat[rid, nr] += -1.0
                else:
                    nbr_rid_list.append(rid)
                    nbr_nly_list.append(nly)
                    nbr_nlx_list.append(nlx)
        nbr_rid_arr = np.asarray(nbr_rid_list, dtype=np.int32)
        nbr_nly_arr = np.asarray(nbr_nly_list, dtype=np.int32)
        nbr_nlx_arr = np.asarray(nbr_nlx_list, dtype=np.int32)

        L_tf = tf.constant(L_mat, tf.float32)
        A = self.get_latent_operator_matrix()
        spd_loss = tf.constant(0.0, tf.float32)

        stiffness = tf.linalg.LinearOperatorKronecker([
            tf.linalg.LinearOperatorFullMatrix(L_tf),
            tf.linalg.LinearOperatorFullMatrix(A),
        ]).to_dense()

        if tf.reduce_any(tf.math.is_nan(stiffness)) or tf.reduce_any(tf.math.is_inf(stiffness)):
            z = tf.constant(0.0, tf.float32)
            return z, z, z, z

        try:
            L_chol = tf.linalg.cholesky(stiffness)
            use_chol = True
        except tf.errors.InvalidArgumentError:
            lu, p_lu = tf.linalg.lu(stiffness)
            use_chol = False

        lat_loss_acc = tf.constant(0.0, tf.float32)
        rec_loss_acc = tf.constant(0.0, tf.float32)
        n_valid = 0

        int_global_y = int_local_yx[:, 0] + y_min
        int_global_x = int_local_yx[:, 1] + x_min
        bnd_t = patch_boundary_idx_np[:, 0]
        bnd_y = patch_boundary_idx_np[:, 1]
        bnd_x = patch_boundary_idx_np[:, 2]
        bnd_latents = latent_values_on_patch_boundary_aligned

        for t_n in patch_time_indices_np:
            t_n = int(t_n)
            bmask = (bnd_t == t_n)
            b_idx = np.nonzero(bmask)[0].astype(np.int32)
            if b_idx.size == 0:
                continue

            b_y_t = bnd_y[b_idx]
            b_x_t = bnd_x[b_idx]
            n_b_t = int(b_idx.size)
            b_lat_t = tf.gather(bnd_latents, b_idx, axis=0)

            # Boundary lookup at this timestep: maps local (nly, nlx) -> j
            b_lookup = -np.ones((ph, pw), dtype=np.int32)
            b_lookup[(b_y_t - y_min).astype(np.int32),
                     (b_x_t - x_min).astype(np.int32)] = np.arange(n_b_t, dtype=np.int32)

            # Vectorised neighbour-to-boundary mapping. M_t[i, j] counts how
            # many boundary neighbours interior row i has at boundary index j
            # at this timestep.
            if nbr_rid_arr.size > 0:
                bj_flat = b_lookup[nbr_nly_arr, nbr_nlx_arr]
                keep = bj_flat >= 0
            else:
                keep = np.zeros(0, dtype=bool)

            if keep.any():
                rid_keep = nbr_rid_arr[keep]
                bj_keep = bj_flat[keep]
                M_t = np.zeros((n_int, n_b_t), dtype=np.float32)
                np.add.at(M_t, (rid_keep, bj_keep), 1.0)
                M_t_tf = tf.constant(M_t, dtype=tf.float32)
                # Latent RHS: (n_int, r) = (M_t @ b_lat_t) @ A^T (A symmetric)
                Mb = tf.matmul(M_t_tf, b_lat_t)
                rhs_mat = tf.matmul(Mb, A, transpose_b=True)
                # Bypass RHS: physical boundary value sums per interior row
                u_at_bnd_t = self.U[t_n, b_y_t.astype(np.int32),
                                    b_x_t.astype(np.int32)].astype(np.float32)
                rhs_phys = (M_t @ u_at_bnd_t).astype(np.float32)
            else:
                rhs_mat = tf.zeros((n_int, self.num_latentdim), tf.float32)
                rhs_phys = np.zeros(n_int, dtype=np.float32)

            rhs_vec = tf.reshape(rhs_mat, (-1, 1))

            if use_chol:
                sol_vec = tf.linalg.cholesky_solve(L_chol, rhs_vec)
            else:
                sol_vec = tf.linalg.lu_solve(lu, p_lu, rhs_vec)

            if tf.reduce_any(tf.math.is_nan(sol_vec)):
                continue

            sol = tf.reshape(sol_vec, (n_int, self.num_latentdim))

            # Decoder bypass: harmonic interpolation of physical boundary values
            u_interp = np.linalg.solve(L_mat, rhs_phys).astype(np.float32)
            u_interp_tf = tf.constant(u_interp.reshape(-1, 1), dtype=tf.float32)
            sol_bypass = tf.concat([sol, u_interp_tf], axis=1)

            int_t = np.full((n_int,), t_n, dtype=np.int32)
            idx_all = np.stack([int_t, int_global_y.astype(np.int32), int_global_x.astype(np.int32)], axis=1)
            int_feats = self._stack_mask_patch_features_from_idx(idx_all)
            lat_true = self.interior_encoder(tf.constant(int_feats, tf.float32), training=True)
            lat_loss_acc += tf.reduce_sum(tf.square(sol - lat_true))

            u_pred = self.decoder(sol_bypass, training=True)
            u_true = self.U[int_t, int_global_y, int_global_x].astype(np.float32).reshape(-1, 1)
            rec_loss_acc += tf.reduce_sum(tf.square(u_pred - tf.constant(u_true, tf.float32)))
            n_valid += 1

        if n_valid == 0:
            z = tf.constant(0.0, tf.float32)
            return z, z, z, z

        total_pts = tf.cast(n_valid * n_int, tf.float32)
        lat_loss = lat_loss_acc / total_pts
        rec_loss = rec_loss_acc / total_pts
        # Sum kernel_regularizer L2 penalties; they are NOT auto-collected by
        # custom training loops, so we add them explicitly.
        reg_losses = (self.interior_encoder.losses
                      + self.boundary_encoder.losses
                      + self.decoder.losses)
        reg_loss = tf.add_n(reg_losses) if len(reg_losses) > 0 else tf.constant(0.0, tf.float32)
        total = lat_loss + alpha_recon * rec_loss + spd_loss + reg_loss
        return total, lat_loss, rec_loss, spd_loss

    # =========================================================================
    # Training
    # =========================================================================
    def train(self, epochs, patch_dim, num_patches, clip_norm=GRAD_CLIP_NORM):
        loss_history = {"total": [], "latent": [], "recon": [], "spd": []}
        for epoch in range(epochs):
            self.create_patches(patch_dim=patch_dim, num_patches=num_patches)
            ep_total = ep_lat = ep_rec = ep_spd = 0.0

            for pk in range(num_patches):
                with tf.GradientTape() as tape:
                    lat_on_bnd = self._encode_and_align_patch_boundary_latents(
                        self.patch_boundary_idx[pk],
                        self.patch_boundary_global_boundary_idx[pk],
                        self.patch_boundary_global_interior_idx[pk],
                        training=True,
                    )
                    p_times = np.unique(self.patch_boundary_idx[pk][:, 0])
                    total, lat, rec, spd = self.compute_pde_loss(
                        patch_center_spatial_y_index=int(self.patch_center_idx[pk, 1]),
                        patch_center_spatial_x_index=int(self.patch_center_idx[pk, 2]),
                        patch_time_indices_in_patch=p_times,
                        patch_boundary_idx_tyx=self.patch_boundary_idx[pk],
                        latent_values_on_patch_boundary_aligned=lat_on_bnd,
                    )

                grads = tape.gradient(total, self.trainable_vars)
                loss_val = float(total.numpy())
                if np.isnan(loss_val) or np.isinf(loss_val):
                    continue
                grads = [(tf.zeros_like(v) if g is None else g) for g, v in zip(grads, self.trainable_vars)]
                if any(tf.reduce_any(tf.math.is_nan(g)).numpy() for g in grads):
                    continue
                grads, _ = tf.clip_by_global_norm(grads, clip_norm)
                self.optimizer.apply_gradients(zip(grads, self.trainable_vars))

                ep_total += loss_val
                ep_lat += float(lat.numpy())
                ep_rec += float(rec.numpy())
                ep_spd += float(spd.numpy())

            inv = 1.0 / float(num_patches)
            ep_total *= inv; ep_lat *= inv; ep_rec *= inv; ep_spd *= inv
            loss_history["total"].append(ep_total)
            loss_history["latent"].append(ep_lat)
            loss_history["recon"].append(ep_rec)
            loss_history["spd"].append(ep_spd)
            print(f"epoch {epoch+1:04d} | total {ep_total:.6e} | "
                  f"latent {ep_lat:.6e} | recon {ep_rec:.6e}")
        return loss_history

    # =========================================================================
    # Plotting
    # =========================================================================
    def plot_training_history(self, loss_history, save_path=None, show=False):
        import matplotlib.pyplot as plt
        epochs = range(1, len(loss_history["total"]) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, loss_history["total"], "k-", lw=2, label="Total", marker="o", markersize=3)
        ax.plot(epochs, loss_history["latent"], "b--", lw=1.5, label="Latent", marker="s", markersize=3)
        ax.plot(epochs, loss_history["recon"], "r--", lw=1.5, label="Recon", marker="^", markersize=3)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Training Loss History"); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    # =========================================================================
    # Reconstruction (independent per timestep — no autoregressive)
    # =========================================================================
    def reconstruct_field_at_timestep(self, t_index, use_fast_solver=False):
        t = int(t_index)

        # Use cached spatial structure (built in split_interior_boundary)
        interior_indices = self._interior_indices
        boundary_indices = self._boundary_indices
        num_interior = interior_indices.shape[0]
        num_boundary = boundary_indices.shape[0]
        obs_mask = self.obs_mask_spatial           # True = boundary
        K = self._K_sparse                         # cached sparse Laplacian (float64)
        bnd_contribs = self._bnd_contribs          # cached boundary neighbour lists
        M_pre = self._M_pre                        # cached diagonal preconditioner

        boundary_y = boundary_indices[:, 0].astype(np.int32)
        boundary_x = boundary_indices[:, 1].astype(np.int32)
        boundary_t = np.full(num_boundary, t, dtype=np.int32)
        u_boundary_raw = self.U_original[boundary_t, boundary_y, boundary_x].astype(np.float32)

        # Encode boundary — reuse _stack_mask_patch_features_from_idx (no duplication)
        idx_bnd = np.stack([boundary_t, boundary_y, boundary_x], axis=1).astype(np.int32)
        bnd_feats = self._stack_mask_patch_features_from_idx(idx_bnd, apply_obs_mask=True)
        bnd_latents = self.boundary_encoder(tf.constant(bnd_feats, tf.float32), training=False).numpy()
        bnd_latents = np.nan_to_num(bnd_latents, nan=0.0, posinf=NAN_CLAMP, neginf=-NAN_CLAMP)
        latent_dim = bnd_latents.shape[1]

        # A matrix (per-call since weights change during training; negligible cost)
        A_np = self.get_latent_operator_matrix().numpy().astype(np.float64)
        A_np = 0.5 * (A_np + A_np.T)
        eigvals, Q = np.linalg.eigh(A_np)
        if float(np.min(eigvals)) <= EPS_EIG:
            eigvals = eigvals + EPS_CHOL

        # Latent RHS: A @ sum(boundary latents for each interior neighbour)
        rhs = np.zeros((num_interior, latent_dim), dtype=np.float64)
        for rid in range(num_interior):
            bids = bnd_contribs[rid]
            if len(bids) > 0:
                bnd_lat_sum = bnd_latents[bids, :].astype(np.float64).sum(axis=0)
                rhs[rid, :] = A_np @ bnd_lat_sum
        rhs = np.nan_to_num(rhs, nan=0.0, posinf=NAN_CLAMP, neginf=-NAN_CLAMP)

        # Solve in eigenbasis (one CG per latent dimension, shared K and preconditioner)
        tol = CG_TOL_FAST if use_fast_solver else CG_TOL_ACCURATE
        maxiter = CG_MAXITER_FAST if use_fast_solver else CG_MAXITER_ACCURATE
        rhs_tilde = rhs @ Q
        lat_tilde = np.zeros_like(rhs_tilde, dtype=np.float64)
        for k_dim in range(latent_dim):
            b = rhs_tilde[:, k_dim] / float(eigvals[k_dim])
            if np.isnan(b).any() or np.isinf(b).any():
                continue
            try:
                xk, _ = cg(K, b, M=M_pre, tol=tol, maxiter=maxiter)
                lat_tilde[:, k_dim] = xk
            except Exception:
                pass

        latent_interior = (lat_tilde @ Q.T).astype(np.float32)

        # --- Decoder bypass: harmonic interpolation of standardised boundary values ---
        # Reuses same K and preconditioner — one additional CG solve for the scalar field
        bnd_u_std = self.U[t, boundary_y, boundary_x].astype(np.float64)
        rhs_phys = np.zeros(num_interior, dtype=np.float64)
        for rid in range(num_interior):
            bids = bnd_contribs[rid]
            if len(bids) > 0:
                rhs_phys[rid] = bnd_u_std[bids].sum()
        u_interp, _ = cg(K, rhs_phys, M=M_pre, tol=tol, maxiter=maxiter)
        u_interp = u_interp.astype(np.float32).reshape(-1, 1)

        # Decode with bypass
        latent_with_bypass = np.concatenate([latent_interior, u_interp], axis=1)
        u_int_std = self.decoder(tf.constant(latent_with_bypass, tf.float32), training=False).numpy().reshape(-1)
        interior_y = interior_indices[:, 0]
        interior_x = interior_indices[:, 1]
        u_int_pred = u_int_std * self.U_std[interior_y, interior_x] + self.U_mean[interior_y, interior_x]

        # Assemble full field
        u_pred = np.zeros((self.ny, self.nx), dtype=np.float32)
        u_true = self.U_original[t, :, :].astype(np.float32)
        u_pred[boundary_y, boundary_x] = u_boundary_raw
        u_pred[interior_y, interior_x] = u_int_pred

        u_err = np.abs(u_pred - u_true)
        int_err = u_err[interior_y, interior_x]

        return {
            "u_pred": u_pred, "u_true": u_true, "u_error": u_err,
            "boundary_mask": obs_mask, "interior_mask": ~obs_mask,
            "mse": float(np.mean(int_err**2)),
            "mae": float(np.mean(int_err)),
            "max_error": float(np.max(int_err)),
            "t_index": t,
        }

    # =========================================================================
    # Evaluation
    # =========================================================================
    def evaluate_on_test_timesteps(self, verbose=True):
        if len(self.test_time_indices) == 0:
            print("No test timesteps.")
            return {"mse": float("nan"), "mae": float("nan"), "per_step": []}

        print(f"\n[evaluate] {len(self.test_time_indices)} test timesteps...")
        per_step = []
        for i, t in enumerate(self.test_time_indices):
            res = self.reconstruct_field_at_timestep(int(t), use_fast_solver=True)
            per_step.append(res)
            if verbose and ((i + 1) % 10 == 0 or i == 0 or i == len(self.test_time_indices) - 1):
                print(f"  t={t:4d} | MAE {res['mae']:.4e} | MSE {res['mse']:.4e}")

        summary = {
            "mse": float(np.mean([r["mse"] for r in per_step])),
            "mae": float(np.mean([r["mae"] for r in per_step])),
            "max_error": float(np.mean([r["max_error"] for r in per_step])),
            "per_step": per_step,
        }
        if verbose:
            print(f"\n  [Test summary] Mean MAE {summary['mae']:.4e} | Mean MSE {summary['mse']:.4e}")
        return summary

    def save_test_results(self, save_path="test_results.pkl", precomputed_results=None):
        if precomputed_results is not None:
            results_list = precomputed_results
        else:
            summary = self.evaluate_on_test_timesteps()
            results_list = summary["per_step"]
        payload = {"test_time_indices": np.array(self.test_time_indices, dtype=np.int32), "results": results_list}
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved {len(results_list)} test results to {save_path}")

    # =========================================================================
    # Plotting
    # =========================================================================
    def plot_field_reconstruction(self, results, save_path=None, show=False):
        import matplotlib.pyplot as plt
        u_true = results["u_true"]
        u_pred = results["u_pred"]
        abs_err = np.abs(np.nan_to_num(results["u_error"], nan=0.0))
        t_index = results["t_index"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        vmin = min(np.percentile(u_true, 2), np.percentile(u_pred, 2))
        vmax = max(np.percentile(u_true, 98), np.percentile(u_pred, 98))

        ax = axes[0]
        im = ax.imshow(u_true, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"True Field (t={t_index})"); plt.colorbar(im, ax=ax)
        ax = axes[1]
        im = ax.imshow(u_pred, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title("Predicted Field"); plt.colorbar(im, ax=ax)
        ax = axes[2]
        ev = np.percentile(abs_err, 95)
        im = ax.imshow(abs_err, cmap="hot", origin="lower", aspect="auto", vmin=0, vmax=ev)
        ax.set_title("Absolute Error"); plt.colorbar(im, ax=ax)
        ax.text(0.02, 0.98,
                f"MSE: {results['mse']:.4e}\nMAE: {results['mae']:.4e}\nMax: {results['max_error']:.4e}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.suptitle(f"Field Reconstruction at t={t_index}", fontsize=14, y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_test_mae_over_time(self, test_results, save_path=None, show=False):
        import matplotlib.pyplot as plt
        t_indices = [r["t_index"] for r in test_results]
        maes = [r["mae"] for r in test_results]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_indices, maes, "b-o", markersize=3, lw=1.5)
        ax.set_xlabel("Timestep"); ax.set_ylabel("MAE")
        ax.set_title("Test MAE over time (independent per-timestep)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig



# NOTE: The __main__ block has been removed.
# Use run_v8_bypass.py as the entry point — it reads from config.py
# and handles the full 3-year dataset correctly.
if __name__ == "__main__":
    raise SystemExit(
        "Do not run v8_bypass_sinn.py directly.\n"
        "Use:  python run_v8_bypass.py\n"
    )
