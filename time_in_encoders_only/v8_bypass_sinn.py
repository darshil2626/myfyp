import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Dense, Input
from keras.models import Model
import pickle
import matplotlib

matplotlib.use("Agg")


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

        self.mask_radius = 1
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
    def standardise_u(self, eps: float = 1e-8, time_indices=None):
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

        self._chol_eps = tf.constant(1e-6, dtype=tf.float32)
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
    def _stack_mask_patch_features_from_idx(self, idx_tyx, radius=None):
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
        feats_bnd = self._stack_mask_patch_features_from_idx(patch_boundary_global_boundary_idx_tyx)
        feats_int = self._stack_mask_patch_features_from_idx(patch_boundary_global_interior_idx_tyx)
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

        L_mat = np.zeros((n_int, n_int), dtype=np.float32)
        bnd_nbrs = [[] for _ in range(n_int)]
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for rid, (ly, lx) in enumerate(int_local_yx):
            L_mat[rid, rid] += 4.0
            for dy, dx in steps:
                nly, nlx = int(ly + dy), int(lx + dx)
                nr = int(int_row_id[nly, nlx])
                if nr >= 0:
                    L_mat[rid, nr] += -1.0
                else:
                    bnd_nbrs[rid].append((nly, nlx))

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
            b_lat_t = tf.gather(bnd_latents, b_idx, axis=0)

            b_lookup = -np.ones((ph, pw), dtype=np.int32)
            for j, (yy, xx) in enumerate(zip(b_y_t, b_x_t)):
                b_lookup[int(yy - y_min), int(xx - x_min)] = j

            rhs_blocks = []
            for rid in range(n_int):
                rhs_r = tf.zeros((self.num_latentdim,), tf.float32)
                for (nly, nlx) in bnd_nbrs[rid]:
                    bj = int(b_lookup[nly, nlx])
                    if bj >= 0:
                        rhs_r = rhs_r + tf.linalg.matvec(A, b_lat_t[bj, :])
                rhs_blocks.append(rhs_r)

            rhs_vec = tf.concat([tf.reshape(v, (self.num_latentdim, 1)) for v in rhs_blocks], axis=0)

            if use_chol:
                sol_vec = tf.linalg.cholesky_solve(L_chol, rhs_vec)
            else:
                sol_vec = tf.linalg.lu_solve(lu, p_lu, rhs_vec)

            if tf.reduce_any(tf.math.is_nan(sol_vec)):
                continue

            sol = tf.reshape(sol_vec, (n_int, self.num_latentdim))

            # --- Decoder bypass: harmonic interpolation of physical boundary values ---
            # Solve scalar Laplace L @ u_interp = rhs_physical using same
            # Laplacian and boundary structure. This gives the decoder direct
            # access to what boundary interpolation alone would predict,
            # so it only needs to learn the residual correction.
            rhs_phys = np.zeros(n_int, dtype=np.float32)
            for rid in range(n_int):
                for (nly, nlx) in bnd_nbrs[rid]:
                    bj = int(b_lookup[nly, nlx])
                    if bj >= 0:
                        rhs_phys[rid] += self.U[t_n, int(b_y_t[bj]), int(b_x_t[bj])]
            u_interp = np.linalg.solve(L_mat, rhs_phys).astype(np.float32)
            u_interp_tf = tf.constant(u_interp.reshape(-1, 1), dtype=tf.float32)

            # Append bypass to latent: decoder sees [latent_field, harmonic_interp]
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
        total = lat_loss + alpha_recon * rec_loss + spd_loss
        return total, lat_loss, rec_loss, spd_loss

    # =========================================================================
    # Training
    # =========================================================================
    def train(self, epochs, patch_dim, num_patches, clip_norm=1.0):
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
        import scipy.sparse as sp
        from scipy.sparse.linalg import cg
        from scipy.sparse import diags

        t = int(t_index)
        b_thick = int(self.b_thick)

        obs_mask = np.zeros((self.ny, self.nx), dtype=bool)
        obs_mask[:b_thick, :] = True; obs_mask[-b_thick:, :] = True
        obs_mask[:, :b_thick] = True; obs_mask[:, -b_thick:] = True

        interior_indices = np.argwhere(~obs_mask)
        boundary_indices = np.argwhere(obs_mask)
        num_interior = interior_indices.shape[0]
        num_boundary = boundary_indices.shape[0]

        boundary_y = boundary_indices[:, 0].astype(np.int32)
        boundary_x = boundary_indices[:, 1].astype(np.int32)
        boundary_t = np.full(num_boundary, t, dtype=np.int32)
        u_boundary_raw = self.U_original[boundary_t, boundary_y, boundary_x].astype(np.float32)

        # --- Vectorised boundary feature builder ---
        def stack_observed_only(idx_tyx):
            idx_tyx = np.asarray(idx_tyx, dtype=np.int32)
            N = idx_tyx.shape[0]
            r = int(self.mask_radius)
            win = 2 * r + 1; ws = win * win
            nts = 1 + self.n_past_steps

            tt = idx_tyx[:, 0]; yy = idx_tyx[:, 1]; xx = idx_tyx[:, 2]
            t_norm = tt.astype(np.float32) / max(self.nt - 1, 1)
            y_norm = yy.astype(np.float32) / max(self.ny - 1, 1)
            x_norm = xx.astype(np.float32) / max(self.nx - 1, 1)
            coord_feats = self._fourier_embed_coords(t_norm, y_norm, x_norm)

            dy_off = np.repeat(np.arange(-r, r + 1), win).astype(np.int32)
            dx_off = np.tile(np.arange(-r, r + 1), win).astype(np.int32)
            y_all = yy[:, None] + dy_off[None, :]
            x_all = xx[:, None] + dx_off[None, :]
            sv = (y_all >= 0) & (y_all < self.ny) & (x_all >= 0) & (x_all < self.nx)
            ys = np.clip(y_all, 0, self.ny - 1)
            xs = np.clip(x_all, 0, self.nx - 1)
            obs_v = sv & obs_mask[ys, xs]

            u_win = np.zeros((N, ws * nts), dtype=np.float32)
            m_win = np.zeros((N, ws * nts), dtype=np.float32)
            for p in range(nts):
                ts = tt - p
                tv = ts >= 0
                tsa = np.clip(ts, 0, self.nt - 1)
                valid = obs_v & tv[:, None]
                vals = self.U[tsa[:, None], ys, xs]
                off = p * ws
                u_win[:, off:off + ws] = np.where(valid, vals, 0.0)
                m_win[:, off:off + ws] = valid.astype(np.float32)
            return np.concatenate([coord_feats, u_win, m_win], axis=1).astype(np.float32)

        # Encode boundary
        idx_bnd = np.stack([boundary_t, boundary_y, boundary_x], axis=1).astype(np.int32)
        bnd_feats = stack_observed_only(idx_bnd)
        bnd_latents = self.boundary_encoder(tf.constant(bnd_feats, tf.float32), training=False).numpy()
        bnd_latents = np.nan_to_num(bnd_latents, nan=0.0, posinf=1e3, neginf=-1e3)
        latent_dim = bnd_latents.shape[1]

        # Build Laplacian
        interior_row_map = -np.ones((self.ny, self.nx), dtype=np.int32)
        for rid, (y, x) in enumerate(interior_indices):
            interior_row_map[y, x] = rid
        bnd_map = {(int(y), int(x)): i for i, (y, x) in enumerate(boundary_indices)}
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        rows, cols, vals = [], [], []
        bnd_contribs = [[] for _ in range(num_interior)]
        for rid, (y, x) in enumerate(interior_indices):
            rows.append(rid); cols.append(rid); vals.append(4.0)
            for dy, dx in steps:
                ny_, nx_ = int(y + dy), int(x + dx)
                nr = interior_row_map[ny_, nx_]
                if nr >= 0:
                    rows.append(rid); cols.append(int(nr)); vals.append(-1.0)
                else:
                    bi = bnd_map.get((ny_, nx_))
                    if bi is not None:
                        bnd_contribs[rid].append(bi)

        K = sp.csr_matrix((vals, (rows, cols)), shape=(num_interior, num_interior), dtype=np.float32)

        # A matrix
        A_np = self.get_latent_operator_matrix().numpy().astype(np.float64)
        A_np = 0.5 * (A_np + A_np.T)
        eigvals, Q = np.linalg.eigh(A_np)
        if float(np.min(eigvals)) <= 1e-8:
            eigvals = eigvals + 1e-6

        # RHS (vectorised)
        rhs = np.zeros((num_interior, latent_dim), dtype=np.float64)
        for rid in range(num_interior):
            bids = bnd_contribs[rid]
            if len(bids) > 0:
                bnd_lat_sum = bnd_latents[bids, :].astype(np.float64).sum(axis=0)
                rhs[rid, :] = A_np @ bnd_lat_sum
        rhs = np.nan_to_num(rhs, nan=0.0, posinf=1e3, neginf=-1e3)

        # Solve in eigenbasis
        rhs_tilde = rhs @ Q
        diag_K = np.array(K.sum(axis=1)).ravel()
        diag_K[diag_K == 0] = 1.0
        M_pre = diags(1.0 / diag_K, dtype=np.float64, format='csr')

        tol = 1e-2 if use_fast_solver else 1e-4
        maxiter = 1000 if use_fast_solver else 5000

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

        # --- Decoder bypass: harmonic interpolation of physical boundary values ---
        # Solve scalar Laplace K @ u_interp = rhs_phys for standardised boundary values
        bnd_u_std = self.U[t, boundary_y, boundary_x].astype(np.float64)
        rhs_phys = np.zeros(num_interior, dtype=np.float64)
        for rid in range(num_interior):
            bids = bnd_contribs[rid]
            if len(bids) > 0:
                rhs_phys[rid] = bnd_u_std[bids].sum()

        # Solve with CG (same preconditioner)
        u_interp, _ = cg(K.astype(np.float64), rhs_phys, M=M_pre, tol=tol, maxiter=maxiter)
        u_interp = u_interp.astype(np.float32).reshape(-1, 1)

        # Append bypass to latent
        latent_with_bypass = np.concatenate([latent_interior, u_interp], axis=1)

        # Decode
        u_int_std = self.decoder(tf.constant(latent_with_bypass, tf.float32), training=False).numpy().reshape(-1)
        interior_y = interior_indices[:, 0]
        interior_x = interior_indices[:, 1]
        u_int_pred = u_int_std * self.U_std[interior_y, interior_x] + self.U_mean[interior_y, interior_x]

        # Assemble
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


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # ---- Config ----
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
    epochs = 50
    n_past_steps = 5
    train_fraction = 0.8

    # ---- Load data ----
    data_path = r"training_data_speed.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    X = data["X"]; Y = data["Y"]; U = data["U"]; T = data["T"]
    n_keep = 500
    U = U[-n_keep:]; T = T[-n_keep:]
    print(f"Using last {n_keep} timesteps. U shape: {np.asarray(U).shape}")

    # ---- Setup ----
    solver = sinn(X, Y, U, T, debug=False)
    solver.split_train_test_timesteps(mode="sequential", train_frac=train_fraction)
    solver.standardise_u(time_indices=solver.train_time_indices)
    solver.split_interior_boundary(b_thick, include_t0, include_tT)
    solver.build_models(num_latentdim, num_units, num_layers, dropout, l2_reg, lr,
                        n_past_steps=n_past_steps)

    # ---- Train ----
    loss_history = solver.train(epochs, patch_dim, num_patches)
    print(f"\nTraining complete! Final loss: {loss_history['total'][-1]:.6e}")
    solver.plot_training_history(loss_history, save_path="training_loss_v3bp.png", show=False)

    # ---- Training sanity check ----
    train_mid = int(solver.train_time_indices[len(solver.train_time_indices) // 2])
    res_train = solver.reconstruct_field_at_timestep(train_mid)
    solver.plot_field_reconstruction(res_train, save_path=f"field_train_t{train_mid}.png")
    print(f"[Train t={train_mid}] MAE: {res_train['mae']:.6e}")

    # ---- Test evaluation (independent per timestep) ----
    test_summary = solver.evaluate_on_test_timesteps()
    print(f"\n[Test] Mean MAE: {test_summary['mae']:.6e}")
    print(f"[Test] Mean MSE: {test_summary['mse']:.6e}")

    solver.plot_test_mae_over_time(test_summary["per_step"], save_path="test_mae_over_time_v3bp.png")

    test_steps = test_summary["per_step"]
    for label, idx in [("early", 0), ("mid", len(test_steps)//2), ("late", -1)]:
        res = test_steps[idx]
        solver.plot_field_reconstruction(res, save_path=f"field_test_{label}_t{res['t_index']}.png")

    solver.save_test_results(save_path="test_results_v3bp.pkl",
                              precomputed_results=test_summary["per_step"])

    A_final = solver.get_latent_operator_matrix().numpy()
    print(f"\nA matrix (3x3):\n{A_final[:3, :3]}")
    print(f"Frobenius dist from I: {np.linalg.norm(A_final - np.eye(solver.num_latentdim)):.4f}")
    print(f"Eigenvalues: {np.linalg.eigvalsh(A_final)}")
