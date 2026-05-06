import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Dense, Input
from keras.models import Model
import pickle


class sinn:
    """
    Structure-Informed Neural Network (SINN) with temporal source term
    and multi-scale Fourier feature embeddings.

    Extension of v4 with Fourier feature coordinate embeddings that
    replace raw (t, y, x) normalised coordinates with:

        gamma(z) = [sin(2*pi*B*z), cos(2*pi*B*z)]

    where B is a random frequency matrix sampled at multiple scales.
    This directly addresses spectral bias — the tendency of neural
    networks to learn low-frequency components first and struggle
    with high-frequency structure.
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

        # --- Fourier feature embedding ---
        # Multi-scale random Fourier features replace raw (t,y,x) coordinates.
        # Frequencies are sampled from multiple scales to capture both
        # large-scale gradients and fine-scale structure.
        self.n_fourier_frequencies = 32  # total frequencies -> 64 features (sin+cos)
        self.fourier_scales = [1.0, 5.0, 10.0, 20.0, 50.0]
        self._init_fourier_B()

    # =========================================================================
    # Fourier feature embedding
    # =========================================================================
    def _init_fourier_B(self):
        """
        Initialise the random frequency matrix B for Fourier feature embedding.

        B has shape (n_fourier_frequencies, 3) where 3 = (t, y, x).
        Frequencies are drawn from N(0, sigma^2) for multiple sigma values,
        giving the network explicit access to multiple spatial/temporal scales.
        """
        rng = np.random.RandomState(42)  # fixed seed for reproducibility
        n_per_scale = self.n_fourier_frequencies // len(self.fourier_scales)
        n_remainder = self.n_fourier_frequencies - n_per_scale * len(self.fourier_scales)

        blocks = []
        for i, sigma in enumerate(self.fourier_scales):
            n = n_per_scale + (1 if i < n_remainder else 0)
            blocks.append(rng.randn(n, 3).astype(np.float32) * sigma)

        self.fourier_B = np.concatenate(blocks, axis=0)  # (n_freq, 3)
        self.coord_embed_dim = 2 * self.n_fourier_frequencies  # sin + cos

    def _fourier_embed_coords(self, t_norm, y_norm, x_norm):
        """
        Map normalised coordinates to Fourier features.

        Input: t_norm, y_norm, x_norm — each shape (N,)
        Output: (N, 2 * n_fourier_frequencies) array of [sin(2π B z), cos(2π B z)]
        """
        N = len(t_norm)
        z = np.stack([t_norm, y_norm, x_norm], axis=1).astype(np.float32)  # (N, 3)
        proj = z @ self.fourier_B.T  # (N, n_freq)
        proj = 2.0 * np.pi * proj
        return np.concatenate([np.sin(proj), np.cos(proj)], axis=1).astype(np.float32)  # (N, 2*n_freq)

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
            print(f"[standardise_u] Warning: input has {n_nan} NaN, {n_inf} Inf")
            U_ref = np.nan_to_num(U_ref, nan=0.0, posinf=1e6, neginf=-1e6)
            self.U = np.nan_to_num(self.U, nan=0.0, posinf=1e6, neginf=-1e6)

        self.U_mean = np.mean(U_ref, axis=0)
        self.U_std = np.std(U_ref, axis=0)
        self.U_std = np.maximum(self.U_std, eps)
        self.U = (self.U - self.U_mean) / self.U_std

        n_nan_out = np.isnan(self.U).sum()
        if n_nan_out > 0:
            print(f"[standardise_u] Warning: {n_nan_out} NaN after standardisation")
            self.U = np.nan_to_num(self.U, nan=0.0)

    def unstandardise_u(self, U_pred):
        return U_pred * self.U_std + self.U_mean

    # =========================================================================
    # Masks
    # =========================================================================
    def split_interior_boundary(self, b_thick: int, include_t0: bool = True, include_tT: bool = True):
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

        mask_int = ~mask_bnd

        safe = np.zeros((self.nt, self.ny, self.nx), dtype=bool)
        safe[1:-1, 1:-1, 1:-1] = True
        safe[:, :self.b_thick, :] = False
        safe[:, -self.b_thick:, :] = False
        safe[:, :, :self.b_thick] = False
        safe[:, :, -self.b_thick:] = False

        self.mask_boundary = mask_bnd
        self.mask_fd_safe_interior = safe & mask_int

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
        # Fourier features replace 3 raw coords with coord_embed_dim features
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
        self.decoder = self._build_coder(
            num_units, num_layers, input_shape=self.num_latentdim,
            output_shape=1,
            name="decoder", dropout=dropout, l2_reg=l2_reg,
        )

        # --- Source network G ---
        # Input: Fourier coords + k*k window of r-dim latent values + k*k mask
        kk = k * k
        self.source_input_dim = self.coord_embed_dim + kk * self.num_latentdim + kk
        self.source_network = self._build_coder(
            num_units, num_layers,
            input_shape=self.source_input_dim,
            output_shape=self.num_latentdim,
            name="source_network", dropout=dropout, l2_reg=l2_reg,
        )

        # --- SPD latent operator via Cholesky ---
        self._chol_eps = tf.constant(1e-6, dtype=tf.float32)
        init_B = np.eye(self.num_latentdim, dtype=np.float32)
        self.a_chol_raw = tf.Variable(init_B, trainable=True, dtype=tf.float32, name="a_chol_raw")

        self.trainable_vars = (
            self.interior_encoder.trainable_variables
            + self.boundary_encoder.trainable_variables
            + self.decoder.trainable_variables
            + self.source_network.trainable_variables
            + [self.a_chol_raw]
        )

        print(f"Trainable variables: {len(self.trainable_vars)}")
        print(f"  Source network input dim: {self.source_input_dim}")
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    # =========================================================================
    # Train / test split
    # =========================================================================
    def split_train_test_timesteps(self, mode: str = "sequential", train_frac: float = 0.8):
        """
        mode='sequential': First train_frac of timesteps for training, rest for testing.
        mode='alternate':  Even indices train, odd test.
        mode='all':        Everything for training.
        """
        all_t = np.arange(self.nt, dtype=np.int32)
        if mode == "sequential":
            n_train = int(self.nt * train_frac)
            n_train = max(1, min(n_train, self.nt - 1))
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
            raise ValueError("Patch dimensions too large for the grid.")

        valid_train_t = self.train_time_indices[
            (self.train_time_indices >= t_min) & (self.train_time_indices <= t_max)
        ]
        if len(valid_train_t) == 0:
            raise ValueError("No valid training time indices for patch half-width.")

        num_boundary_patches = int(num_patches * boundary_fraction)

        self.patch_center_idx = np.zeros((num_patches, 3), dtype=np.int32)
        self.patch_interior_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_global_boundary_idx = np.empty(num_patches, dtype=object)
        self.patch_boundary_global_interior_idx = np.empty(num_patches, dtype=object)

        patch_bnd_mask = np.zeros((pt, py, px), dtype=bool)
        patch_bnd_mask[0, :, :] = True
        patch_bnd_mask[-1, :, :] = True
        patch_bnd_mask[:, 0, :] = True
        patch_bnd_mask[:, -1, :] = True
        patch_bnd_mask[:, :, 0] = True
        patch_bnd_mask[:, :, -1] = True
        patch_int_mask = ~patch_bnd_mask

        t_off = np.arange(-t_left, t_right + 1, dtype=np.int32)
        y_off = np.arange(-y_left, y_right + 1, dtype=np.int32)
        x_off = np.arange(-x_left, x_right + 1, dtype=np.int32)
        TT, YY, XX = np.meshgrid(t_off, y_off, x_off, indexing="ij")
        offsets = np.stack([TT.ravel(), YY.ravel(), XX.ravel()], axis=1)
        bnd_offsets = offsets[patch_bnd_mask.ravel()]
        int_offsets = offsets[patch_int_mask.ravel()]

        faces = ['t0', 'tT', 'x_left', 'x_right', 'y_bottom', 'y_top']

        for p in range(num_patches):
            if p < num_boundary_patches:
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
            p_bnd = centre + bnd_offsets
            p_int = centre + int_offsets

            is_gb = self.mask_boundary[p_bnd[:, 0], p_bnd[:, 1], p_bnd[:, 2]]
            self.patch_interior_idx[p] = p_int
            self.patch_boundary_idx[p] = p_bnd
            self.patch_boundary_global_boundary_idx[p] = p_bnd[is_gb]
            self.patch_boundary_global_interior_idx[p] = p_bnd[~is_gb]

    # =========================================================================
    # Feature stacking (encoder inputs)
    # =========================================================================
    def _stack_mask_patch_features_from_idx(self, idx_tyx, radius=None):
        """Vectorised feature stacking — no Python loops over points."""
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

        # Precompute spatial offsets: shape (kk,)
        dy_off = np.repeat(np.arange(-r, r + 1), k).astype(np.int32)
        dx_off = np.tile(np.arange(-r, r + 1), k).astype(np.int32)

        # All neighbor coords: shape (N, kk)
        y_all = y[:, None] + dy_off[None, :]
        x_all = x[:, None] + dx_off[None, :]
        spatial_valid = (y_all >= 0) & (y_all < self.ny) & (x_all >= 0) & (x_all < self.nx)

        # Clip for safe indexing (invalid entries masked out later)
        y_safe = np.clip(y_all, 0, self.ny - 1)
        x_safe = np.clip(x_all, 0, self.nx - 1)

        u_win = np.full((N, kk * n_time_slices), self.mask_pad_value, dtype=np.float32)
        m_win = np.zeros((N, kk * n_time_slices), dtype=np.float32)

        for p in range(n_time_slices):
            t_slice = t - p  # (N,)
            t_valid = t_slice >= 0  # (N,)
            t_safe = np.clip(t_slice, 0, self.nt - 1)

            # Combined mask: (N, kk)
            valid = spatial_valid & t_valid[:, None]

            # Vectorised U lookup: (N, kk)
            vals = self.U[t_safe[:, None], y_safe, x_safe]

            off = p * kk
            u_win[:, off:off + kk] = np.where(valid, vals, self.mask_pad_value)
            m_win[:, off:off + kk] = valid.astype(np.float32)

        return np.concatenate([coord_feats, u_win, m_win], axis=1).astype(np.float32)

    # =========================================================================
    # Source feature builder (operates on latent fields, not physical data)
    # =========================================================================
    def _build_source_features(self, prev_latent_np, interior_local_yx,
                                t_index, y_min, x_min, height, width):
        """Vectorised source features for patch-level training."""
        r = self.mask_radius
        k = 2 * r + 1
        kk = k * k
        rdim = self.num_latentdim
        N = interior_local_yx.shape[0]

        # Lookup table
        lookup = -np.ones((height, width), dtype=np.int32)
        for row_id, (ly, lx) in enumerate(interior_local_yx):
            lookup[ly, lx] = row_id

        # Offsets
        dy_off = np.repeat(np.arange(-r, r + 1), k).astype(np.int32)
        dx_off = np.tile(np.arange(-r, r + 1), k).astype(np.int32)

        ly = interior_local_yx[:, 0]
        lx = interior_local_yx[:, 1]
        y_all = ly[:, None] + dy_off[None, :]  # (N, kk)
        x_all = lx[:, None] + dx_off[None, :]

        valid = (y_all >= 0) & (y_all < height) & (x_all >= 0) & (x_all < width)
        y_safe = np.clip(y_all, 0, height - 1)
        x_safe = np.clip(x_all, 0, width - 1)

        rows = lookup[y_safe, x_safe]  # (N, kk)
        has_row = valid & (rows >= 0)

        # Gather latent values: for valid entries, index into prev_latent_np
        rows_safe = np.clip(rows, 0, N - 1)
        gathered = prev_latent_np[rows_safe]  # (N, kk, rdim)
        mask_3d = has_row[:, :, None]  # (N, kk, 1)
        latent_win = (gathered * mask_3d).reshape(N, kk * rdim).astype(np.float32)
        mask_win = has_row.astype(np.float32)

        t_norm = np.full((N,), t_index / self.nt_norm, dtype=np.float32)
        y_norm = (interior_local_yx[:, 0] + y_min).astype(np.float32) / self.ny_norm
        x_norm = (interior_local_yx[:, 1] + x_min).astype(np.float32) / self.nx_norm
        coord_feats = self._fourier_embed_coords(t_norm, y_norm, x_norm)

        return np.concatenate([coord_feats, latent_win, mask_win], axis=1).astype(np.float32)

    def _build_source_features_global(self, prev_latent_np, interior_indices,
                                       interior_row_map, t_index):
        """Vectorised source features for full-domain inference."""
        r = self.mask_radius
        k = 2 * r + 1
        kk = k * k
        rdim = self.num_latentdim
        N = interior_indices.shape[0]

        dy_off = np.repeat(np.arange(-r, r + 1), k).astype(np.int32)
        dx_off = np.tile(np.arange(-r, r + 1), k).astype(np.int32)

        yi = interior_indices[:, 0]
        xi = interior_indices[:, 1]
        y_all = yi[:, None] + dy_off[None, :]  # (N, kk)
        x_all = xi[:, None] + dx_off[None, :]

        valid = (y_all >= 0) & (y_all < self.ny) & (x_all >= 0) & (x_all < self.nx)
        y_safe = np.clip(y_all, 0, self.ny - 1)
        x_safe = np.clip(x_all, 0, self.nx - 1)

        rows = interior_row_map[y_safe, x_safe]  # (N, kk)
        has_row = valid & (rows >= 0)

        rows_safe = np.clip(rows, 0, N - 1)
        gathered = prev_latent_np[rows_safe]  # (N, kk, rdim)
        mask_3d = has_row[:, :, None]
        latent_win = (gathered * mask_3d).reshape(N, kk * rdim).astype(np.float32)
        mask_win = has_row.astype(np.float32)

        t_norm = np.full((N,), t_index / self.nt_norm, dtype=np.float32)
        y_norm = interior_indices[:, 0].astype(np.float32) / self.ny_norm
        x_norm = interior_indices[:, 1].astype(np.float32) / self.nx_norm
        coord_feats = self._fourier_embed_coords(t_norm, y_norm, x_norm)

        return np.concatenate([coord_feats, latent_win, mask_win], axis=1).astype(np.float32)

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

        tb = patch_boundary_idx_tyx[:, 0]; yb = patch_boundary_idx_tyx[:, 1]; xb = patch_boundary_idx_tyx[:, 2]
        gb_mask = self.mask_boundary[tb, yb, xb]
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
    # PDE loss with source term
    # =========================================================================
    def compute_pde_loss(self, patch_center_spatial_y_index, patch_center_spatial_x_index,
                         patch_time_indices_in_patch, patch_boundary_idx_tyx,
                         latent_values_on_patch_boundary_aligned,
                         alpha_recon=1.0, teacher_forcing_prob=1.0):
        """
        Compute loss with inhomogeneous elliptic PDE: div(A grad ell) = s.

        Time slices are processed sequentially. The source at each slice
        comes from the previous slice's latent field, selected by
        teacher_forcing_prob (curriculum learning).
        """
        patch_boundary_idx_np = np.asarray(patch_boundary_idx_tyx, dtype=np.int32)
        patch_time_indices_np = np.sort(np.asarray(patch_time_indices_in_patch, dtype=np.int32))

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

        pc_ly = int(patch_center_spatial_y_index - y_min)
        pc_lx = int(patch_center_spatial_x_index - x_min)
        if int_row_id[pc_ly, pc_lx] < 0:
            z = tf.constant(0.0, tf.float32)
            return z, z, z, z

        # Build Laplacian + boundary neighbour lists
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

        prev_latent = None  # (n_int, r) from previous time slice

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

            # RHS from boundary contributions
            rhs_blocks = []
            for rid in range(n_int):
                rhs_r = tf.zeros((self.num_latentdim,), tf.float32)
                for (nly, nlx) in bnd_nbrs[rid]:
                    bj = int(b_lookup[nly, nlx])
                    if bj >= 0:
                        rhs_r = rhs_r + tf.linalg.matvec(A, b_lat_t[bj, :])
                rhs_blocks.append(rhs_r)

            rhs_vec = tf.concat([tf.reshape(v, (self.num_latentdim, 1)) for v in rhs_blocks], axis=0)

            # --- Source term from previous latent field ---
            if prev_latent is not None:
                if hasattr(prev_latent, 'numpy'):
                    prev_np = prev_latent.numpy()
                else:
                    prev_np = np.asarray(prev_latent)

                src_feats = self._build_source_features(
                    prev_np, int_local_yx, t_n, y_min, x_min, ph, pw
                )
                src_vals = self.source_network(tf.constant(src_feats, tf.float32), training=True)
                # Add source to RHS (same Kronecker ordering)
                src_vec = tf.concat(
                    [tf.reshape(src_vals[i, :], (self.num_latentdim, 1)) for i in range(n_int)],
                    axis=0
                )
                rhs_vec = rhs_vec + src_vec

            # Solve
            if use_chol:
                sol_vec = tf.linalg.cholesky_solve(L_chol, rhs_vec)
            else:
                sol_vec = tf.linalg.lu_solve(lu, p_lu, rhs_vec)

            if tf.reduce_any(tf.math.is_nan(sol_vec)):
                continue

            sol = tf.reshape(sol_vec, (n_int, self.num_latentdim))

            # Latent consistency with interior encoder
            int_t = np.full((n_int,), t_n, dtype=np.int32)
            idx_all = np.stack([int_t, int_global_y.astype(np.int32), int_global_x.astype(np.int32)], axis=1)
            int_feats = self._stack_mask_patch_features_from_idx(idx_all)
            lat_true = self.interior_encoder(tf.constant(int_feats, tf.float32), training=True)

            lat_loss_acc += tf.reduce_sum(tf.square(sol - lat_true))

            # Reconstruction loss
            u_pred = self.decoder(sol, training=True)
            u_true = self.U[int_t, int_global_y, int_global_x].astype(np.float32).reshape(-1, 1)
            rec_loss_acc += tf.reduce_sum(tf.square(u_pred - tf.constant(u_true, tf.float32)))

            # --- Curriculum: choose prev_latent for next slice ---
            use_tf = (np.random.rand() < teacher_forcing_prob)
            if use_tf:
                prev_latent = lat_true  # teacher forcing
            else:
                prev_latent = sol       # autoregressive

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
    def train(self, epochs, patch_dim, num_patches, clip_norm=1.0,
              tf_warmup_epochs=None):
        """
        Train with curriculum learning.

        tf_warmup_epochs: number of epochs of pure teacher forcing before
                          annealing. Defaults to epochs // 3.
        """
        if tf_warmup_epochs is None:
            tf_warmup_epochs = max(1, epochs // 3)

        loss_history = {"total": [], "latent": [], "recon": [], "spd": []}

        for epoch in range(epochs):
            # Curriculum schedule: linear anneal from 1.0 to 0.0
            if epoch < tf_warmup_epochs:
                tf_prob = 1.0
            else:
                tf_prob = max(0.0, 1.0 - (epoch - tf_warmup_epochs) / max(1, epochs - tf_warmup_epochs))

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
                        alpha_recon=1.0,
                        teacher_forcing_prob=tf_prob,
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
                  f"latent {ep_lat:.6e} | recon {ep_rec:.6e} | "
                  f"tf_prob {tf_prob:.2f}")

        return loss_history

    # =========================================================================
    # Plotting
    # =========================================================================
    def plot_training_history(self, loss_history, save_path=None, show=True):
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
        if show:
            plt.show()
        return fig

    # =========================================================================
    # Single-timestep reconstruction (with optional source)
    # =========================================================================
    def _reconstruct_single_timestep(self, t_index, prev_latent_interior=None,
                                      interior_indices=None, interior_row_map=None,
                                      boundary_indices=None, obs_mask=None,
                                      laplacian_sparse=None, boundary_contributions=None,
                                      A_np=None, eigvals=None, Q=None,
                                      use_fast_solver=False):
        """
        Reconstruct one timestep using boundary-only data + optional source.

        If prev_latent_interior is provided, computes source term from it.
        Otherwise solves the homogeneous equation (s=0).

        Returns dict with 'latent_interior' for feeding into the next step.
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import cg
        from scipy.sparse import diags

        t = int(t_index)
        b_thick = int(getattr(self, "b_thick", 1))
        num_interior = interior_indices.shape[0]
        num_boundary = boundary_indices.shape[0]
        latent_dim = self.num_latentdim

        # ---- Encode boundary ----
        boundary_y = boundary_indices[:, 0].astype(np.int32)
        boundary_x = boundary_indices[:, 1].astype(np.int32)
        boundary_t = np.full(num_boundary, t, dtype=np.int32)

        u_boundary_raw = self.U_original[boundary_t, boundary_y, boundary_x].astype(np.float32)

        def stack_observed_only(idx_tyx):
            """Vectorised inference feature builder — boundary-only observations."""
            idx_tyx = np.asarray(idx_tyx, dtype=np.int32)
            N = idx_tyx.shape[0]
            r = int(self.mask_radius)
            win = 2 * r + 1
            ws = win * win
            nts = 1 + self.n_past_steps

            t_norm = idx_tyx[:, 0].astype(np.float32) / max(self.nt - 1, 1)
            y_norm = idx_tyx[:, 1].astype(np.float32) / max(self.ny - 1, 1)
            x_norm = idx_tyx[:, 2].astype(np.float32) / max(self.nx - 1, 1)
            coord_feats = self._fourier_embed_coords(t_norm, y_norm, x_norm)

            t = idx_tyx[:, 0]; y = idx_tyx[:, 1]; x = idx_tyx[:, 2]
            dy_off = np.repeat(np.arange(-r, r + 1), win).astype(np.int32)
            dx_off = np.tile(np.arange(-r, r + 1), win).astype(np.int32)

            y_all = y[:, None] + dy_off[None, :]  # (N, ws)
            x_all = x[:, None] + dx_off[None, :]
            spatial_valid = (y_all >= 0) & (y_all < self.ny) & (x_all >= 0) & (x_all < self.nx)
            y_safe = np.clip(y_all, 0, self.ny - 1)
            x_safe = np.clip(x_all, 0, self.nx - 1)

            # obs_mask check (vectorised)
            obs_valid = spatial_valid & obs_mask[y_safe, x_safe]

            u_win = np.zeros((N, ws * nts), dtype=np.float32)
            m_win = np.zeros((N, ws * nts), dtype=np.float32)

            for p in range(nts):
                t_slice = t - p
                t_valid = t_slice >= 0
                t_safe = np.clip(t_slice, 0, self.nt - 1)
                valid = obs_valid & t_valid[:, None]
                vals = self.U[t_safe[:, None], y_safe, x_safe]
                off = p * ws
                u_win[:, off:off + ws] = np.where(valid, vals, 0.0)
                m_win[:, off:off + ws] = valid.astype(np.float32)

            return np.concatenate([coord_feats, u_win, m_win], axis=1).astype(np.float32)

        idx_bnd = np.stack([boundary_t, boundary_y, boundary_x], axis=1).astype(np.int32)
        bnd_feats = stack_observed_only(idx_bnd)
        bnd_latents = self.boundary_encoder(tf.constant(bnd_feats, tf.float32), training=False).numpy()
        bnd_latents = np.nan_to_num(bnd_latents, nan=0.0, posinf=1e3, neginf=-1e3)

        # ---- RHS from boundary (vectorised) ----
        rhs = np.zeros((num_interior, latent_dim), dtype=np.float64)
        for rid in range(num_interior):
            bids = boundary_contributions[rid]
            if len(bids) > 0:
                # Sum A @ latent for all boundary neighbours at once
                bnd_lat_sum = bnd_latents[bids, :].astype(np.float64).sum(axis=0)
                rhs[rid, :] = A_np @ bnd_lat_sum

        # ---- Source term ----
        if prev_latent_interior is not None:
            src_feats = self._build_source_features_global(
                prev_latent_interior, interior_indices, interior_row_map, t
            )
            src_vals = self.source_network(tf.constant(src_feats, tf.float32), training=False).numpy()
            src_vals = np.nan_to_num(src_vals, nan=0.0).astype(np.float64)
            rhs = rhs + src_vals

        rhs = np.nan_to_num(rhs, nan=0.0, posinf=1e3, neginf=-1e3)

        # ---- Solve in eigenbasis ----
        rhs_tilde = rhs @ Q
        K = laplacian_sparse.tocsr()

        diag_K = np.array(K.sum(axis=1)).ravel()
        diag_K[diag_K == 0] = 1.0
        M_pre = diags(1.0 / diag_K, dtype=np.float64, format='csr')

        tol = 1e-2 if use_fast_solver else 1e-4
        maxiter = 1000 if use_fast_solver else 5000

        lat_tilde = np.zeros_like(rhs_tilde, dtype=np.float64)
        for k in range(latent_dim):
            lam = float(eigvals[k])
            b = rhs_tilde[:, k] / lam
            if np.isnan(b).any() or np.isinf(b).any():
                continue
            try:
                xk, info = cg(K, b, M=M_pre, tol=tol, maxiter=maxiter)
                lat_tilde[:, k] = xk
            except Exception:
                pass

        latent_interior = (lat_tilde @ Q.T).astype(np.float32)

        # ---- Decode ----
        u_int_std = self.decoder(tf.constant(latent_interior, tf.float32), training=False).numpy().reshape(-1)

        interior_y = interior_indices[:, 0]
        interior_x = interior_indices[:, 1]
        u_int_pred = u_int_std * self.U_std[interior_y, interior_x] + self.U_mean[interior_y, interior_x]

        # ---- Assemble ----
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
            "latent_interior": latent_interior,
        }

    # =========================================================================
    # Sequential reconstruction (autoregressive inference)
    # =========================================================================
    def reconstruct_sequence(self, t_indices, use_fast_solver=True, verbose=True):
        """
        Reconstruct a sequence of timesteps autoregressively.

        At t_indices[0]: s = 0 (standard SINN).
        At t_indices[k>0]: source from previous step's latent field.
        """
        import scipy.sparse as sp

        t_indices = np.sort(np.asarray(t_indices, dtype=np.int32))
        b_thick = int(self.b_thick)

        # Precompute shared structures
        obs_mask = np.zeros((self.ny, self.nx), dtype=bool)
        obs_mask[:b_thick, :] = True; obs_mask[-b_thick:, :] = True
        obs_mask[:, :b_thick] = True; obs_mask[:, -b_thick:] = True

        interior_indices = np.argwhere(~obs_mask)
        boundary_indices = np.argwhere(obs_mask)
        num_interior = interior_indices.shape[0]

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
                ny, nx_ = int(y+dy), int(x+dx)
                nr = interior_row_map[ny, nx_]
                if nr >= 0:
                    rows.append(rid); cols.append(int(nr)); vals.append(-1.0)
                else:
                    bi = bnd_map.get((ny, nx_))
                    if bi is not None:
                        bnd_contribs[rid].append(bi)

        laplacian = sp.csr_matrix((vals, (rows, cols)), shape=(num_interior, num_interior), dtype=np.float32)

        A_np = self.get_latent_operator_matrix().numpy().astype(np.float64)
        A_np = 0.5 * (A_np + A_np.T)
        eigvals, Q = np.linalg.eigh(A_np)
        if float(np.min(eigvals)) <= 1e-8:
            eigvals = eigvals + 1e-6

        if verbose:
            print(f"\n[reconstruct_sequence] {len(t_indices)} timesteps, autoregressive")
            print(f"  Interior: {num_interior}, Boundary: {boundary_indices.shape[0]}")
            print(f"  A eigenvalue range: [{eigvals.min():.3e}, {eigvals.max():.3e}]")

        results = []
        prev_latent = None

        for i, t in enumerate(t_indices):
            res = self._reconstruct_single_timestep(
                t_index=t,
                prev_latent_interior=prev_latent,
                interior_indices=interior_indices,
                interior_row_map=interior_row_map,
                boundary_indices=boundary_indices,
                obs_mask=obs_mask,
                laplacian_sparse=laplacian,
                boundary_contributions=bnd_contribs,
                A_np=A_np, eigvals=eigvals, Q=Q,
                use_fast_solver=use_fast_solver,
            )
            prev_latent = res["latent_interior"]
            results.append(res)

            if verbose and ((i + 1) % 10 == 0 or i == 0 or i == len(t_indices) - 1):
                print(f"  t={t:4d} | MAE {res['mae']:.4e} | MSE {res['mse']:.4e}")

        return results

    # =========================================================================
    # Evaluation
    # =========================================================================
    def evaluate_on_test_timesteps(self, verbose=True):
        """
        Reconstruct held-out test timesteps sequentially (autoregressive).

        Since the test set is the last 20% of time, we first reconstruct
        the final training timestep to initialise the latent field,
        then continue autoregressively into the test period.
        """
        if len(self.test_time_indices) == 0:
            print("No test timesteps.")
            return {"mse": float("nan"), "mae": float("nan"), "per_step": []}

        # Start from the last training timestep to seed the autoregressive chain
        last_train_t = int(self.train_time_indices[-1])
        all_t = np.concatenate([[last_train_t], self.test_time_indices])

        all_results = self.reconstruct_sequence(all_t, use_fast_solver=True, verbose=verbose)

        # Drop the seed step (last training timestep)
        test_results = all_results[1:]

        maes = [r["mae"] for r in test_results]
        mses = [r["mse"] for r in test_results]

        summary = {
            "mse": float(np.mean(mses)),
            "mae": float(np.mean(maes)),
            "max_error": float(np.mean([r["max_error"] for r in test_results])),
            "per_step": test_results,
        }
        if verbose:
            print(f"\n  [Test summary] Mean MAE {summary['mae']:.4e} | "
                  f"Mean MSE {summary['mse']:.4e}")
        return summary

    def save_test_results(self, save_path="test_results.pkl", precomputed_results=None):
        if precomputed_results is not None:
            results_list = precomputed_results
        else:
            summary = self.evaluate_on_test_timesteps()
            results_list = summary["per_step"]

        payload = {
            "test_time_indices": np.array(self.test_time_indices, dtype=np.int32),
            "results": results_list,
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved {len(results_list)} test results to {save_path}")

    # =========================================================================
    # Plotting
    # =========================================================================
    def plot_field_reconstruction(self, results, save_path=None, show=True):
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
        if show:
            plt.show()
        return fig

    def plot_test_mae_over_time_v5(self, test_results, save_path=None, show=True):
        """Plot MAE vs timestep for test results to see error accumulation."""
        import matplotlib.pyplot as plt

        t_indices = [r["t_index"] for r in test_results]
        maes = [r["mae"] for r in test_results]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_indices, maes, "b-o", markersize=3, lw=1.5)
        ax.set_xlabel("Timestep"); ax.set_ylabel("MAE")
        ax.set_title("Test MAE over time (autoregressive rollout)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
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
    patch_dim = [10, 10, 10]  # [px, py, pt]
    num_patches = 100
    epochs = 50
    n_past_steps = 5
    train_fraction = 0.8

    # ---- Load data ----
    # Update this path to your data file
    data_path = r"c:\Users\darsh\Documents\fyp\myfyp\time_in_encoders_only\sst\numerical_data_sst.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    Y = data["Y"]
    U = data["U"]
    T = data["T"]

    n_keep = 500
    U = U[-n_keep:]
    T = T[-n_keep:]
    print(f"Using last {n_keep} timesteps. U shape: {np.asarray(U).shape}")

    # ---- Setup ----
    solver = sinn(X, Y, U, T, debug=False)
    solver.split_train_test_timesteps(mode="sequential", train_frac=train_fraction)
    solver.standardise_u(time_indices=solver.train_time_indices)
    solver.split_interior_boundary(b_thick, include_t0, include_tT)
    solver.build_models(num_latentdim, num_units, num_layers, dropout, l2_reg, lr,
                        n_past_steps=n_past_steps)
    print(f"  Encoder input dim: {solver.encoder_input_dim}")
    print(f"  Source input dim: {solver.source_input_dim}")
    print(f"  Fourier features: {solver.n_fourier_frequencies} frequencies -> {solver.coord_embed_dim} coord features")

    # ---- Train ----
    loss_history = solver.train(epochs, patch_dim, num_patches, tf_warmup_epochs=epochs // 3)

    print("\nTraining complete!")
    print(f"Final loss: {loss_history['total'][-1]:.6e}")

    solver.plot_training_history(loss_history, save_path="training_loss_v5.png", show=True)

    # ---- Evaluate on a training timestep (sanity check) ----
    train_mid = int(solver.train_time_indices[len(solver.train_time_indices) // 2])
    train_results = solver.reconstruct_sequence([train_mid], use_fast_solver=False, verbose=True)
    solver.plot_field_reconstruction(train_results[0], save_path=f"field_train_t{train_mid}.png")

    # ---- Evaluate on test timesteps (autoregressive rollout) ----
    test_summary = solver.evaluate_on_test_timesteps()
    print(f"\n[Test] Mean MAE: {test_summary['mae']:.6e}")
    print(f"[Test] Mean MSE: {test_summary['mse']:.6e}")

    # Plot error accumulation over time
    solver.plot_test_mae_over_time_v5(test_summary["per_step"],
                                   save_path="test_mae_over_time_v5.png")

    # Plot example test reconstructions (early, middle, late)
    test_steps = test_summary["per_step"]
    for label, idx in [("early", 0), ("mid", len(test_steps)//2), ("late", -1)]:
        res = test_steps[idx]
        solver.plot_field_reconstruction(res, save_path=f"field_test_{label}_t{res['t_index']}.png")

    # Save results
    solver.save_test_results(save_path="test_results_v5.pkl",
                              precomputed_results=test_summary["per_step"])

    # Print A matrix info
    A_final = solver.get_latent_operator_matrix().numpy()
    print(f"\nA matrix (3x3 corner):\n{A_final[:3, :3]}")
    print(f"Frobenius dist from I: {np.linalg.norm(A_final - np.eye(solver.num_latentdim)):.4f}")
    print(f"Eigenvalues: {np.linalg.eigvalsh(A_final)}")