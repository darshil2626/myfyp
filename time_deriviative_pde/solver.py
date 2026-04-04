import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Dense, Input
from keras.models import Model
import pickle


class sinn:
	"""
	Time-dependent SINN solver.

	Adaptation from v3_time_conditioned_sinn:
	  - Keeps the same encoder/decoder architecture and patch sampling.
	  - Replaces per-time independent elliptic latent solve with time marching on
		each sampled spatio-temporal patch.

	Latent PDE inside each patch (implicit Euler by default):
		dL/dt - div(A grad(L)) = 0
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

		# Time-marching configuration
		self.time_scheme = "implicit_euler"  # options: implicit_euler, explicit_euler
		self.boundary_forcing_time = "auto"  # options: auto, tn, tn1
		self.inference_implicit_backend = "eigen_cg"  # options: eigen_cg, dense_tf

	# -----------------------------
	# Normalisation
	# -----------------------------
	def standardise_u(self, eps: float = 1e-8, time_indices=None):
		if time_indices is None:
			U_ref = self.U
		else:
			time_indices = np.asarray(time_indices, dtype=np.int32)
			U_ref = self.U[time_indices, :, :]

		n_nan = np.isnan(U_ref).sum()
		n_inf = np.isinf(U_ref).sum()
		if n_nan > 0 or n_inf > 0:
			print(f"[standardise_u] Warning: input has {n_nan} NaN and {n_inf} Inf values")
			U_ref = np.nan_to_num(U_ref, nan=0.0, posinf=1e6, neginf=-1e6)
			self.U = np.nan_to_num(self.U, nan=0.0, posinf=1e6, neginf=-1e6)

		self.U_mean = np.mean(U_ref, axis=0)
		self.U_std = np.std(U_ref, axis=0)
		self.U_std = np.maximum(self.U_std, eps)

		self.U = (self.U - self.U_mean) / self.U_std
		self.U = np.nan_to_num(self.U, nan=0.0)

	def unstandardise_u(self, U_pred):
		return U_pred * self.U_std + self.U_mean

	# -----------------------------
	# Masks
	# -----------------------------
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

	# -----------------------------
	# Models
	# -----------------------------
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
		self.encoder_input_dim = 3 + 2 * (k * k) * n_time_slices
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
		self.optimizer = keras.optimizers.Adam(learning_rate=lr)

	# -----------------------------
	# Train / test time split
	# -----------------------------
	def split_train_test_timesteps(self, mode: str = "alternate"):
		all_t = np.arange(self.nt, dtype=np.int32)
		if mode == "alternate":
			self.train_time_indices = all_t[0::2]
			self.test_time_indices = all_t[1::2]
		elif mode == "all":
			self.train_time_indices = all_t
			self.test_time_indices = np.array([], dtype=np.int32)
		else:
			raise ValueError(f"Unknown mode '{mode}'. Use 'alternate' or 'all'.")

		print(f"[split_train_test_timesteps] mode='{mode}'")
		print(f"  Train time steps : {len(self.train_time_indices)}")
		if len(self.test_time_indices):
			print(f"  Test  time steps : {len(self.test_time_indices)}")

	# -----------------------------
	# Patch sampling
	# -----------------------------
	def create_patch_centres_and_indices_and_values_and_boundary_splits(self, patch_dim, num_patches, boundary_fraction=0.5):
		px, py, pt = int(patch_dim[0]), int(patch_dim[1]), int(patch_dim[2])

		x_left, x_right = px // 2, (px - 1) // 2
		y_left, y_right = py // 2, (py - 1) // 2
		t_left, t_right = pt // 2, (pt - 1) // 2

		x_min, x_max = x_left, self.nx - x_right - 1
		y_min, y_max = y_left, self.ny - y_right - 1
		t_min, t_max = t_left, self.nt - t_right - 1

		if x_min > x_max or y_min > y_max or t_min > t_max:
			raise ValueError("Patch dimensions are too large for the grid.")

		valid_train_t = self.train_time_indices[
			(self.train_time_indices >= t_min) & (self.train_time_indices <= t_max)
		]
		if len(valid_train_t) == 0:
			raise ValueError("No valid training time indices remain after patch half-width filtering.")

		num_boundary_patches = int(num_patches * boundary_fraction)

		self.patch_center_idx = np.zeros((num_patches, 3), dtype=np.int32)
		self.patch_interior_idx = np.empty(num_patches, dtype=object)
		self.patch_boundary_idx = np.empty(num_patches, dtype=object)
		self.patch_boundary_global_boundary_idx = np.empty(num_patches, dtype=object)
		self.patch_boundary_global_interior_idx = np.empty(num_patches, dtype=object)

		patch_boundary_mask_local = np.zeros((pt, py, px), dtype=bool)
		patch_boundary_mask_local[0, :, :] = True
		patch_boundary_mask_local[-1, :, :] = True
		patch_boundary_mask_local[:, 0, :] = True
		patch_boundary_mask_local[:, -1, :] = True
		patch_boundary_mask_local[:, :, 0] = True
		patch_boundary_mask_local[:, :, -1] = True
		patch_interior_mask_local = ~patch_boundary_mask_local

		t_offsets = np.arange(-t_left, t_right + 1, dtype=np.int32)
		y_offsets = np.arange(-y_left, y_right + 1, dtype=np.int32)
		x_offsets = np.arange(-x_left, x_right + 1, dtype=np.int32)
		TT, YY, XX = np.meshgrid(t_offsets, y_offsets, x_offsets, indexing="ij")

		offsets = np.stack([TT.ravel(), YY.ravel(), XX.ravel()], axis=1)
		boundary_offsets = offsets[patch_boundary_mask_local.ravel()]
		interior_offsets = offsets[patch_interior_mask_local.ravel()]

		for p in range(num_boundary_patches):
			face = np.random.choice(["t0", "tT", "x_left", "x_right", "y_bottom", "y_top"])

			if face == "t0":
				t_c = int(valid_train_t[0])
				y_c = np.random.randint(y_min, y_max + 1)
				x_c = np.random.randint(x_min, x_max + 1)
			elif face == "tT":
				t_c = int(valid_train_t[-1])
				y_c = np.random.randint(y_min, y_max + 1)
				x_c = np.random.randint(x_min, x_max + 1)
			elif face == "x_left":
				x_c = x_left
				y_c = np.random.randint(y_min, y_max + 1)
				t_c = int(np.random.choice(valid_train_t))
			elif face == "x_right":
				x_c = self.nx - x_right - 1
				y_c = np.random.randint(y_min, y_max + 1)
				t_c = int(np.random.choice(valid_train_t))
			elif face == "y_bottom":
				y_c = y_left
				x_c = np.random.randint(x_min, x_max + 1)
				t_c = int(np.random.choice(valid_train_t))
			else:
				y_c = self.ny - y_right - 1
				x_c = np.random.randint(x_min, x_max + 1)
				t_c = int(np.random.choice(valid_train_t))

			self.patch_center_idx[p, :] = [t_c, y_c, x_c]
			patch_bnd_global = np.array([t_c, y_c, x_c], dtype=np.int32) + boundary_offsets
			patch_int_global = np.array([t_c, y_c, x_c], dtype=np.int32) + interior_offsets

			is_global_boundary = self.mask_boundary[
				patch_bnd_global[:, 0], patch_bnd_global[:, 1], patch_bnd_global[:, 2]
			]

			self.patch_interior_idx[p] = patch_int_global
			self.patch_boundary_idx[p] = patch_bnd_global
			self.patch_boundary_global_boundary_idx[p] = patch_bnd_global[is_global_boundary]
			self.patch_boundary_global_interior_idx[p] = patch_bnd_global[~is_global_boundary]

		for p in range(num_boundary_patches, num_patches):
			t_c = int(np.random.choice(valid_train_t))
			y_c = np.random.randint(y_min, y_max + 1)
			x_c = np.random.randint(x_min, x_max + 1)

			self.patch_center_idx[p, :] = [t_c, y_c, x_c]
			patch_bnd_global = np.array([t_c, y_c, x_c], dtype=np.int32) + boundary_offsets
			patch_int_global = np.array([t_c, y_c, x_c], dtype=np.int32) + interior_offsets

			is_global_boundary = self.mask_boundary[
				patch_bnd_global[:, 0], patch_bnd_global[:, 1], patch_bnd_global[:, 2]
			]

			self.patch_interior_idx[p] = patch_int_global
			self.patch_boundary_idx[p] = patch_bnd_global
			self.patch_boundary_global_boundary_idx[p] = patch_bnd_global[is_global_boundary]
			self.patch_boundary_global_interior_idx[p] = patch_bnd_global[~is_global_boundary]

		num_touching = sum(1 for p in range(num_patches) if len(self.patch_boundary_global_boundary_idx[p]) > 0)
		print(f"  - {num_touching} total patches touch global boundary ({100 * num_touching / num_patches:.1f}%)")

	# -----------------------------
	# Feature stacking
	# -----------------------------
	def _stack_mask_patch_features_from_idx(self, idx_tyx, radius=None):
		idx_tyx = np.asarray(idx_tyx, dtype=np.int32)
		r = self.mask_radius if radius is None else int(radius)
		k = 2 * r + 1
		kk = k * k
		n_time_slices = 1 + self.n_past_steps

		if idx_tyx.size == 0:
			d = 3 + 2 * kk * n_time_slices
			return np.zeros((0, d), dtype=np.float32)

		t = idx_tyx[:, 0]
		y = idx_tyx[:, 1]
		x = idx_tyx[:, 2]
		N = idx_tyx.shape[0]

		t_norm = t.astype(np.float32) / self.nt_norm
		y_norm = y.astype(np.float32) / self.ny_norm
		x_norm = x.astype(np.float32) / self.nx_norm

		u_win = np.full((N, kk * n_time_slices), self.mask_pad_value, dtype=np.float32)
		m_win = np.zeros((N, kk * n_time_slices), dtype=np.float32)

		for i in range(N):
			ti = int(t[i])
			yi = int(y[i])
			xi = int(x[i])

			for p in range(n_time_slices):
				t_slice = ti - p
				slice_offset = p * kk
				if t_slice < 0:
					continue

				ptr = 0
				for dy in range(-r, r + 1):
					yy = yi + dy
					for dx in range(-r, r + 1):
						xx = xi + dx
						if (0 <= yy < self.ny) and (0 <= xx < self.nx):
							u_win[i, slice_offset + ptr] = float(self.U[t_slice, yy, xx])
							m_win[i, slice_offset + ptr] = 1.0
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
		B = self.a_chol_raw
		L = tf.linalg.band_part(B, -1, 0)
		diag = tf.linalg.diag_part(L)
		diag_pos = tf.nn.softplus(diag) + self._chol_eps
		L = tf.linalg.set_diag(L, diag_pos)
		return tf.matmul(L, L, transpose_b=True)

	# -----------------------------
	# Time-dependent PDE helpers
	# -----------------------------
	def _infer_grid_spacing(self):
		"""Infer uniform grid spacing from X and Y; fall back to 1.0 if unavailable."""
		dx = 1.0
		dy = 1.0

		try:
			x_vals = np.unique(np.asarray(self.X).astype(np.float64).ravel())
			y_vals = np.unique(np.asarray(self.Y).astype(np.float64).ravel())

			if x_vals.size > 1:
				diffs_x = np.diff(np.sort(x_vals))
				diffs_x = diffs_x[np.abs(diffs_x) > 0]
				if diffs_x.size > 0:
					dx = float(np.median(np.abs(diffs_x)))

			if y_vals.size > 1:
				diffs_y = np.diff(np.sort(y_vals))
				diffs_y = diffs_y[np.abs(diffs_y) > 0]
				if diffs_y.size > 0:
					dy = float(np.median(np.abs(diffs_y)))
		except Exception:
			dx = 1.0
			dy = 1.0

		dx = dx if dx > 0 else 1.0
		dy = dy if dy > 0 else 1.0
		return dx, dy

	def _build_patch_spatial_operator(self, H: int, W: int):
		spatial_interior_mask = np.zeros((H, W), dtype=bool)
		if H >= 3 and W >= 3:
			spatial_interior_mask[1:-1, 1:-1] = True

		interior_local_yx = np.argwhere(spatial_interior_mask).astype(np.int32)
		n_int = int(interior_local_yx.shape[0])
		if n_int == 0:
			return None, None, None

		interior_row_id = -np.ones((H, W), dtype=np.int32)
		for row_id, (ly, lx) in enumerate(interior_local_yx):
			interior_row_id[ly, lx] = row_id

		dx, dy = self._infer_grid_spacing()
		inv_dx2 = 1.0 / (dx * dx)
		inv_dy2 = 1.0 / (dy * dy)

		K = np.zeros((n_int, n_int), dtype=np.float32)
		boundary_neighbour_local_yx_per_row = [[] for _ in range(n_int)]
		neighbour_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

		for row_id, (ly, lx) in enumerate(interior_local_yx):
			K[row_id, row_id] += 2.0 * (inv_dx2 + inv_dy2)
			for dy, dx in neighbour_steps:
				nly = int(ly + dy)
				nlx = int(lx + dx)
				nbr = int(interior_row_id[nly, nlx])
				w = -inv_dy2 if dy != 0 else -inv_dx2
				if nbr >= 0:
					K[row_id, nbr] += w
				else:
					# Store boundary neighbour and its positive RHS weight (-w).
					boundary_neighbour_local_yx_per_row[row_id].append((nly, nlx, -w))

		return interior_local_yx, tf.constant(K, dtype=tf.float32), boundary_neighbour_local_yx_per_row

	def _assemble_boundary_rhs_for_time(
		self,
		patch_boundary_idx_np: np.ndarray,
		latent_boundary_aligned: tf.Tensor,
		t_n: int,
		y_min: int,
		x_min: int,
		boundary_neighbour_local_yx_per_row,
		A: tf.Tensor,
	):
		bmask = patch_boundary_idx_np[:, 0] == int(t_n)
		bidx = np.nonzero(bmask)[0].astype(np.int32)
		if bidx.size == 0:
			return None

		by = patch_boundary_idx_np[bidx, 1]
		bx = patch_boundary_idx_np[bidx, 2]
		blat = tf.gather(latent_boundary_aligned, bidx, axis=0)

		H = int(np.max(patch_boundary_idx_np[:, 1]) - y_min + 1)
		W = int(np.max(patch_boundary_idx_np[:, 2]) - x_min + 1)
		boundary_lookup = -np.ones((H, W), dtype=np.int32)
		for j, (yy, xx) in enumerate(zip(by, bx)):
			boundary_lookup[int(yy - y_min), int(xx - x_min)] = j

		rhs_rows = []
		for row_id in range(len(boundary_neighbour_local_yx_per_row)):
			rhs_r = tf.zeros((self.num_latentdim,), dtype=tf.float32)
			for (nly, nlx, coeff) in boundary_neighbour_local_yx_per_row[row_id]:
				bj = int(boundary_lookup[nly, nlx])
				if bj >= 0:
					rhs_r = rhs_r + tf.cast(coeff, tf.float32) * tf.linalg.matvec(A, blat[bj, :])
			rhs_rows.append(rhs_r)

		return tf.stack(rhs_rows, axis=0)

	def _implicit_euler_step_dense_tf(self, K_tf: tf.Tensor, A_tf: tf.Tensor, latent_current: tf.Tensor, rhs_boundary: tf.Tensor, dt: float) -> tf.Tensor:
		"""
		Shared dense TensorFlow implicit Euler step.

		(I + dt * (A ⊗ K)) vec(L_next) = vec(L_current + dt * rhs_boundary)
		"""
		n_int = int(latent_current.shape[0])
		r = int(latent_current.shape[1])
		dt_tf = tf.cast(dt, tf.float32)

		K_op = tf.linalg.LinearOperatorFullMatrix(K_tf)
		A_op = tf.linalg.LinearOperatorFullMatrix(A_tf)
		kron_AK = tf.linalg.LinearOperatorKronecker([A_op, K_op]).to_dense()
		I_kron = tf.eye(r * n_int, dtype=tf.float32)

		rhs_mat = latent_current + dt_tf * rhs_boundary
		rhs_vec = tf.reshape(tf.transpose(rhs_mat), (-1, 1))
		system_mat = I_kron + dt_tf * kron_AK
		latent_next_vec = tf.linalg.solve(system_mat, rhs_vec)
		latent_next = tf.transpose(tf.reshape(latent_next_vec, (r, n_int)))
		return latent_next

	def _collect_patch_dt_values(self, patch_boundary_idx_tyx: np.ndarray):
		"""Collect positive dt values between sorted unique patch times for diagnostics."""
		pb = np.asarray(patch_boundary_idx_tyx, dtype=np.int32)
		if pb.size == 0:
			return []
		times = np.unique(pb[:, 0]).astype(np.int32)
		times.sort()
		if times.size < 2:
			return []
		dts = []
		for n in range(len(times) - 1):
			dt = float(self.T[int(times[n + 1])] - self.T[int(times[n])])
			if dt > 0.0:
				dts.append(dt)
		return dts

	# -----------------------------
	# Time-dependent PDE loss (new core)
	# -----------------------------
	def compute_time_dependent_pde_loss(
		self,
		patch_center_idx_tyx: np.ndarray,
		patch_boundary_idx_tyx: np.ndarray,
		latent_boundary_aligned: tf.Tensor,
		alpha_recon: float = 1.0,
		time_scheme: str = None,
		boundary_forcing_time: str = None,
	):
		del patch_center_idx_tyx

		patch_boundary_idx_np = np.asarray(patch_boundary_idx_tyx, dtype=np.int32)
		if patch_boundary_idx_np.shape[0] == 0:
			z = tf.constant(0.0, tf.float32)
			return z, z, z, z, 0

		patch_times = np.unique(patch_boundary_idx_np[:, 0]).astype(np.int32)
		patch_times.sort()
		if patch_times.size < 2:
			z = tf.constant(0.0, tf.float32)
			return z, z, z, z, 0

		y_min = int(patch_boundary_idx_np[:, 1].min())
		y_max = int(patch_boundary_idx_np[:, 1].max())
		x_min = int(patch_boundary_idx_np[:, 2].min())
		x_max = int(patch_boundary_idx_np[:, 2].max())
		H = y_max - y_min + 1
		W = x_max - x_min + 1

		interior_local_yx, K_tf, bnd_neighbours = self._build_patch_spatial_operator(H, W)
		if interior_local_yx is None:
			z = tf.constant(0.0, tf.float32)
			return z, z, z, z, 0

		n_int = int(interior_local_yx.shape[0])
		A = self.get_latent_operator_matrix()
		spd_loss = tf.constant(0.0, dtype=tf.float32)

		scheme = self.time_scheme if time_scheme is None else str(time_scheme)
		forcing_choice = self.boundary_forcing_time if boundary_forcing_time is None else str(boundary_forcing_time)
		if forcing_choice not in ("auto", "tn", "tn1"):
			forcing_choice = "auto"

		r = int(self.num_latentdim)

		interior_global_y = interior_local_yx[:, 0] + y_min
		interior_global_x = interior_local_yx[:, 1] + x_min

		t0 = int(patch_times[0])
		idx_init = np.stack(
			[
				np.full((n_int,), t0, dtype=np.int32),
				interior_global_y.astype(np.int32),
				interior_global_x.astype(np.int32),
			],
			axis=1,
		)
		feats_init = self._stack_mask_patch_features_from_idx(idx_init)
		latent_current = self.interior_encoder(tf.constant(feats_init, dtype=tf.float32), training=True)

		latent_consistency_loss = tf.constant(0.0, tf.float32)
		reconstruction_loss = tf.constant(0.0, tf.float32)
		num_steps = 0

		for n in range(len(patch_times) - 1):
			tn = int(patch_times[n])
			tn1 = int(patch_times[n + 1])

			dt = float(self.T[tn1] - self.T[tn])
			if dt <= 0.0:
				continue

			if forcing_choice == "tn":
				forcing_t = tn
			elif forcing_choice == "tn1":
				forcing_t = tn1
			else:
				forcing_t = tn1 if scheme == "implicit_euler" else tn

			rhs_boundary_n = self._assemble_boundary_rhs_for_time(
				patch_boundary_idx_np=patch_boundary_idx_np,
				latent_boundary_aligned=latent_boundary_aligned,
				t_n=forcing_t,
				y_min=y_min,
				x_min=x_min,
				boundary_neighbour_local_yx_per_row=bnd_neighbours,
				A=A,
			)
			if rhs_boundary_n is None:
				continue

			dt_tf = tf.cast(dt, tf.float32)
			if scheme == "explicit_euler":
				diffusion_term = -tf.matmul(tf.matmul(K_tf, latent_current), A, transpose_b=True) + rhs_boundary_n
				latent_next_pred = latent_current + dt_tf * diffusion_term
			else:
				latent_next_pred = self._implicit_euler_step_dense_tf(
					K_tf=K_tf,
					A_tf=A,
					latent_current=latent_current,
					rhs_boundary=rhs_boundary_n,
					dt=dt,
				)

			if tf.reduce_any(tf.math.is_nan(latent_next_pred)) or tf.reduce_any(tf.math.is_inf(latent_next_pred)):
				continue

			idx_target = np.stack(
				[
					np.full((n_int,), tn1, dtype=np.int32),
					interior_global_y.astype(np.int32),
					interior_global_x.astype(np.int32),
				],
				axis=1,
			)
			feats_target = self._stack_mask_patch_features_from_idx(idx_target)
			latent_next_true = self.interior_encoder(tf.constant(feats_target, dtype=tf.float32), training=True)

			latent_consistency_loss += tf.reduce_mean(tf.square(latent_next_pred - latent_next_true))

			u_pred = self.decoder(latent_next_pred, training=True)
			u_true = self.U[
				np.full((n_int,), tn1, dtype=np.int32),
				interior_global_y.astype(np.int32),
				interior_global_x.astype(np.int32),
			].astype(np.float32).reshape(-1, 1)
			u_true_tf = tf.constant(u_true, dtype=tf.float32)
			reconstruction_loss += tf.reduce_mean(tf.square(u_pred - u_true_tf))

			latent_current = latent_next_pred
			num_steps += 1

		if num_steps == 0:
			z = tf.constant(0.0, tf.float32)
			return z, z, z, z, 0

		latent_loss = latent_consistency_loss / tf.cast(num_steps, tf.float32)
		recon_loss = reconstruction_loss / tf.cast(num_steps, tf.float32)
		total_loss = latent_loss + alpha_recon * recon_loss + spd_loss
		return total_loss, latent_loss, recon_loss, spd_loss, num_steps

	# -----------------------------
	# Training
	# -----------------------------
	def train(self, epochs, patch_dim, num_patches, clip_norm: float = 1.0, time_scheme: str = None, boundary_forcing_time: str = None, log_diagnostics: bool = True):
		loss_history = {"total": [], "latent": [], "recon": [], "spd": []}

		for epoch in range(epochs):
			self.create_patch_centres_and_indices_and_values_and_boundary_splits(
				patch_dim=patch_dim,
				num_patches=num_patches,
			)

			epoch_total_loss = 0.0
			epoch_latent_loss = 0.0
			epoch_recon_loss = 0.0
			epoch_spd_loss = 0.0
			num_valid_patches = 0
			num_zero_step_patches = 0
			num_nonfinite_loss_patches = 0
			num_nan_grad_patches = 0
			epoch_dts = []

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

					total_loss, latent_loss, recon_loss, spd_loss, num_steps = self.compute_time_dependent_pde_loss(
						patch_center_idx_tyx=patch_center_idx_tyx,
						patch_boundary_idx_tyx=patch_boundary_idx_tyx,
						latent_boundary_aligned=latent_on_patch_boundary_aligned,
						alpha_recon=1.0,
						time_scheme=time_scheme,
						boundary_forcing_time=boundary_forcing_time,
					)

				if num_steps == 0:
					num_zero_step_patches += 1
					continue

				epoch_dts.extend(self._collect_patch_dt_values(patch_boundary_idx_tyx))

				loss_val = float(total_loss.numpy())
				if np.isnan(loss_val) or np.isinf(loss_val):
					num_nonfinite_loss_patches += 1
					continue

				grads = tape.gradient(total_loss, self.trainable_vars)
				grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, self.trainable_vars)]

				has_nan_grad = any(tf.reduce_any(tf.math.is_nan(g)).numpy() for g in grads)
				if has_nan_grad:
					num_nan_grad_patches += 1
					continue

				grads, _ = tf.clip_by_global_norm(grads, clip_norm)
				self.optimizer.apply_gradients(zip(grads, self.trainable_vars))
				num_valid_patches += 1

				epoch_total_loss += float(total_loss.numpy())
				epoch_latent_loss += float(latent_loss.numpy())
				epoch_recon_loss += float(recon_loss.numpy())
				epoch_spd_loss += float(spd_loss.numpy())

			if num_valid_patches == 0:
				print(f"epoch {epoch+1:04d} | warning: no valid patches this epoch")
				loss_history["total"].append(float("nan"))
				loss_history["latent"].append(float("nan"))
				loss_history["recon"].append(float("nan"))
				loss_history["spd"].append(float("nan"))
				continue

			inv = 1.0 / float(num_valid_patches)
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
				f"spd {epoch_spd_loss:.6e} | valid_patches {num_valid_patches}/{num_patches}"
			)

			if log_diagnostics:
				eigvals = np.linalg.eigvalsh(self.get_latent_operator_matrix().numpy().astype(np.float64))
				if len(epoch_dts) > 0:
					dt_median = float(np.median(epoch_dts))
					dt_min = float(np.min(epoch_dts))
					dt_max = float(np.max(epoch_dts))
					print(
						f"  diag | dt median/min/max: {dt_median:.3e}/{dt_min:.3e}/{dt_max:.3e} | "
						f"A eig min/max: {eigvals.min():.3e}/{eigvals.max():.3e}"
					)
				else:
					print(
						f"  diag | dt median/min/max: n/a | "
						f"A eig min/max: {eigvals.min():.3e}/{eigvals.max():.3e}"
					)
				print(
					f"  diag | skipped patches -> zero_steps: {num_zero_step_patches}, "
					f"nonfinite_loss: {num_nonfinite_loss_patches}, nan_grad: {num_nan_grad_patches}"
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
	# Train / Test evaluation
	# -----------------------------
	def evaluate_on_test_timesteps(self, t_indices=None, verbose=True, window_steps=None, forcing_mode=None, max_steps=None):
		"""
		Reconstruct fields at held-out (test) time steps and return aggregate metrics.
		"""
		if t_indices is None:
			t_indices = self.test_time_indices
		t_indices = np.asarray(t_indices, dtype=np.int32)

		if len(t_indices) == 0:
			print("[evaluate_on_test_timesteps] No test time steps to evaluate.")
			return {"mse": float("nan"), "mae": float("nan"), "max_error": float("nan"), "per_step": []}

		print(f"\n[evaluate_on_test_timesteps] Evaluating {len(t_indices)} held-out time step(s)...")

		per_step = []
		all_mse, all_mae, all_max = [], [], []

		if max_steps is not None:
			max_steps = int(max_steps)
			t_indices = t_indices[:max_steps]

		for t in t_indices:
			result = self.reconstruct_field_at_timestep(
				t_index=int(t),
				use_fast_solver=True,
				window_steps=window_steps,
				forcing_mode=forcing_mode,
			)
			per_step.append(result)
			all_mse.append(result["mse"])
			all_mae.append(result["mae"])
			all_max.append(result["max_error"])
			if verbose:
				print(f"  t={t:4d} | MSE {result['mse']:.4e} | MAE {result['mae']:.4e} | "
					  f"Max {result['max_error']:.4e}")

		summary = {
			"mse": float(np.mean(all_mse)),
			"mae": float(np.mean(all_mae)),
			"max_error": float(np.mean(all_max)),
			"per_step": per_step,
		}
		print(f"\n  [Test summary] Mean MSE {summary['mse']:.4e} | "
			  f"Mean MAE {summary['mae']:.4e} | Mean Max {summary['max_error']:.4e}")
		return summary

	def save_test_results(self, save_path="test_reconstruction_results.pkl", precomputed_results=None):
		"""
		Save every held-out test time step reconstruction to a pkl file.
		"""
		t_indices = self.test_time_indices
		if len(t_indices) == 0:
			print("No test time steps to save.")
			return

		if precomputed_results is not None:
			results_list = precomputed_results
			print(f"Using {len(results_list)} precomputed test reconstructions...")
		else:
			print(f"Reconstructing {len(t_indices)} test time steps...")
			results_list = []
			for i, t in enumerate(t_indices):
				result = self.reconstruct_field_at_timestep(t_index=int(t))
				results_list.append(result)
				if (i + 1) % 20 == 0 or (i + 1) == len(t_indices):
					print(f"  {i+1}/{len(t_indices)} done")

		payload = {
			"test_time_indices": np.array(t_indices, dtype=np.int32),
			"results": results_list,
		}

		with open(save_path, "wb") as f:
			pickle.dump(payload, f)

		print(f"Saved {len(results_list)} test reconstructions to {save_path}")

	# -----------------------------
	# Reconstruction
	# -----------------------------
	def _stack_mask_patch_features_observed_only(self, idx_tyx: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
		"""
		Build mask-cloud features using only observed values defined by obs_mask.
		"""
		idx_tyx = np.asarray(idx_tyx, dtype=np.int32)
		N = idx_tyx.shape[0]

		r = int(getattr(self, "mask_radius", 1))
		win = 2 * r + 1
		win_sz = win * win
		n_time_slices = 1 + self.n_past_steps

		feats = np.zeros((N, 3 + 2 * win_sz * n_time_slices), dtype=np.float32)

		t_norm = idx_tyx[:, 0].astype(np.float32) / max(self.nt - 1, 1)
		y_norm = idx_tyx[:, 1].astype(np.float32) / max(self.ny - 1, 1)
		x_norm = idx_tyx[:, 2].astype(np.float32) / max(self.nx - 1, 1)
		feats[:, 0] = t_norm
		feats[:, 1] = y_norm
		feats[:, 2] = x_norm

		u_offset = 3
		m_offset = 3 + win_sz * n_time_slices

		for i in range(N):
			ti, yi, xi = idx_tyx[i]

			for p in range(n_time_slices):
				t_slice = int(ti) - p
				slice_off = p * win_sz
				if t_slice < 0:
					continue

				ptr = 0
				for dy in range(-r, r + 1):
					yy = yi + dy
					for dx in range(-r, r + 1):
						xx = xi + dx
						in_bounds = (0 <= yy < self.ny) and (0 <= xx < self.nx)
						if in_bounds and obs_mask[yy, xx]:
							feats[i, u_offset + slice_off + ptr] = float(self.U[t_slice, yy, xx])
							feats[i, m_offset + slice_off + ptr] = 1.0
						ptr += 1

		return feats

	def reconstruct_field_at_timestep(self, t_index, use_fast_solver=False, window_steps=None, forcing_mode=None):
		"""
		Boundary-only time-dependent reconstruction consistent with training PDE.

		Procedure:
		  1) Encode boundary latents over a short time window ending at t.
		  2) Solve an elliptic latent problem at the window start to initialise L^0.
		  3) March latent interior forward with implicit Euler using boundary forcing.
		  4) Decode the final latent state at t.
		"""
		import scipy.sparse as sp
		from scipy.sparse.linalg import cg
		from scipy.sparse import diags

		t = int(t_index)
		print(f"\n[Boundary-only] Reconstructing field at timestep t={t}...")

		b_thick = int(getattr(self, "b_thick", 1))

		obs_mask = np.zeros((self.ny, self.nx), dtype=bool)
		obs_mask[:b_thick, :] = True
		obs_mask[-b_thick:, :] = True
		obs_mask[:, :b_thick] = True
		obs_mask[:, -b_thick:] = True

		spatial_interior_mask = ~obs_mask
		spatial_boundary_mask = obs_mask

		interior_indices = np.argwhere(spatial_interior_mask)
		boundary_indices = np.argwhere(spatial_boundary_mask)

		num_interior = interior_indices.shape[0]
		num_boundary = boundary_indices.shape[0]

		print(f"  Interior points: {num_interior}")
		print(f"  Observed boundary points: {num_boundary}")

		if num_interior == 0:
			raise ValueError(f"No interior points! Check b_thick={b_thick} vs grid ({self.ny},{self.nx}).")

		print("  Step 1: Encoding observed boundary conditions over time window...")
		boundary_y = boundary_indices[:, 0].astype(np.int32)
		boundary_x = boundary_indices[:, 1].astype(np.int32)
		u_boundary_raw = self.U_original[t, boundary_y, boundary_x].astype(np.float32)

		if window_steps is None:
			window_len = max(1, self.n_past_steps)
		else:
			window_len = max(1, int(window_steps))

		t_start = max(0, t - window_len)
		window_times = np.arange(t_start, t + 1, dtype=np.int32)
		boundary_latents_by_time = {}

		for tw in window_times:
			idx_bnd_tw = np.stack([
				np.full(num_boundary, int(tw), dtype=np.int32),
				boundary_y,
				boundary_x,
			], axis=1).astype(np.int32)
			boundary_features_tw = self._stack_mask_patch_features_observed_only(idx_bnd_tw, obs_mask)
			lat_tw = self.boundary_encoder(tf.constant(boundary_features_tw, dtype=tf.float32), training=False).numpy()
			lat_tw = np.nan_to_num(lat_tw, nan=0.0, posinf=1e3, neginf=-1e3)
			boundary_latents_by_time[int(tw)] = lat_tw

		latent_dim = boundary_latents_by_time[int(window_times[0])].shape[1]
		print(f"  Latent dimension: {latent_dim}")

		print("  Step 2: Building Laplacian operator...")
		interior_row_map = -np.ones((self.ny, self.nx), dtype=np.int32)
		for row_id, (y, x) in enumerate(interior_indices):
			interior_row_map[y, x] = row_id

		bnd_map = {(int(y), int(x)): i for i, (y, x) in enumerate(boundary_indices)}

		dx, dy = self._infer_grid_spacing()
		inv_dx2 = 1.0 / (dx * dx)
		inv_dy2 = 1.0 / (dy * dy)

		row_indices, col_indices, values = [], [], []
		neighbour_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
		boundary_contributions = [[] for _ in range(num_interior)]

		for row_id, (y, x) in enumerate(interior_indices):
			row_indices.append(row_id)
			col_indices.append(row_id)
			values.append(2.0 * (inv_dx2 + inv_dy2))
			for dy, dx in neighbour_steps:
				nyy, nxx = int(y + dy), int(x + dx)
				nbr_row = interior_row_map[nyy, nxx]
				w = -inv_dy2 if dy != 0 else -inv_dx2
				if nbr_row >= 0:
					row_indices.append(row_id)
					col_indices.append(int(nbr_row))
					values.append(w)
				else:
					bidx = bnd_map.get((nyy, nxx), None)
					if bidx is not None:
						boundary_contributions[row_id].append((bidx, -w))

		laplacian_sparse = sp.csr_matrix(
			(values, (row_indices, col_indices)),
			shape=(num_interior, num_interior),
			dtype=np.float32,
		)

		print("  Step 3: Getting PDE operator (A matrix)...")
		A_matrix_np = self.get_latent_operator_matrix().numpy().astype(np.float64)
		A_matrix_np = 0.5 * (A_matrix_np + A_matrix_np.T)
		eigvals, Q = np.linalg.eigh(A_matrix_np)

		min_eig = float(np.min(eigvals))
		max_eig = float(np.max(eigvals))
		shift = 0.0
		if min_eig <= 1e-8:
			shift = 1e-6
			eigvals = eigvals + shift
			print(f"    [Warning] min eigenvalue too small ({min_eig:.3e}), shifted by {shift}")

		cond_number = max_eig / (min_eig + shift) if (min_eig + shift) > 0 else float("inf")
		if cond_number > 1e6:
			print(f"    [Warning] A is ill-conditioned (cond ~= {cond_number:.2e})")
		print(f"    A eigenvalue range: [{min_eig + shift:.3e}, {max_eig:.3e}]")

		print("  Step 4: Solving initial latent state at window start (elliptic)...")
		rhs0 = np.zeros((num_interior, latent_dim), dtype=np.float64)
		blat0 = boundary_latents_by_time[int(t_start)].astype(np.float64)
		for row_id in range(num_interior):
			for bidx, coeff in boundary_contributions[row_id]:
				rhs0[row_id, :] += coeff * (A_matrix_np @ blat0[bidx, :])

		rhs0 = np.nan_to_num(rhs0, nan=0.0, posinf=1e3, neginf=-1e3)
		rhs0_tilde = rhs0 @ Q

		print("  Step 5: Time-marching latents with implicit Euler...")
		K = laplacian_sparse.tocsr()
		latent_tilde = np.zeros_like(rhs0_tilde, dtype=np.float64)

		diag_K = np.array(K.sum(axis=1)).ravel()
		diag_K[diag_K == 0] = 1.0
		precond_inv = 1.0 / diag_K
		M = diags(precond_inv, dtype=np.float64, format="csr")

		if use_fast_solver:
			tol = 1e-2
			maxiter = 1000
		else:
			tol = 1e-4
			maxiter = 5000

		for k in range(latent_dim):
			lam = float(eigvals[k])
			b = rhs0_tilde[:, k] / lam

			if np.isnan(b).any() or np.isinf(b).any():
				print(f"    [cg] dim {k}: RHS contains NaN/Inf, using zero solution")
				latent_tilde[:, k] = np.zeros_like(b)
				continue

			try:
				xk, info = cg(K, b, M=M, tol=tol, maxiter=maxiter)
				if info == 0:
					latent_tilde[:, k] = xk
				else:
					if use_fast_solver:
						latent_tilde[:, k] = xk
					else:
						print(f"    [cg] dim {k}: warning info={info} (partial convergence accepted)")
						latent_tilde[:, k] = xk
			except Exception as e:
				print(f"    [cg] dim {k}: solve failed ({e}), using zero solution")
				latent_tilde[:, k] = np.zeros_like(b)

		latent_current = (latent_tilde @ Q.T).astype(np.float64)

		forcing_choice = self.boundary_forcing_time if forcing_mode is None else str(forcing_mode)
		if forcing_choice not in ("auto", "tn", "tn1"):
			forcing_choice = "auto"
		backend = str(getattr(self, "inference_implicit_backend", "eigen_cg"))
		if backend not in ("eigen_cg", "dense_tf"):
			backend = "eigen_cg"

		K_tf_dense = tf.constant(K.toarray(), dtype=tf.float32) if backend == "dense_tf" else None
		A_tf_dense = tf.constant(A_matrix_np.astype(np.float32), dtype=tf.float32) if backend == "dense_tf" else None
		for n in range(len(window_times) - 1):
			tn = int(window_times[n])
			tn1 = int(window_times[n + 1])
			dt = float(self.T[tn1] - self.T[tn])
			if dt <= 0.0:
				continue

			if forcing_choice == "tn":
				forcing_t = tn
			elif forcing_choice == "tn1":
				forcing_t = tn1
			else:
				forcing_t = tn1

			blat = boundary_latents_by_time[forcing_t].astype(np.float64)
			rhs = np.zeros((num_interior, latent_dim), dtype=np.float64)
			for row_id in range(num_interior):
				for bidx, coeff in boundary_contributions[row_id]:
					rhs[row_id, :] += coeff * (A_matrix_np @ blat[bidx, :])

			rhs = np.nan_to_num(rhs, nan=0.0, posinf=1e3, neginf=-1e3)
			if backend == "dense_tf":
				latent_next_tf = self._implicit_euler_step_dense_tf(
					K_tf=K_tf_dense,
					A_tf=A_tf_dense,
					latent_current=tf.constant(latent_current.astype(np.float32), dtype=tf.float32),
					rhs_boundary=tf.constant(rhs.astype(np.float32), dtype=tf.float32),
					dt=dt,
				)
				latent_current = latent_next_tf.numpy().astype(np.float64)
			else:
				latent_tilde_curr = latent_current @ Q
				rhs_tilde = rhs @ Q
				latent_tilde_next = np.zeros_like(latent_tilde_curr)

				for k in range(latent_dim):
					lam = float(eigvals[k])
					Sk = (sp.eye(num_interior, format="csr", dtype=np.float64) + (dt * lam) * K)
					bk = latent_tilde_curr[:, k] + dt * rhs_tilde[:, k]

					diag_Sk = Sk.diagonal().copy()
					diag_Sk[diag_Sk == 0] = 1.0
					Mk = diags(1.0 / diag_Sk, dtype=np.float64, format="csr")

					xk, info = cg(Sk, bk, M=Mk, tol=tol, maxiter=maxiter)
					if info != 0:
						xk = np.nan_to_num(xk, nan=0.0, posinf=0.0, neginf=0.0)
					latent_tilde_next[:, k] = xk

				latent_current = latent_tilde_next @ Q.T

		latent_interior = latent_current.astype(np.float32)

		print("  Step 6: Decoding latents to physical field...")
		u_interior_pred_standardised = self.decoder(tf.constant(latent_interior, dtype=tf.float32), training=False).numpy().reshape(-1)

		print("  Step 7: Unstandardising predictions...")
		interior_y = interior_indices[:, 0]
		interior_x = interior_indices[:, 1]
		u_interior_pred = (
			u_interior_pred_standardised * self.U_std[interior_y, interior_x]
			+ self.U_mean[interior_y, interior_x]
		)

		print("  Step 8: Assembling full field...")
		u_pred_full = np.zeros((self.ny, self.nx), dtype=np.float32)
		u_true_full = self.U_original[t, :, :].astype(np.float32)

		u_pred_full[boundary_y, boundary_x] = u_boundary_raw
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
		abs_error = np.abs(results["u_error"])
		abs_error = np.nan_to_num(abs_error, nan=0.0, posinf=0.0, neginf=0.0)
		boundary_mask = results["boundary_mask"]
		t_index = results["t_index"]

		fig, axes = plt.subplots(1, 3, figsize=(18, 5))
		vmin_true = np.percentile(u_true, 2)
		vmax_true = np.percentile(u_true, 98)
		vmin_pred = np.percentile(u_pred, 2)
		vmax_pred = np.percentile(u_pred, 98)
		vmin = min(vmin_true, vmin_pred)
		vmax = max(vmax_true, vmax_pred)

		boundary_coords = np.argwhere(boundary_mask)

		ax = axes[0]
		im1 = ax.imshow(u_true, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
		ax.set_title(f"True Field (t={t_index})", fontsize=13, fontweight="bold")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.scatter(boundary_coords[:, 1], boundary_coords[:, 0], c="red", s=1, alpha=0.3)
		plt.colorbar(im1, ax=ax).set_label("u")

		ax = axes[1]
		im2 = ax.imshow(u_pred, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax, aspect="auto")
		ax.set_title("Predicted Field (Reconstructed)", fontsize=13, fontweight="bold")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.scatter(boundary_coords[:, 1], boundary_coords[:, 0], c="red", s=1, alpha=0.3)
		plt.colorbar(im2, ax=ax).set_label("u")

		ax = axes[2]
		err_vmin = np.percentile(abs_error, 5)
		err_vmax = np.percentile(abs_error, 95)
		im3 = ax.imshow(abs_error, cmap="hot", origin="lower", aspect="auto", vmin=err_vmin, vmax=err_vmax)
		ax.set_title("Absolute Error", fontsize=13, fontweight="bold")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.scatter(boundary_coords[:, 1], boundary_coords[:, 0], c="cyan", s=1, alpha=0.5)
		plt.colorbar(im3, ax=ax).set_label("error")

		ax.text(
			0.02,
			0.98,
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

	def run_inference_ablation(self, window_steps_list, forcing_mode_list, max_test_steps=20, verbose=False, save_path="ablation_results.pkl"):
		"""
		Run a compact ablation over inference window length and forcing mode.
		"""
		window_steps_list = [max(1, int(w)) for w in window_steps_list]
		forcing_mode_list = [str(m) for m in forcing_mode_list]

		results = []
		print("\n[run_inference_ablation] Starting ablation...")
		for w in window_steps_list:
			for mode in forcing_mode_list:
				print(f"  - Evaluating window_steps={w}, forcing_mode='{mode}'")
				summary = self.evaluate_on_test_timesteps(
					verbose=verbose,
					window_steps=w,
					forcing_mode=mode,
					max_steps=max_test_steps,
				)
				entry = {
					"window_steps": w,
					"forcing_mode": mode,
					"mse": summary["mse"],
					"mae": summary["mae"],
					"max_error": summary["max_error"],
				}
				results.append(entry)

		if save_path:
			with open(save_path, "wb") as f:
				pickle.dump(results, f)
			print(f"[run_inference_ablation] Saved results to {save_path}")

		print("[run_inference_ablation] Completed.")
		for r in results:
			print(
				f"  window={r['window_steps']:2d}, mode={r['forcing_mode']:>4s} | "
				f"MSE {r['mse']:.4e} | MAE {r['mae']:.4e} | Max {r['max_error']:.4e}"
			)
		return results


if __name__ == "__main__":
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

	with open(r"c:\Users\darsh\Documents\fyp\myfyp\time_deriviative_pde\training_data_speed.pkl", "rb") as f:
		data = pickle.load(f)

	X = data["X"]
	Y = data["Y"]
	U = data["U"]
	T = data["T"]

	n_keep = 500
	U = U[-n_keep:]
	T = T[-n_keep:]
	print(f"Using last {n_keep} timesteps. New U shape: {len(U)} timesteps")

	solver = sinn(X, Y, U, T, debug=False)
	solver.split_train_test_timesteps(mode="alternate")
	solver.standardise_u(time_indices=solver.train_time_indices)
	solver.split_interior_boundary(b_thick, include_t0, include_tT)
	solver.build_models(num_latentdim, num_units, num_layers, dropout, l2_reg, lr, n_past_steps=n_past_steps)

	loss_history = solver.train(epochs, patch_dim, num_patches)

	print("\nTraining complete!")
	print(f"Final total loss: {loss_history['total'][-1]:.6e}")
	print(f"Final latent loss: {loss_history['latent'][-1]:.6e}")
	print(f"Final recon loss: {loss_history['recon'][-1]:.6e}")
	print(f"Final spd loss: {loss_history['spd'][-1]:.6e}")

	solver.plot_training_history(loss_history, save_path="training_loss_time_dependent.png", show=True)

	# ---- Evaluate on a seen (train) time step for reference ----
	train_example_t = int(solver.train_time_indices[len(solver.train_time_indices) // 2])
	results_train = solver.reconstruct_field_at_timestep(t_index=train_example_t)
	solver.plot_field_reconstruction(results_train, save_path=f"field_train_t{train_example_t}.png", show=True)
	print(f"[Train step t={train_example_t}] Reconstruction MAE: {results_train['mae']:.6e}")

	# ---- Evaluate on all held-out (odd) time steps ----
	test_summary = solver.evaluate_on_test_timesteps()
	print(f"\n[Test (unseen) steps] Mean MAE: {test_summary['mae']:.6e}")

	# ---- Save all test reconstructions to pkl for animation ----
	solver.save_test_results(save_path="test_reconstruction_results.pkl", precomputed_results=test_summary["per_step"])

	if len(solver.test_time_indices) > 0:
		test_example_t = int(solver.test_time_indices[len(solver.test_time_indices) // 2])
		results_test = solver.reconstruct_field_at_timestep(t_index=test_example_t)
		solver.plot_field_reconstruction(results_test, save_path=f"field_test_t{test_example_t}.png", show=True)
		print(f"[Test step t={test_example_t}] Reconstruction MAE: {results_test['mae']:.6e}")

	A_final = solver.get_latent_operator_matrix().numpy()
	print("A matrix (first 3x3):")
	print(A_final[:3, :3])
	print(f"\nFrobenius distance from identity: {np.linalg.norm(A_final - np.eye(solver.num_latentdim)):.4f}")
	print(f"Eigenvalues: {np.linalg.eigvalsh(A_final)}")
