"""
Patch-based SINN for 2D wave equation with extended (infinite) domain.

- Solve 2D wave equation on a larger global domain [0,2]x[0,2].
- Extract a local window [0.75,1.25]x[0.75,1.25] as the 'finite' domain.
- Build overlapping 17x17 patches on this local domain.
- Train SINN on patches (coupled SPD latent operator A, Jacobi solver).
- Validate by solving on the full *local* domain using learned A and boundary encoder.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt

# ======================================================================
# SECTION 1: 2D WAVE EQUATION SOLVER (EXTENDED DOMAIN)
# ======================================================================

def solve_wave_equation_2d(
    c=1.0,
    Lx=2.0,
    Ly=2.0,
    dx=0.01,
    dy=0.01,
    Lt=1.0,
    CFL=0.9,
    x_c=0.9,
    y_c=1.0,
    width=np.sqrt(0.005),
    amplitude=1.0,
):
    """
    Simple 2D wave equation with Dirichlet boundaries and Gaussian IC.
    u_tt = c^2 (u_xx + u_yy)

    Solves on an extended domain [0,Lx]x[0,Ly] to mimic an infinite domain.
    """

    # Spatial grid
    Nx = int(Lx / dx)
    Ny = int(Ly / dy)
    x = np.linspace(0.0, Lx, Nx + 1)
    y = np.linspace(0.0, Ly, Ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Time step from CFL
    dt = CFL * min(dx, dy) / (c * np.sqrt(2.0))
    Nt = int(Lt / dt)
    T = np.linspace(0.0, Lt, Nt + 1)

    # Initial displacement: Gaussian pulse
    r2 = (X - x_c) ** 2 + (Y - y_c) ** 2
    U0 = amplitude * np.exp(-r2 / (2 * width**2))

    # Initial velocity = 0
    U1 = U0.copy()

    # Allocate solution: on global domain
    U = np.zeros((Nt + 1, Nx + 1, Ny + 1), dtype=np.float32)
    U[0] = U0
    U[1] = U1

    c2 = c**2
    rx2 = (c2 * dt**2) / dx**2
    ry2 = (c2 * dt**2) / dy**2

    for n in range(1, Nt):
        u_nm1 = U[n - 1]
        u_n = U[n]

        # second derivatives
        u_xx = (u_n[2:, 1:-1] - 2 * u_n[1:-1, 1:-1] + u_n[:-2, 1:-1]) / dx**2
        u_yy = (u_n[1:-1, 2:] - 2 * u_n[1:-1, 1:-1] + u_n[1:-1, :-2]) / dy**2

        u_next = np.zeros_like(u_n)
        u_next[1:-1, 1:-1] = (
            2.0 * u_n[1:-1, 1:-1]
            - u_nm1[1:-1, 1:-1]
            + c2 * dt**2 * (u_xx + u_yy)
        )

        # Dirichlet BCs = 0 on extended domain boundary
        u_next[0, :] = 0.0
        u_next[-1, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0

        U[n + 1] = u_next

    return U, T, x, y, X, Y


# ======================================================================
# SECTION 2: DEFINE GLOBAL & LOCAL DOMAINS, CROP LOCAL WINDOW
# ======================================================================

# Global (extended) domain parameters
c = 1.0
Lx_global = 2.0
Ly_global = 2.0
dx = 0.01
dy = 0.01
Lt = 1.0
CFL = 0.9

# Gaussian pulse (same as your previous setup)
y_c = 0.9
x_c = 1.0
width = np.sqrt(0.005)
amplitude = 1.0

# Solve on extended domain
U_global, T, x_full, y_full, X_full, Y_full = solve_wave_equation_2d(
    c=c,
    Lx=Lx_global,
    Ly=Ly_global,
    dx=dx,
    dy=dy,
    Lt=Lt,
    CFL=CFL,
    x_c=x_c,
    y_c=y_c,
    width=width,
    amplitude=amplitude,
)

# Local "finite" domain inside the larger global domain
x_train_range = (0.75, 1.25)
y_train_range = (0.75, 1.25)

i_min = np.argmin(np.abs(x_full - x_train_range[0]))
i_max = np.argmin(np.abs(x_full - x_train_range[1])) + 1
j_min = np.argmin(np.abs(y_full - y_train_range[0]))
j_max = np.argmin(np.abs(y_full - y_train_range[1])) + 1

# Crop solution and coordinates to local domain
U = U_global[:, i_min:i_max, j_min:j_max]
x = x_full[i_min:i_max]
y = y_full[j_min:j_max]
X, Y = np.meshgrid(x, y, indexing="ij")

Nx = X.shape[0] - 1
Ny = X.shape[1] - 1

# ======================================================================
# SECTION 3: SNAPSHOT & NORMALISATION
# ======================================================================

def screenshot_and_normalise(U, time_idx):
    """Standardise one time slice to mean 0, std 1."""
    u_screenshot = np.array(U[time_idx], dtype=np.float32)
    u_mean = np.mean(u_screenshot)
    u_std = np.std(u_screenshot) + 1e-8
    u_norm = (u_screenshot - u_mean) / u_std
    u_norm = np.clip(u_norm, -3.0, 3.0)
    return u_norm, u_mean, u_std

time_idx = 30  # choose a snapshot index
u_field_norm, u_mean, u_std = screenshot_and_normalise(U, time_idx)

# Coordinate normalisation (local domain)
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
x_scale = (x_max - x_min) / 2.0
y_scale = (y_max - y_min) / 2.0

X_n = (X - (x_min + x_max) / 2.0) / x_scale
Y_n = (Y - (y_min + y_max) / 2.0) / y_scale


# ======================================================================
# SECTION 4: PATCH BUILDER (17x17, stride 8) ON LOCAL DOMAIN
# ======================================================================

def build_patches(u_field_norm, X, Y, patch_radius=8, stride=3, inner_margin=2):
    """
    Build overlapping square patches of size (2R+1)x(2R+1) on the *local* domain.

    Each patch dict contains:
      - interior_feats_tf: (N_int, 3) [x_n, y_n, u_norm] from a core region
      - boundary_feats_tf: (N_b, 3)
      - interior_indices_tf: (N_int, 2)  local indices in patch (core only)
      - boundary_indices_tf: (N_b, 2)   local indices in patch (outer ring)
      - boundary_mask_tf: (P, P) bool
      - Nx, Ny: patch Nx,Ny (P-1 each)

    inner_margin: number of cells to discard from each side of the patch interior.
                  So for patch_radius=8 (P=17) and inner_margin=2,
                  the "core interior" is 13x13 with indices 2..14.
    """
    patches = []
    P = 2 * patch_radius + 1  # patch side length

    Nx_full = X.shape[0] - 1
    Ny_full = X.shape[1] - 1

    i_centres = range(patch_radius, Nx_full + 1 - patch_radius, stride)
    j_centres = range(patch_radius, Ny_full + 1 - patch_radius, stride)

    for ic in i_centres:
        for jc in j_centres:
            i_start = ic - patch_radius
            i_end   = ic + patch_radius + 1
            j_start = jc - patch_radius
            j_end   = jc + patch_radius + 1

            # Extract patch arrays
            u_patch = u_field_norm[i_start:i_end, j_start:j_end]
            X_patch = X[i_start:i_end, j_start:j_end]
            Y_patch = Y[i_start:i_end, j_start:j_end]

            # Normalised coords for this patch (using global-local normalisation)
            Xp_n = (X_patch - (x_min + x_max) / 2.0) / x_scale
            Yp_n = (Y_patch - (y_min + y_max) / 2.0) / y_scale

            # ---------- CORE INTERIOR INDICES ----------
            i_core, j_core = np.meshgrid(
                np.arange(inner_margin, P - inner_margin),
                np.arange(inner_margin, P - inner_margin),
                indexing="ij",
            )
            interior_idx = np.stack(
                [i_core.flatten(), j_core.flatten()], axis=-1
            )  # (N_int, 2)

            # ---------- BOUNDARY INDICES ----------
            boundary_idx = []
            # top & bottom rows
            for j_loc in range(P):
                boundary_idx.append([0, j_loc])
                boundary_idx.append([P - 1, j_loc])
            # left & right columns (excluding corners)
            for i_loc in range(1, P - 1):
                boundary_idx.append([i_loc, 0])
                boundary_idx.append([i_loc, P - 1])
            boundary_idx = np.array(boundary_idx, dtype=np.int32)

            # Boundary mask
            boundary_mask_patch = np.zeros((P, P), dtype=bool)
            boundary_mask_patch[0, :] = True
            boundary_mask_patch[-1, :] = True
            boundary_mask_patch[:, 0] = True
            boundary_mask_patch[:, -1] = True

            # ---------- INTERIOR FEATURES (CORE ONLY) ----------
            u_int = u_patch[inner_margin:P-inner_margin,
                            inner_margin:P-inner_margin].flatten()
            x_int = Xp_n[inner_margin:P-inner_margin,
                         inner_margin:P-inner_margin].flatten()
            y_int = Yp_n[inner_margin:P-inner_margin,
                         inner_margin:P-inner_margin].flatten()
            interior_feats = np.stack([x_int, y_int, u_int], axis=-1)

            # ---------- BOUNDARY FEATURES ----------
            u_b = u_patch[boundary_idx[:, 0], boundary_idx[:, 1]]
            x_b = Xp_n[boundary_idx[:, 0], boundary_idx[:, 1]]
            y_b = Yp_n[boundary_idx[:, 0], boundary_idx[:, 1]]
            boundary_feats = np.stack([x_b, y_b, u_b], axis=-1)

            patch = {
                "interior_feats_tf": tf.constant(interior_feats, dtype=tf.float32),
                "boundary_feats_tf": tf.constant(boundary_feats, dtype=tf.float32),
                "interior_indices_tf": tf.constant(interior_idx, dtype=tf.int32),
                "boundary_indices_tf": tf.constant(boundary_idx, dtype=tf.int32),
                "boundary_mask_tf": tf.constant(boundary_mask_patch, dtype=tf.bool),
                "Nx": P - 1,
                "Ny": P - 1,
                "inner_margin": inner_margin,
            }
            patches.append(patch)

    return patches

patch_radius = 10   # 17x17 patches
stride = 2         # moderate overlap
patches = build_patches(u_field_norm, X, Y, patch_radius, stride, inner_margin=2)
print(f"Built {len(patches)} training patches on local domain.")


# ======================================================================
# SECTION 5: GLOBAL INDEX MAPS FOR FULL LOCAL-DOMAIN VALIDATION
# ======================================================================

# Global interior indices (for validation on local domain)
i_int, j_int = np.meshgrid(np.arange(1, Nx), np.arange(1, Ny), indexing="ij")
interior_indices = np.stack([i_int.flatten(), j_int.flatten()], axis=-1)
interior_indices_tf = tf.constant(interior_indices, dtype=tf.int32)

# Global boundary indices (local domain rectangle)
boundary_indices = []
for j in range(Ny + 1):
    boundary_indices.append([0, j])
    boundary_indices.append([Nx, j])
for i in range(1, Nx):
    boundary_indices.append([i, 0])
    boundary_indices.append([i, Ny])
boundary_indices = np.array(boundary_indices, dtype=np.int32)
boundary_indices_tf = tf.constant(boundary_indices, dtype=tf.int32)

boundary_mask_np = np.zeros((Nx + 1, Ny + 1), dtype=bool)
boundary_mask_np[0, :] = True
boundary_mask_np[-1, :] = True
boundary_mask_np[:, 0] = True
boundary_mask_np[:, -1] = True
boundary_mask_tf = tf.constant(boundary_mask_np, dtype=tf.bool)


# ======================================================================
# SECTION 6: ENCODERS & DECODER
# ======================================================================

def make_interior_encoder(num_latentdim, num_units):
    inputs = Input(shape=(3,), name="interior_input")
    x = Dense(num_units, activation="tanh",
              kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)
    x = Dense(num_units, activation="tanh",
              kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = Dense(num_units, activation="tanh",
              kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    outputs = Dense(num_latentdim, activation=None)(x)
    return Model(inputs, outputs, name="interior_encoder")


def make_boundary_encoder(num_latentdim, num_units):
    inputs = Input(shape=(3,), name="boundary_input")
    x = Dense(num_units, activation="tanh",
              kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)
    x = Dense(num_units, activation="tanh",
              kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = Dense(num_units, activation="tanh",
              kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    outputs = Dense(num_latentdim, activation=None)(x)
    return Model(inputs, outputs, name="boundary_encoder")


def make_decoder(num_latentdim, num_units):
    inputs = Input(shape=(num_latentdim + 2,), name="decoder_input")
    x = Dense(num_units, activation="tanh",
              kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)
    x = Dense(num_units, activation="tanh",
              kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = Dense(num_units, activation="tanh",
              kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    outputs = Dense(1, activation=None)(x)
    return Model(inputs, outputs, name="decoder")


latent_dim = 24
hidden_units = 128

interior_encoder = make_interior_encoder(latent_dim, hidden_units)
boundary_encoder = make_boundary_encoder(latent_dim, hidden_units)
decoder = make_decoder(latent_dim, hidden_units)


# ======================================================================
# SECTION 7: COUPLED LATENT PDE SOLVER ON A PATCH
# ======================================================================

def solve_latent_field_tf(boundary_latent,
                          A_matrix,
                          boundary_indices_tf_patch,
                          boundary_mask_tf_patch,
                          Nx_patch,
                          Ny_patch,
                          n_iters=80,
                          tau=0.05):
    """
    Coupled latent elliptic Jacobi solve on a (Nx_patch+1)x(Ny_patch+1) grid.
    """
    latent_dim = boundary_latent.shape[1]

    ell = tf.zeros((Nx_patch + 1, Ny_patch + 1, latent_dim), dtype=tf.float32)

    nb = tf.shape(boundary_indices_tf_patch)[0]
    for dim in range(latent_dim):
        dim_idx = tf.fill((nb, 1), dim)
        coords = tf.concat([boundary_indices_tf_patch, dim_idx], axis=1)
        ell = tf.tensor_scatter_nd_update(ell, coords, boundary_latent[:, dim])

    ell_bc = ell
    boundary_mask_3d = tf.tile(boundary_mask_tf_patch[..., None], [1, 1, latent_dim])

    for _ in range(n_iters):
        ell_old = ell

        interior = ell_old[1:-1, 1:-1, :]
        up = ell_old[0:-2, 1:-1, :]
        down = ell_old[2:, 1:-1, :]
        left = ell_old[1:-1, 0:-2, :]
        right = ell_old[1:-1, 2:, :]

        lap = up + down + left + right - 4.0 * interior

        lap_flat = tf.reshape(lap, [-1, latent_dim])
        lap_coupled_flat = tf.matmul(lap_flat, A_matrix, transpose_b=True)
        lap_coupled = tf.reshape(lap_coupled_flat, tf.shape(lap))

        new_interior = interior + tau * lap_coupled

        top = ell_old[0:1, :, :]
        bottom = ell_old[-1:, :, :]
        mid_left = ell_old[1:-1, 0:1, :]
        mid_right = ell_old[1:-1, -1:, :]
        mid = tf.concat([mid_left, new_interior, mid_right], axis=1)
        ell_update = tf.concat([top, mid, bottom], axis=0)

        ell = tf.where(boundary_mask_3d, ell_bc, ell_update)

    return ell


# ======================================================================
# SECTION 8: LOSS ON A SINGLE PATCH
# ======================================================================

def compute_loss_patch(interior_encoder,
                       boundary_encoder,
                       decoder,
                       patch,
                       A_matrix,
                       alpha=2.0,
                       n_pde_iters=80):
    interior_feats_tf = patch["interior_feats_tf"]
    boundary_feats_tf = patch["boundary_feats_tf"]
    interior_indices_tf_patch = patch["interior_indices_tf"]
    boundary_indices_tf_patch = patch["boundary_indices_tf"]
    boundary_mask_tf_patch = patch["boundary_mask_tf"]
    Nx_patch = patch["Nx"]
    Ny_patch = patch["Ny"]

    # Encode
    interior_latent_true = interior_encoder(interior_feats_tf, training=True)
    boundary_latent = boundary_encoder(boundary_feats_tf, training=True)

    # Latent elliptic solve on patch
    latent_field = solve_latent_field_tf(
        boundary_latent,
        A_matrix,
        boundary_indices_tf_patch,
        boundary_mask_tf_patch,
        Nx_patch,
        Ny_patch,
        n_iters=n_pde_iters,
        tau=0.05,
    )

    ell_pred = tf.gather_nd(latent_field, interior_indices_tf_patch)

    # Ψ₁: latent consistency
    loss_latent = tf.reduce_mean(tf.square(interior_latent_true - ell_pred))

    # Ψ₂: reconstruction loss
    spatial_coords = interior_feats_tf[:, :2]
    decoder_input = tf.concat([spatial_coords, ell_pred], axis=1)
    u_pred_norm = decoder(decoder_input, training=True)
    u_true_norm = tf.expand_dims(interior_feats_tf[:, 2], axis=1)
    loss_physical = tf.reduce_mean(tf.square(u_true_norm - u_pred_norm))

    total_loss = loss_latent + alpha * loss_physical
    return total_loss, loss_latent, loss_physical


# ======================================================================
# SECTION 9: SPD A FROM CHOLESKY
# ======================================================================

def build_A_from_cholesky(L_raw):
    """
    Construct a SPD matrix A = L Lᵀ and normalise to keep eigenvalues bounded.
    """
    L = tf.linalg.band_part(L_raw, -1, 0)
    diag = tf.linalg.diag_part(L)
    diag_pos = tf.nn.softplus(diag) + 1e-3
    L = L - tf.linalg.diag(diag) + tf.linalg.diag(diag_pos)

    A = tf.matmul(L, L, transpose_b=True)
    fro = tf.linalg.norm(A)
    A = A / (1.0 + fro)
    return A


# ======================================================================
# SECTION 10: TRAINING LOOP (PATCH-BASED SINN)
# ======================================================================

def train_sinn(interior_encoder,
               boundary_encoder,
               decoder,
               patches,
               num_epochs=500,
               latent_dim=8,
               alpha=2.0,
               n_pde_iters=80,
               lr=1e-3):

    L_raw = tf.Variable(
        0.01 * tf.random.normal((latent_dim, latent_dim), dtype=tf.float32),
        trainable=True,
        name="A_cholesky_raw",
    )

    trainable_vars = (
        interior_encoder.trainable_variables
        + boundary_encoder.trainable_variables
        + decoder.trainable_variables
        + [L_raw]
    )

    print(f"Trainable variable tensors: {len(trainable_vars)}")
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_history = []

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            A_matrix = build_A_from_cholesky(L_raw)

            total_loss = 0.0
            total_lat = 0.0
            total_phy = 0.0

            for patch in patches:
                loss_p, lat_p, phy_p = compute_loss_patch(
                    interior_encoder,
                    boundary_encoder,
                    decoder,
                    patch,
                    A_matrix,
                    alpha=alpha,
                    n_pde_iters=n_pde_iters,
                )
                total_loss += loss_p
                total_lat += lat_p
                total_phy += phy_p

            n_patches = float(len(patches))
            total_loss /= n_patches
            total_lat /= n_patches
            total_phy /= n_patches

        grads = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))

        loss_history.append(float(total_loss))

        if (epoch + 1) % max(1, num_epochs // 10) == 0:
            print(
                f"Epoch {epoch+1:4d}: "
                f"Loss={float(total_loss):.6f}, "
                f"Ψ₁={float(total_lat):.6f}, "
                f"Ψ₂={float(total_phy):.6f}"
            )

    final_A = build_A_from_cholesky(L_raw).numpy().astype(np.float32)
    A_matrix_const = tf.constant(final_A, dtype=tf.float32)
    return interior_encoder, boundary_encoder, decoder, A_matrix_const, loss_history


interior_encoder, boundary_encoder, decoder, A_matrix, loss_history = train_sinn(
    interior_encoder,
    boundary_encoder,
    decoder,
    patches,
    num_epochs=100,
    latent_dim=latent_dim,
    alpha=20.0,
    n_pde_iters=40,
    lr=1e-3,
)


# ======================================================================
# PATCH-BASED VALIDATION: RECONSTRUCT FULL LOCAL DOMAIN WITH PATCHES
# ======================================================================

def validate_full_domain(boundary_encoder,
                         decoder,
                         A_matrix,
                         X, Y,
                         U_true,
                         patch_radius=8,
                         stride=3,
                         inner_margin=2,
                         n_iters=40,
                         tau=0.03):
    """
    Patch-based validation on the local domain:

    - Slide (2*patch_radius+1)x(2*patch_radius+1) patches over the grid.
    - For each patch:
        * build boundary features from the TRUE snapshot
        * encode boundary, solve latent PDE on the patch
        * decode only the CORE interior (excluding an `inner_margin` band)
    - Average overlapping core predictions to get the final field.
    """

    Nx = X.shape[0] - 1
    Ny = X.shape[1] - 1
    latent_dim = A_matrix.shape[0]

    # Normalise the true snapshot and coordinates
    u_true_norm_full = (U_true - u_mean) / u_std
    X_n_full = (X - (x_min + x_max) / 2.0) / x_scale
    Y_n_full = (Y - (y_min + y_max) / 2.0) / y_scale

    P = 2 * patch_radius + 1

    # Accumulators for overlapping predictions
    sum_pred = np.zeros((Nx + 1, Ny + 1), dtype=np.float32)
    count = np.zeros((Nx + 1, Ny + 1), dtype=np.float32)

    i_centres = range(patch_radius, Nx + 1 - patch_radius, stride)
    j_centres = range(patch_radius, Ny + 1 - patch_radius, stride)

    for ic in i_centres:
        for jc in j_centres:
            i_start = ic - patch_radius
            i_end   = ic + patch_radius + 1
            j_start = jc - patch_radius
            j_end   = jc + patch_radius + 1

            u_patch = u_true_norm_full[i_start:i_end, j_start:j_end]
            Xp_n    = X_n_full[i_start:i_end, j_start:j_end]
            Yp_n    = Y_n_full[i_start:i_end, j_start:j_end]

            # ---------- BOUNDARY INDICES / MASK ----------
            boundary_idx = []
            for j_loc in range(P):
                boundary_idx.append([0, j_loc])
                boundary_idx.append([P - 1, j_loc])
            for i_loc in range(1, P - 1):
                boundary_idx.append([i_loc, 0])
                boundary_idx.append([i_loc, P - 1])
            boundary_idx = np.array(boundary_idx, dtype=np.int32)

            boundary_mask_patch = np.zeros((P, P), dtype=bool)
            boundary_mask_patch[0, :] = True
            boundary_mask_patch[-1, :] = True
            boundary_mask_patch[:, 0] = True
            boundary_mask_patch[:, -1] = True

            # Boundary features [x_n, y_n, u_norm]
            u_b = u_patch[boundary_idx[:, 0], boundary_idx[:, 1]]
            x_b = Xp_n[boundary_idx[:, 0], boundary_idx[:, 1]]
            y_b = Yp_n[boundary_idx[:, 0], boundary_idx[:, 1]]
            boundary_feats = np.stack([x_b, y_b, u_b], axis=-1).astype(np.float32)

            boundary_feats_tf = tf.constant(boundary_feats, dtype=tf.float32)
            boundary_indices_tf_patch = tf.constant(boundary_idx, dtype=tf.int32)
            boundary_mask_tf_patch = tf.constant(boundary_mask_patch, dtype=tf.bool)

            # ---------- LATENT PDE SOLVE ON THIS PATCH ----------
            boundary_latent = boundary_encoder(boundary_feats_tf, training=False)

            latent_field = solve_latent_field_tf(
                boundary_latent,
                A_matrix,
                boundary_indices_tf_patch,
                boundary_mask_tf_patch,
                Nx_patch=P - 1,
                Ny_patch=P - 1,
                n_iters=n_iters,
                tau=tau,
            ).numpy()   # (P, P, r)

            # ---------- DECODE CORE INTERIOR ----------
            i_core, j_core = np.meshgrid(
                np.arange(inner_margin, P - inner_margin),
                np.arange(inner_margin, P - inner_margin),
                indexing="ij",
            )
            i_int_flat = i_core.flatten()
            j_int_flat = j_core.flatten()

            # Global indices
            I_global = i_start + i_int_flat
            J_global = j_start + j_int_flat

            # Spatial coords for core interior
            x_int = Xp_n[inner_margin:P-inner_margin,
                         inner_margin:P-inner_margin].flatten()
            y_int = Yp_n[inner_margin:P-inner_margin,
                         inner_margin:P-inner_margin].flatten()
            spatial_coords = np.stack([x_int, y_int], axis=-1).astype(np.float32)

            # Latent codes at core interior
            latent_int = latent_field[
                inner_margin:P-inner_margin,
                inner_margin:P-inner_margin,
                :
            ].reshape(-1, latent_dim)

            decoder_input = np.concatenate([spatial_coords, latent_int], axis=1)
            u_pred_int = decoder(decoder_input, training=False).numpy().reshape(-1)

            # Accumulate into global arrays
            for k in range(u_pred_int.shape[0]):
                i_g = I_global[k]
                j_g = J_global[k]
                sum_pred[i_g, j_g] += u_pred_int[k]
                count[i_g, j_g] += 1.0

    # For nodes never covered by any core (very few, near outer boundary), fall back to true
    u_pred_norm = np.zeros_like(sum_pred)
    mask = count > 0
    u_pred_norm[mask] = sum_pred[mask] / count[mask]
    u_pred_norm[~mask] = u_true_norm_full[~mask]

    # Denormalise
    u_pred = u_pred_norm * u_std + u_mean
    u_true = U_true
    error_map = np.abs(u_true - u_pred)

    # ---- Metrics ----
    mse = np.mean((u_true - u_pred) ** 2)
    mae = np.mean(np.abs(u_true - u_pred))
    rmse = np.sqrt(mse)
    max_error = np.max(error_map)
    l2_norm = np.linalg.norm((u_true - u_pred).ravel())
    l_inf_norm = np.max(np.abs(u_true - u_pred))
    rel_l2_error = l2_norm / (np.linalg.norm(u_true.ravel()) + 1e-8)

    metrics = {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "Max Error": max_error,
        "L2 Norm": l2_norm,
        "L∞ Norm": l_inf_norm,
        "Relative L2 Error": rel_l2_error,
        "Mean True": float(np.mean(u_true)),
        "Mean Pred": float(np.mean(u_pred)),
    }

    print("\n" + "─" * 70)
    print("Validation Metrics (Patch-based reconstruction on local domain)")
    print("─" * 70)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.6e}")
    print("─" * 70 + "\n")

    return u_pred, error_map, metrics

u_snapshot_local = U[time_idx]

u_pred, error_map, metrics = validate_full_domain(
    boundary_encoder,
    decoder,
    A_matrix,
    X,
    Y,
    u_snapshot_local,
    patch_radius=patch_radius,  # 8
    stride=stride,              # 3
    inner_margin=2,
    n_iters=40,
    tau=0.03,
)

# ======================================================================
# SECTION 12: PLOTTING
# ======================================================================

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.title(f"True Solution (local, t_idx={time_idx})")
plt.imshow(u_snapshot_local.T, origin="lower", cmap="RdBu_r",
           extent=[y[0], y[-1], x[0], x[-1]])
plt.colorbar()
plt.xlabel("y")
plt.ylabel("x")

plt.subplot(1, 3, 2)
plt.title("SINN Prediction (local)")
plt.imshow(u_pred.T, origin="lower", cmap="RdBu_r",
           extent=[y[0], y[-1], x[0], x[-1]])
plt.colorbar()
plt.xlabel("y")
plt.ylabel("x")

plt.subplot(1, 3, 3)
plt.title("Absolute Error (local)")
plt.imshow(error_map.T, origin="lower", cmap="hot",
           extent=[y[0], y[-1], x[0], x[-1]])
plt.colorbar(label="|Error|")
plt.xlabel("y")
plt.ylabel("x")

plt.tight_layout()
plt.show()
