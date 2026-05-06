'''
## Flaws in Your Current Time-Dependent Approach

Based on careful review of the SINN paper and your implementation, here are the key flaws:

### 1. **Fundamental Architectural Mismatch with SINN Theory**

Your current approach violates the core SINN principle: **the elliptic system in the latent space should be purely spatial, without temporal evolution**.[1]

According to the paper:
> "Information is then passed from the boundary to the interior of the latent space using an elliptic system. This embeds a general class of well-posed PDEs into the SINN."[1]

The elliptic system is specifically designed for **boundary value problems (static spatial information transfer)**. By including time as a feature $$[x, y, t, u]$$, you're attempting to handle temporal dynamics with a spatially-oriented architecture, which is conceptually inconsistent.[1]

### 2. **Loss Function Does Not Enforce PDE Structure Across Time**

Your loss function (Section 4) computes:

$$\Psi_1 = \text{latent consistency error (spatial)}$$
$$\Psi_2 = \text{physical reconstruction error (spatial)}$$

**Problem:** There is **no temporal coupling constraint**. The network can learn any temporal pattern without respecting the underlying wave equation's dynamics. Each time step is treated independently, so there's nothing preventing non-physical temporal behavior.[1]

The paper's training approach explicitly enforces the elliptic structure through local patches that embed PDE information during training. Your approach lacks this structural enforcement.

### 3. **Training Uses All Time Data Simultaneously (Not Paper's Methodology)**

The paper trains **independent SINNs for each time snapshot**, treating $$u(x, y, t_i)$$ as separate spatial boundary observation problems.[1]

Your approach concatenates all temporal data, which:
- Mixes training information from vastly different physical states
- Doesn't allow the single shared network to specialize to individual time snapshots
- Makes it unclear whether the network is learning genuine temporal evolution or just memorizing data patterns across time

### 4. **Decoder Cannot Learn Temporal Physics**

The decoder $$\delta: [x, y, t, \ell] \rightarrow u$$ is a **semi-local network**. According to the paper:[1]

> "For any x ∈Ω, the value of (δℓ)(x) must only depend on the values of ℓ in a small neighbourhood Nx ⊂Ω of x."

By including time as a fourth dimension, you're forcing the decoder to extrapolate temporal behavior from local spatial-temporal neighborhoods. This is fundamentally different from the **global temporal evolution** that a wave equation demands. The wave equation couples all spatial points through time derivatives, which a semi-local decoder cannot capture.

### 5. **Boundary Encoder Cannot Properly Handle Time Derivatives**

For a time-dependent PDE like the wave equation, you need $$\frac{\partial u}{\partial t}$$ information. However, your boundary encoder takes $$[x, y, t, u]$$ as input.[1]

**Problem:** You're not providing any temporal derivative information to the boundary encoder, yet the wave equation fundamentally depends on $$\partial^2 u / \partial t^2$$. This information is simply not in your training data passed to the encoder.

### 6. **No Temporal Coupling in the Latent Space**

The elliptic system $$D_A \ell = 0$$ in the latent space is **static**—it has no time dependence.[1]

For a wave equation with temporal evolution, you need:
- Either a **parabolic or hyperbolic PDE** in the latent space (not elliptic), OR
- A **time-stepping scheme** that couples predictions from adjacent time steps

Your approach does neither. It solves a static elliptic system at each time snapshot independently.

***

## Summary of Fundamental Issues

| Issue | Your Approach | SINN Paper's Prescription |
|-------|---|---|
| **Temporal Evolution** | Encoded in time-conditioned networks | Train separate SINNs per time step |
| **Loss Function** | Spatial + physical reconstruction | Spatial + elliptic structure enforcement |
| **Latent PDE** | Static elliptic (but with time input) | Static elliptic (spatial only) |
| **Boundary Data** | $$[x, y, t, u]$$ without derivatives | $$[x, y, u]$$ + derivatives if available |
| **Temporal Coupling** | Via network weights across time | Via sequential independent predictions |

The paper acknowledges this limitation explicitly: it leaves **extending SINNs to time-dependent problems as future work**.[1]
'''

# ============================================================================
# SINN IMPLEMENTATION - TIME-DEPENDENT VERSION
# Processes all time steps with time as an input feature
# ============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ============================================================================
# SECTION 1: WAVE EQUATION SOLVER
# ============================================================================

def solve_wave_equation(wave_dict, init_dict):
    """Solves 2D wave equation and returns solution array U[time, x, y]"""
    Nx = int(wave_dict['Lx'] / wave_dict['dx'])
    Ny = int(wave_dict['Ly'] / wave_dict['dy'])
    x = np.linspace(0, wave_dict['Lx'], Nx + 1)
    y = np.linspace(0, wave_dict['Ly'], Ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    dt = wave_dict['CFL'] * min(wave_dict['dx'], wave_dict['dy']) / wave_dict['c'] / np.sqrt(2)
    Nt = int(wave_dict['Lt'] / dt)
    T = np.linspace(0, wave_dict['Lt'], Nt + 1)
    
    print(f"Wave Solver: Grid {Nx} × {Ny}, Nt={Nt}, dt={dt:.6f}")
    
    x_c = np.atleast_1d(init_dict['x_c'])
    y_c = np.atleast_1d(init_dict['y_c'])
    width = np.atleast_1d(init_dict['width'])
    amplitude = np.atleast_1d(init_dict['amplitude'])
    
    u0 = np.zeros_like(X, dtype=float)
    for xx_c, yy_c, w, a in zip(x_c, y_c, width, amplitude):
        u0 += a * np.exp(-((X - xx_c) ** 2 + (Y - yy_c) ** 2) / (2 * w**2))
    
    def apply_bc(U, bc_type='neumann'):
        if bc_type == 'dirichlet':
            U[0, :] = 0
            U[-1, :] = 0
            U[:, 0] = 0
            U[:, -1] = 0
        elif bc_type == 'neumann':
            U[0, :] = U[1, :]
            U[-1, :] = U[-2, :]
            U[:, 0] = U[:, 1]
            U[:, -1] = U[:, -2]
        return U
    
    U = np.zeros((Nt + 1, Nx + 1, Ny + 1))
    U[0] = u0
    
    alpha = (wave_dict['c'] * dt) ** 2
    
    Lap0 = (
        (U[0, 2:, 1:-1] - 2 * U[0, 1:-1, 1:-1] + U[0, :-2, 1:-1]) / wave_dict['dx'] ** 2
        + (U[0, 1:-1, 2:] - 2 * U[0, 1:-1, 1:-1] + U[0, 1:-1, :-2]) / wave_dict['dy'] ** 2
    )
    
    U[1, 1:-1, 1:-1] = U[0, 1:-1, 1:-1] + 0.5 * alpha * Lap0
    U[1] = apply_bc(U[1], wave_dict['bc_type'])
    
    for n in range(1, Nt):
        Lap = (
            (U[n, 2:, 1:-1] - 2 * U[n, 1:-1, 1:-1] + U[n, :-2, 1:-1]) / wave_dict['dx'] ** 2
            + (U[n, 1:-1, 2:] - 2 * U[n, 1:-1, 1:-1] + U[n, 1:-1, :-2]) / wave_dict['dy'] ** 2
        )
        
        U[n + 1, 1:-1, 1:-1] = (
            2 * U[n, 1:-1, 1:-1]
            - U[n - 1, 1:-1, 1:-1]
            + alpha * Lap
        )
        
        U[n + 1] = apply_bc(U[n + 1], wave_dict['bc_type'])
    
    print(f"Wave solution computed: U.shape = {U.shape}")
    return U, x, y, X, Y, T

wave_dict = {
    'c': 1.0,
    'Lt': 5.0,
    'Lx': 1.0,
    'Ly': 1.0,
    'dx': 0.05,
    'dy': 0.05,
    'CFL': 0.9,
    'bc_type': 'neumann',
}

init_dict = {
    'x_c': 0.25,
    'y_c': 0.5,
    'width': np.sqrt(0.005),
    'amplitude': 1.0
}

U, x, y, X, Y, T = solve_wave_equation(wave_dict, init_dict)

# ============================================================================
# SECTION 2: DATA PREPARATION FOR ALL TIME STEPS
# ============================================================================

def prepare_all_timesteps_data(U, X, Y, T, time_indices=None):
    """
    Prepare training data for all time steps (or subset if specified).
    Returns concatenated interior and boundary data with temporal dimension.
    """
    if time_indices is None:
        time_indices = np.arange(U.shape[0])
    
    all_interior_data = []
    all_boundary_data = []
    
    # Global normalization across all time steps
    u_mean = np.mean(U)
    u_std = np.std(U) + 1e-8
    
    for t_idx in time_indices:
        # Extract and normalize snapshot
        u_snapshot = np.array(U[t_idx], dtype=np.float32)
        u_norm = (u_snapshot - u_mean) / u_std
        u_norm = np.clip(u_norm, -3, 3)
        
        time_val = T[t_idx]
        
        # Extract interior points
        u_interior = u_norm[1:-1, 1:-1].flatten()
        x_interior = X[1:-1, 1:-1].flatten()
        y_interior = Y[1:-1, 1:-1].flatten()
        t_interior = np.full_like(u_interior, fill_value=time_val)
        
        interior_data = np.stack([x_interior, y_interior, t_interior, u_interior], axis=-1)
        all_interior_data.append(interior_data)
        
        # Extract boundary points
        u_boundary = []
        x_boundary = []
        y_boundary = []
        
        # Top edge
        for j in range(Y.shape[1]):
            u_boundary.append(u_norm[0, j])
            x_boundary.append(X[0, j])
            y_boundary.append(Y[0, j])
        
        # Bottom edge
        for j in range(Y.shape[1]):
            u_boundary.append(u_norm[-1, j])
            x_boundary.append(X[-1, j])
            y_boundary.append(Y[-1, j])
        
        # Left edge (skip corners)
        for i in range(1, X.shape[0] - 1):
            u_boundary.append(u_norm[i, 0])
            x_boundary.append(X[i, 0])
            y_boundary.append(Y[i, 0])
        
        # Right edge (skip corners)
        for i in range(1, X.shape[0] - 1):
            u_boundary.append(u_norm[i, -1])
            x_boundary.append(X[i, -1])
            y_boundary.append(Y[i, -1])
        
        u_boundary = np.array(u_boundary)
        x_boundary = np.array(x_boundary)
        y_boundary = np.array(y_boundary)
        t_boundary = np.full_like(u_boundary, fill_value=time_val)
        
        boundary_data = np.stack([x_boundary, y_boundary, t_boundary, u_boundary], axis=-1)
        all_boundary_data.append(boundary_data)
    
    # Concatenate all time steps
    interior_feats = np.concatenate(all_interior_data, axis=0)
    boundary_feats = np.concatenate(all_boundary_data, axis=0)
    
    print(f"Prepared data for {len(time_indices)} time steps")
    print(f"Interior features shape: {interior_feats.shape}")
    print(f"Boundary features shape: {boundary_feats.shape}")
    print(f"Normalization: mean={u_mean:.6f}, std={u_std:.6f}")
    
    return interior_feats, boundary_feats, u_mean, u_std

# Prepare data for ALL time steps
time_indices = np.arange(U.shape[0])  # All time steps
interior_feats, boundary_feats, u_mean, u_std = prepare_all_timesteps_data(
    U, X, Y, T, time_indices=time_indices
)

interior_feats_tf = tf.constant(interior_feats, dtype=tf.float32)
boundary_feats_tf = tf.constant(boundary_feats, dtype=tf.float32)

# ============================================================================
# SECTION 3: TIME-DEPENDENT ENCODERS AND DECODER
# ============================================================================

def make_interior_encoder(num_latentdim, num_units):
    """Interior encoder: [x, y, t, u] → ℓ (now includes time)"""
    inputs = Input(shape=(4,), name='interior_input')  # x, y, t, u
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(inputs)
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.2)(x)
    outputs = Dense(num_latentdim, activation=None)(x)
    return Model(inputs, outputs, name='interior_encoder')

def make_boundary_encoder(num_latentdim, num_units):
    """Boundary encoder: [x, y, t, u] → ℓ (now includes time)"""
    inputs = Input(shape=(4,), name='boundary_input')  # x, y, t, u
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(inputs)
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.2)(x)
    outputs = Dense(num_latentdim, activation=None)(x)
    return Model(inputs, outputs, name='boundary_encoder')

def make_decoder(num_latentdim, num_units):
    """Decoder: [x, y, t, ℓ] → u (now includes time)"""
    inputs = Input(shape=(num_latentdim + 3,), name='decoder_input')  # x, y, t, ℓ
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(inputs)
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.2)(x)
    outputs = Dense(1, activation=None)(x)
    return Model(inputs, outputs, name='decoder')

num_latentdim = 10
interior_encoder = make_interior_encoder(num_latentdim=num_latentdim, num_units=128)
boundary_encoder = make_boundary_encoder(num_latentdim=num_latentdim, num_units=128)
decoder = make_decoder(num_latentdim=num_latentdim, num_units=128)

# ============================================================================
# SECTION 4: LOSS FUNCTION FOR ALL TIME STEPS
# ============================================================================

def compute_loss(interior_latent_true, boundary_latent, decoder,
                 interior_feats, A_matrix, latent_dim=8, alpha=1.0):
    """Loss computation with time-aware models"""
    batch_size = tf.shape(interior_feats)[0]
    boundary_latent_mean = tf.reduce_mean(boundary_latent, axis=0)  # (latent_dim,)
    
    A_diag = tf.linalg.diag_part(A_matrix)
    
    # Solve PDE for each latent dimension
    ell_pred_list = []
    for dim in range(latent_dim):
        bc_val = boundary_latent_mean[dim]
        weight = A_diag[dim]
        
        patch = tf.ones((5, 5), dtype=tf.float32) * bc_val
        for _ in range(20):
            interior = patch[1:-1, 1:-1]
            patch_updated = tf.tensor_scatter_nd_update(
                patch,
                tf.stack(tf.meshgrid(tf.range(1, 4), tf.range(1, 4), indexing='ij'), axis=-1),
                weight * (
                    patch[:-2, 1:-1] + patch[2:, 1:-1] +
                    patch[1:-1, :-2] + patch[1:-1, 2:]
                ) / 4.0
            )
            patch = patch_updated
        
        ell_pred_list.append(patch[2, 2])
    
    ell_pred = tf.stack(ell_pred_list, axis=0)
    ell_pred = tf.tile(tf.expand_dims(ell_pred, 0), [batch_size, 1])
    
    # Loss term 1: Latent space consistency
    loss_latent = tf.reduce_mean(tf.square(interior_latent_true - ell_pred))
    
    # Loss term 2: Physical space reconstruction
    spatial_temporal_coords = interior_feats[:, :3]  # [x, y, t]
    decoder_input = tf.concat([spatial_temporal_coords, ell_pred], axis=1)
    u_pred = decoder(decoder_input, training=True)
    u_true = tf.expand_dims(interior_feats[:, 3], axis=1)
    
    loss_physical = tf.reduce_mean(tf.square(u_true - u_pred))
    
    total_loss = loss_latent + alpha * loss_physical
    
    return total_loss, loss_latent, loss_physical

# ============================================================================
# SECTION 5: TRAINING ON ALL TIME STEPS
# ============================================================================

def train(interior_encoder, boundary_encoder, decoder,
          interior_feats_tf, boundary_feats_tf,
          num_epochs=100, num_latentdim=8):
    """Training loop for time-dependent SINN"""
    
    A_matrix = tf.Variable(np.eye(num_latentdim, dtype=np.float32),
                          trainable=True, dtype=tf.float32, name='pde_operator')
    
    trainable_vars = (
        interior_encoder.trainable_variables +
        boundary_encoder.trainable_variables +
        decoder.trainable_variables +
        [A_matrix]
    )
    
    print(f"Trainable variables: {len(trainable_vars)}")
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss_history = []
    
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            interior_latent = interior_encoder(interior_feats_tf, training=True)
            boundary_latent = boundary_encoder(boundary_feats_tf, training=True)
            
            total_loss, loss_lat, loss_phy = compute_loss(
                interior_latent, boundary_latent, decoder,
                interior_feats_tf, A_matrix, latent_dim=num_latentdim, alpha=3.0
            )
        
        grads = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        loss_history.append(float(total_loss))
        
        if (epoch + 1) % (num_epochs // 10) == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss:.6f}, Ψ_latent={loss_lat:.6f}, Ψ_phys={loss_phy:.6f}")
    
    return interior_encoder, boundary_encoder, decoder, A_matrix, loss_history

print("\n" + "="*70)
print("TRAINING SINN ON ALL TIME STEPS")
print("="*70 + "\n")

interior_encoder, boundary_encoder, decoder, A_matrix, loss_history = train(
    interior_encoder, boundary_encoder, decoder, 
    interior_feats_tf, boundary_feats_tf, 
    num_epochs=200, num_latentdim=num_latentdim
)

# ============================================================================
# SECTION 6: VALIDATION ON EACH TIME STEP
# ============================================================================

def validate_single_timestep(boundary_encoder, decoder, A_matrix, 
                             boundary_feats_tf, u_mean, u_std,
                             X, Y, T, U_true, time_idx, n_iters=100):
    """Validate SINN prediction for a single time step"""
    
    Nx, Ny = X.shape[0] - 1, X.shape[1] - 1
    latent_dim = A_matrix.shape[0]
    
    # Encode boundary at this time step
    boundary_latent = boundary_encoder(boundary_feats_tf, training=False).numpy()
    
    # For this specific time step, filter boundary data
    t_target = T[time_idx]
    
    # Create mapping from (x,y) to latent code for this time
    boundary_dict = {}
    boundary_count = 0
    
    # Re-extract boundary features for this time step only
    u_snapshot = np.array(U_true[time_idx], dtype=np.float32)
    u_norm = (u_snapshot - u_mean) / u_std
    u_norm = np.clip(u_norm, -3, 3)
    
    for j in range(Y.shape[1]):
        xi, yi = 0, j
        boundary_dict[(xi, yi)] = boundary_latent[boundary_count]
        boundary_count += 1
    
    for j in range(Y.shape[1]):
        xi, yi = Nx, j
        boundary_dict[(xi, yi)] = boundary_latent[boundary_count]
        boundary_count += 1
    
    for i in range(1, Nx):
        xi, yi = i, 0
        boundary_dict[(xi, yi)] = boundary_latent[boundary_count]
        boundary_count += 1
    
    for i in range(1, Nx):
        xi, yi = i, Ny
        boundary_dict[(xi, yi)] = boundary_latent[boundary_count]
        boundary_count += 1
    
    # Solve PDE with spatially-varying boundary
    latent_field = np.zeros((Nx + 1, Ny + 1, latent_dim), dtype=np.float32)
    A_matrix_np = A_matrix.numpy()
    
    for dim in range(latent_dim):
        print(f"  Latent dimension {dim+1}/{latent_dim}...")
        
        ell_field = np.zeros((Nx + 1, Ny + 1), dtype=np.float32)
        boundary_mask = np.zeros((Nx + 1, Ny + 1), dtype=bool)
        
        # Set boundary values
        for (xi, yi), latent_code in boundary_dict.items():
            if 0 <= xi < Nx+1 and 0 <= yi < Ny+1:
                ell_field[xi, yi] = latent_code[dim]
                boundary_mask[xi, yi] = True
        
        # Fill remaining edges
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                if i == 0 or i == Nx or j == 0 or j == Ny:
                    boundary_mask[i, j] = True
                    if ell_field[i, j] == 0:
                        dists = [(abs(i - xi), abs(j - yi)) for xi, yi in boundary_dict.keys()]
                        if dists:
                            nearest_idx = np.argmin(dists)
                            nearest_key = list(boundary_dict.keys())[nearest_idx]
                            ell_field[i, j] = boundary_dict[nearest_key][dim]
        
        weight = float(A_matrix_np[dim, dim])
        
        # Jacobi iteration
        for iteration in range(n_iters):
            ell_old = ell_field.copy()
            for i in range(1, Nx):
                for j in range(1, Ny):
                    if not boundary_mask[i, j]:
                        laplacian = (
                            ell_old[i+1, j] + ell_old[i-1, j] +
                            ell_old[i, j+1] + ell_old[i, j-1] - 4 * ell_old[i, j]
                        )
                        ell_field[i, j] = ell_old[i, j] + 0.25 * weight * laplacian
        
        latent_field[:, :, dim] = ell_field
    
    # Decode latent field
    print(f"\n  Decoding latent field to physical space...")
    u_pred = np.zeros((Nx + 1, Ny + 1), dtype=np.float32)
    
    for i in range(Nx + 1):
        if (i + 1) % max(1, (Nx // 5)) == 0:
            print(f"    Row {i+1}/{Nx+1}")
        
        for j in range(Ny + 1):
            latent_code = latent_field[i, j, :]
            spatial_temporal_input = np.array([X[i, j], Y[i, j], t_target], dtype=np.float32)
            decoder_input = np.concatenate([spatial_temporal_input, latent_code])
            decoder_input_tf = tf.constant(decoder_input[np.newaxis, :], dtype=tf.float32)
            u_pred_norm = decoder(decoder_input_tf, training=False).numpy().squeeze()
            u_pred[i, j] = float(u_pred_norm) * u_std + u_mean
    
    # Compute metrics
    u_true = U_true[time_idx]
    error_map = np.abs(u_true - u_pred)
    
    mse = np.mean((u_true - u_pred) ** 2)
    mae = np.mean(np.abs(u_true - u_pred))
    rmse = np.sqrt(mse)
    max_error = np.max(error_map)
    l2_norm = np.linalg.norm(error_map.flatten())
    l_inf_norm = np.max(np.abs(u_true - u_pred))
    u_true_norm = np.linalg.norm(u_true.flatten())
    rel_l2_error = l2_norm / (u_true_norm + 1e-8)
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Max Error': max_error,
        'L2 Norm': l2_norm,
        'L∞ Norm': l_inf_norm,
        'Relative L2 Error': rel_l2_error,
        'Mean Prediction': np.mean(u_pred),
        'Mean True': np.mean(u_true)
    }
    
    return u_pred, latent_field, error_map, metrics

# ============================================================================
# SECTION 7: VALIDATE ON MULTIPLE TIME STEPS
# ============================================================================

print("\n" + "="*70)
print("VALIDATING SINN ON SELECTED TIME STEPS")
print("="*70)

# Select a subset of time steps for validation
validation_time_indices = np.array([5, 10, 15, 20, 30])
validation_time_indices = validation_time_indices[validation_time_indices < U.shape[0]]

validation_results = {}

for t_idx in validation_time_indices:
    print(f"\nValidating time step {t_idx} (t = {T[t_idx]:.4f})...")
    print("-" * 70)
    
    u_pred, latent_field, error_map, metrics = validate_single_timestep(
        boundary_encoder, decoder, A_matrix, boundary_feats_tf, u_mean, u_std,
        X, Y, T, U, t_idx, n_iters=200
    )
    
    validation_results[t_idx] = {
        'u_pred': u_pred,
        'latent_field': latent_field,
        'error_map': error_map,
        'metrics': metrics
    }
    
    print(f"\nMetrics for time step {t_idx}:")
    print(f"  MSE: {metrics['MSE']:.6e}")
    print(f"  MAE: {metrics['MAE']:.6e}")
    print(f"  RMSE: {metrics['RMSE']:.6e}")
    print(f"  Relative L2 Error: {metrics['Relative L2 Error']:.6e}")

# ============================================================================
# SECTION 8: VISUALIZATION
# ============================================================================

def plot_all_timesteps_comparison(U_true, validation_results, T, validation_time_indices):
    """
    Plot all validation timesteps in one figure with subplots.
    Each row contains: True Solution | SINN Prediction | Absolute Error
    """
    num_timesteps = len(validation_time_indices)
    fig, axes = plt.subplots(num_timesteps, 3, figsize=(16, 5*num_timesteps))
    
    # Flatten axes for easier indexing if there's only one timestep
    if num_timesteps == 1:
        axes = axes.reshape(1, -1)
    
    # Find global max for consistent color scaling
    u_true_all = U_true[validation_time_indices]
    global_vmax = np.max(np.abs(u_true_all))
    
    error_vmax = 0
    for t_idx in validation_time_indices:
        error_vmax = max(error_vmax, np.max(validation_results[t_idx]['error_map']))
    
    for row, t_idx in enumerate(validation_time_indices):
        u_true = U_true[t_idx]
        u_pred = validation_results[t_idx]['u_pred']
        error_map = validation_results[t_idx]['error_map']
        metrics = validation_results[t_idx]['metrics']
        t_val = T[t_idx]
        
        # Column 0: True Solution
        im0 = axes[row, 0].imshow(u_true, origin='lower', cmap='RdBu_r', 
                                   vmin=-global_vmax, vmax=global_vmax)
        axes[row, 0].set_title(f'True Solution (t_idx={t_idx}, t={t_val:.4f})', 
                               fontsize=11, fontweight='bold')
        axes[row, 0].set_xlabel('y', fontsize=10)
        axes[row, 0].set_ylabel('x', fontsize=10)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        
        # Column 1: SINN Prediction
        im1 = axes[row, 1].imshow(u_pred, origin='lower', cmap='RdBu_r', 
                                   vmin=-global_vmax, vmax=global_vmax)
        axes[row, 1].set_title(f'SINN Prediction | MAE: {metrics["MAE"]:.2e}', 
                               fontsize=11, fontweight='bold')
        axes[row, 1].set_xlabel('y', fontsize=10)
        axes[row, 1].set_ylabel('x', fontsize=10)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        
        # Column 2: Absolute Error
        im2 = axes[row, 2].imshow(error_map, origin='lower', cmap='hot_r')
        axes[row, 2].set_title(f'Absolute Error | Max: {metrics["Max Error"]:.2e}, RMSE: {metrics["RMSE"]:.2e}', 
                               fontsize=11, fontweight='bold')
        axes[row, 2].set_xlabel('y', fontsize=10)
        axes[row, 2].set_ylabel('x', fontsize=10)
        cbar = plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
        cbar.set_label('|Error|', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('sinn_validation_all_timesteps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved as 'sinn_validation_all_timesteps.png'")
    
# Call the comprehensive visualization
plot_all_timesteps_comparison(U, validation_results, T, validation_time_indices)

def plot_training_history(loss_history):
    """Plot training loss over epochs"""
    epochs = np.arange(1, len(loss_history) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(epochs, loss_history, 'b-', linewidth=2, label='Total Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss History (All Time Steps)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_training_history(loss_history)

def plot_error_distribution(error_map):
    """Histogram of prediction errors"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(error_map.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Absolute Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.axvline(np.mean(error_map), color='r', linestyle='--', linewidth=2, 
              label=f'Mean: {np.mean(error_map):.2e}')
    ax.axvline(np.median(error_map), color='g', linestyle='--', linewidth=2, 
              label=f'Median: {np.median(error_map):.2e}')
    ax.legend()
    plt.tight_layout()
    plt.show()

# plot_error_distribution(validation_results[first_t_idx]['error_map'])

# ============================================================================
# SECTION 9: SUMMARY OF RESULTS ACROSS TIME STEPS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY OF VALIDATION RESULTS ACROSS TIME STEPS")
print("="*70)

summary_table = []
for t_idx in sorted(validation_results.keys()):
    metrics = validation_results[t_idx]['metrics']
    summary_table.append({
        'Time Index': t_idx,
        'Time': f"{T[t_idx]:.4f}",
        'MSE': f"{metrics['MSE']:.6e}",
        'MAE': f"{metrics['MAE']:.6e}",
        'RMSE': f"{metrics['RMSE']:.6e}",
        'Rel L2 Error': f"{metrics['Relative L2 Error']:.6e}"
    })

import pandas as pd
df_summary = pd.DataFrame(summary_table)
print("\n", df_summary.to_string(index=False))

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
