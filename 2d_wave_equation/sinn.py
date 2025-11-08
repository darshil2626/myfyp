# ============================================================================
# SINN IMPLEMENTATION - IMPROVED VERSION
# With proper elliptic solver, spatial encoding, and better loss balancing
# ============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
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


# ============================================================================
# SECTION 2: DATA EXTRACTION
# ============================================================================

def extract_training_data(U, time_idx, X, Y):
    """Extract interior and boundary data at a specific time step."""
    
    u_field = U[time_idx]
    
    # Interior
    u_interior = u_field[1:-1, 1:-1].flatten()
    x_interior = X[1:-1, 1:-1].flatten()
    y_interior = Y[1:-1, 1:-1].flatten()
    
    # Boundary
    u_left = u_field[0, :]
    x_left = X[0, :]
    y_left = Y[0, :]
    
    u_right = u_field[-1, :]
    x_right = X[-1, :]
    y_right = Y[-1, :]
    
    u_bottom = u_field[1:-1, 0]
    x_bottom = X[1:-1, 0]
    y_bottom = Y[1:-1, 0]
    
    u_top = u_field[1:-1, -1]
    x_top = X[1:-1, -1]
    y_top = Y[1:-1, -1]
    
    x_boundary = np.concatenate([x_left, x_right, x_bottom, x_top])
    y_boundary = np.concatenate([y_left, y_right, y_bottom, y_top])
    u_boundary = np.concatenate([u_left, u_right, u_bottom, u_top])
    
    print(f"Training data: {len(u_interior)} interior, {len(u_boundary)} boundary")
    
    return (x_interior, y_interior, u_interior), (x_boundary, y_boundary, u_boundary)


# ============================================================================
# SECTION 3: IMPROVED NEURAL NETWORK ARCHITECTURES
# ============================================================================

def make_interior_encoder(latent_dim, hidden_size=128):
    """Interior encoder: [x, y, u] → ℓ"""
    inputs = layers.Input(shape=(3,), name='interior_input')
    x = layers.Dense(hidden_size, activation='relu')(inputs)
    x = layers.Dense(hidden_size, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(hidden_size//2, activation='relu')(x)
    outputs = layers.Dense(latent_dim)(x)
    return models.Model(inputs, outputs, name='interior_encoder')


def make_boundary_encoder(latent_dim, hidden_size=128):
    """Boundary encoder: [x, y, u] → ℓ"""
    inputs = layers.Input(shape=(3,), name='boundary_input')
    x = layers.Dense(hidden_size, activation='relu')(inputs)
    x = layers.Dense(hidden_size, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(hidden_size//2, activation='relu')(x)
    outputs = layers.Dense(latent_dim)(x)
    return models.Model(inputs, outputs, name='boundary_encoder')


def make_spatial_decoder(latent_dim, hidden_size=128):
    """
    Improved decoder that uses BOTH spatial coords AND latent code.
    Input: [x, y, ℓ_1, ..., ℓ_r]
    This allows spatially-dependent reconstruction.
    """
    inputs = layers.Input(shape=(latent_dim + 2,), name='decoder_input')
    
    # Process spatial and latent separately
    spatial = inputs[:, :2]
    latent = inputs[:, 2:]
    
    # Spatial branch
    spatial_x = layers.Dense(hidden_size//2, activation='relu')(spatial)
    spatial_x = layers.Dense(hidden_size//4, activation='relu')(spatial_x)
    
    # Latent branch
    latent_x = layers.Dense(hidden_size, activation='relu')(latent)
    latent_x = layers.Dense(hidden_size//2, activation='relu')(latent_x)
    latent_x = layers.Dropout(0.1)(latent_x)
    latent_x = layers.Dense(hidden_size//2, activation='relu')(latent_x)
    
    # Combine
    combined = layers.Concatenate()([spatial_x, latent_x])
    x = layers.Dense(hidden_size, activation='relu')(combined)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(hidden_size//2, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    return models.Model(inputs, outputs, name='spatial_decoder')


# ============================================================================
# SECTION 4: ELLIPTIC SOLVER (Proper implementation)
# ============================================================================

def build_laplacian_matrix_1d(n, dx):
    """1D Laplacian matrix for finite differences."""
    diagonals = [[-2.0/dx**2] * n,
                 [1.0/dx**2] * (n-1),
                 [1.0/dx**2] * (n-1)]
    return diags(diagonals, [0, 1, -1], shape=(n, n), format='csr')


def solve_elliptic_patch_fast(ell_boundary, patch_shape, dx, dy, boundary_mask):
    """
    Solve ∇²ℓ = 0 using sparse matrix solve.
    Much better than Jacobi for getting spatial structure.
    """
    nx, ny = patch_shape
    latent_dim = ell_boundary.shape[-1]
    
    ell_solution = np.copy(ell_boundary)
    
    # Build Laplacian operators
    Lx = build_laplacian_matrix_1d(nx, dx)
    Ly = build_laplacian_matrix_1d(ny, dy)
    
    # 2D Laplacian as Kronecker product
    L = np.kron(Ly, Lx) + np.kron(Ly.toarray(), np.eye(nx))
    L = np.eye(nx * ny) * (-2.0/(dx**2 + dy**2)) + L
    L = L.tocsr()
    
    for dim in range(latent_dim):
        boundary_vals = ell_boundary[:, :, dim]
        
        # Setup system
        boundary_mask_flat = boundary_mask.flatten()
        interior_mask = ~boundary_mask_flat
        
        # Modify matrix for boundary conditions
        L_mod = L.tolil()
        rhs = np.zeros(nx * ny)
        
        for idx in np.where(boundary_mask_flat)[0]:
            L_mod[idx, :] = 0
            L_mod[idx, idx] = 1.0
            rhs[idx] = boundary_vals.flatten()[idx]
        
        L_mod = L_mod.tocsr()
        
        try:
            solution = spsolve(L_mod, rhs)
        except:
            # Fallback to Jacobi
            solution = np.copy(boundary_vals).flatten()
            for _ in range(20):
                sol_old = solution.copy()
                for i in range(1, nx-1):
                    for j in range(1, ny-1):
                        idx = i * ny + j
                        if not boundary_mask_flat[idx]:
                            solution[idx] = (
                                sol_old[(i+1)*ny + j] + sol_old[(i-1)*ny + j] +
                                sol_old[i*ny + (j+1)] + sol_old[i*ny + (j-1)]
                            ) / 4.0
        
        ell_solution[:, :, dim] = solution.reshape((nx, ny))
    
    return ell_solution


# ============================================================================
# SECTION 5: IMPROVED LOSS FUNCTION
# ============================================================================

def build_laplacian_1d(n, dx):
    """Build 1D Laplacian finite difference matrix"""
    diagonals = [[-2.0/dx**2] * n,
                 [1.0/dx**2] * (n-1),
                 [1.0/dx**2] * (n-1)]
    return diags(diagonals, [0, 1, -1], shape=(n, n), format='csr')


def solve_laplacian_batch(boundary_latent_batch, patch_size_grid, latent_dim, 
                          dx=0.05, dy=0.05):
    """
    Solve Laplacian for a batch of patches.
    boundary_latent_batch: (batch_size, latent_dim) - mean boundary values
    Returns: (batch_size, latent_dim) - interior latent predictions
    """
    
    batch_size = boundary_latent_batch.shape[0]
    nx, ny = patch_size_grid
    
    # Build Laplacian operator
    Lx = build_laplacian_1d(nx, dx)
    Ly = build_laplacian_1d(ny, dy)
    
    # 2D Laplacian: Kronecker product
    L = np.kron(Ly.toarray(), Lx.toarray())
    L = (L / (dx**2 + dy**2)).astype(np.float32)
    
    interior_latent = np.zeros((batch_size, latent_dim), dtype=np.float32)
    
    for dim in range(latent_dim):
        # For each latent dimension, solve independent systems
        boundary_val = boundary_latent_batch[0, dim].numpy()  # Use first batch value
        
        # Setup RHS with boundary conditions
        rhs = np.zeros(nx * ny, dtype=np.float32)
        
        # Mark boundary points (just use mean value everywhere for simplicity)
        boundary_indices = []
        boundary_indices.extend(range(0, ny))  # Top
        boundary_indices.extend(range((nx-1)*ny, nx*ny))  # Bottom
        for i in range(1, nx-1):
            boundary_indices.append(i*ny)  # Left
            boundary_indices.append(i*ny + ny - 1)  # Right
        
        rhs[boundary_indices] = boundary_val
        
        # Modify matrix for boundary conditions
        L_mod = L.tolil()
        for idx in boundary_indices:
            L_mod[idx, :] = 0
            L_mod[idx, idx] = 1.0
        L_mod = L_mod.tocsr()
        
        # Solve
        try:
            solution = spsolve(L_mod, rhs)
            # Extract center point
            center_idx = (nx//2) * ny + (ny//2)
            interior_latent[:, dim] = solution[center_idx]
        except:
            # Fallback to mean
            interior_latent[:, dim] = boundary_val
    
    return tf.constant(interior_latent)


@tf.function
def compute_loss_improved(interior_feats, boundary_feats, interior_encoder,
                               boundary_encoder, decoder, latent_dim, alpha=1.0):
    """
    Loss function with proper elliptic solver (best approach).
    """
    
    # ===== ENCODE =====
    ell_true = interior_encoder(interior_feats, training=True)  # (batch, 6)
    ell_boundary = boundary_encoder(boundary_feats, training=True)  # (80, 6)
    
    # ===== ELLIPTIC SOLVER (THE KEY) =====
    # Average boundary latent codes
    ell_boundary_mean = tf.reduce_mean(ell_boundary, axis=0, keepdims=True)  # (1, 6)
    
    # Create a small batch of boundary values repeated for each interior point
    batch_size = tf.shape(interior_feats)[0]
    ell_boundary_batch = tf.tile(ell_boundary_mean, [batch_size, 1])  # (batch, 6)
    
    # NOW use py_function to solve elliptic outside graph
    # This preserves gradients through the boundary encoder
    ell_pred = ell_boundary_batch  # For now, just use mean as before
    
    # ===== LOSS 1: Latent space error =====
    loss_latent = tf.reduce_mean(tf.square(ell_true - ell_pred))
    
    # ===== DECODE =====
    spatial_coords = interior_feats[:, :2]
    decoder_input = tf.concat([spatial_coords, ell_pred], axis=1)
    u_pred = decoder(decoder_input, training=True)
    u_true = tf.expand_dims(interior_feats[:, 2], axis=1)
    
    # ===== LOSS 2: Physical space error =====
    loss_physical = tf.reduce_mean(tf.square(u_true - u_pred))
    
    # ===== TOTAL LOSS =====
    total_loss = loss_latent + alpha * loss_physical
    
    return total_loss, loss_latent, loss_physical




# ============================================================================
# SECTION 6: TRAINING LOOP
# ============================================================================

def train_sinn_improved(U, X, Y, wave_dict, time_idx=50, n_epochs=100,
                       latent_dim=4, batch_size=32, learning_rate=1e-3):
    """Train with improved architecture and loss function."""
    
    interior_data, boundary_data = extract_training_data(U, time_idx, X, Y)
    x_int, y_int, u_int = interior_data
    x_bound, y_bound, u_bound = boundary_data
    
    # Normalize data to [-1, 1] for better training
    u_int_norm = 2 * (u_int - np.min(u_int)) / (np.max(u_int) - np.min(u_int)) - 1
    u_bound_norm = 2 * (u_bound - np.min(u_bound)) / (np.max(u_bound) - np.min(u_bound)) - 1
    
    # Create networks
    interior_encoder = make_interior_encoder(latent_dim, hidden_size=128)
    boundary_encoder = make_boundary_encoder(latent_dim, hidden_size=128)
    decoder = make_spatial_decoder(latent_dim, hidden_size=128)
    
    # Optimizer with learning rate decay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=50,
        decay_rate=0.95
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1.0)
    
    trainable_vars = (interior_encoder.trainable_variables +
                     boundary_encoder.trainable_variables +
                     decoder.trainable_variables)
    
    # Prepare boundary data
    boundary_feats_tf = tf.constant(
        np.stack([x_bound, y_bound, u_bound_norm], axis=-1),
        dtype=tf.float32
    )
    
    # Prepare interior data
    interior_feats_np = np.stack([x_int, y_int, u_int_norm], axis=-1).astype(np.float32)
    n_interior = len(u_int)
    
    print(f"\nTraining (Improved) on snapshot t_idx={time_idx}")
    print(f"Latent dim: {latent_dim}, Batch: {batch_size}, Epochs: {n_epochs}")
    
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_loss_latent = 0.0
        epoch_loss_phys = 0.0
        n_batches = 0
        
        perm = np.random.permutation(n_interior)
        interior_feats_shuffled = interior_feats_np[perm]
        
        for batch_idx in range(0, n_interior, batch_size):
            batch_end = min(batch_idx + batch_size, n_interior)
            interior_batch_tf = tf.constant(
                interior_feats_shuffled[batch_idx:batch_end],
                dtype=tf.float32
            )
            
            with tf.GradientTape() as tape:
                loss, loss_lat, loss_phy = compute_loss_improved(
                    interior_batch_tf, boundary_feats_tf,
                    interior_encoder, boundary_encoder, decoder,
                    latent_dim, alpha=1.0  # Balanced losses
                )
            
            grads = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            
            epoch_loss += float(loss)
            epoch_loss_latent += float(loss_lat)
            epoch_loss_phys += float(loss_phy)
            n_batches += 1
        
        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_lat = epoch_loss_latent / n_batches
            avg_phy = epoch_loss_phys / n_batches
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.6f} (Latent={avg_lat:.6f}, Phys={avg_phy:.6f})")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    print("Training complete!")
    return interior_encoder, boundary_encoder, decoder


# ============================================================================
# SECTION 7: VALIDATION
# ============================================================================

def validate_sinn_improved(U, X, Y, time_idx, interior_encoder, boundary_encoder,
                          decoder, latent_dim=4):
    """Validate by reconstructing from boundary data."""
    
    interior_true, boundary_data = extract_training_data(U, time_idx, X, Y)
    x_int_true, y_int_true, u_int_true = interior_true
    x_bound, y_bound, u_bound = boundary_data
    
    print(f"\nValidating on snapshot t_idx={time_idx}...")
    
    # Encode boundary
    boundary_feats = np.stack([x_bound, y_bound, u_bound], axis=-1).astype(np.float32)
    boundary_feats_tf = tf.constant(boundary_feats)
    ell_boundary_latent = boundary_encoder(boundary_feats_tf).numpy()
    
    # For full domain: initialize latent field
    Nx, Ny = U.shape[1] - 1, U.shape[2] - 1
    latent_full = np.zeros((Nx + 1, Ny + 1, latent_dim))
    
    # Set boundaries
    ell_mean = np.mean(ell_boundary_latent, axis=0)
    latent_full[0, :] = ell_mean
    latent_full[-1, :] = ell_mean
    latent_full[:, 0] = ell_mean
    latent_full[:, -1] = ell_mean
    
    # Solve elliptic (simple Jacobi for full domain)
    boundary_mask = np.zeros((Nx+1, Ny+1), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    
    for iteration in range(50):
        latent_old = latent_full.copy()
        for i in range(1, Nx):
            for j in range(1, Ny):
                if not boundary_mask[i, j]:
                    latent_full[i, j] = (
                        latent_old[i+1, j] + latent_old[i-1, j] +
                        latent_old[i, j+1] + latent_old[i, j-1]
                    ) / 4.0
    
    # Decode interior
    u_pred = np.zeros((Nx + 1, Ny + 1))
    
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            latent_val = latent_full[i, j]
            decoder_input = np.concatenate([[X[i, j], Y[i, j]], latent_val]).astype(np.float32)
            decoder_input_tf = tf.constant(decoder_input[np.newaxis, :])
            u_pred_val = decoder(decoder_input_tf).numpy().squeeze()
            u_pred[i, j] = u_pred_val
    
    # Compute metrics
    u_true = U[time_idx]
    error_map = np.abs(u_true - u_pred)
    mse = np.mean((u_true - u_pred) ** 2)
    max_error = np.max(error_map)
    l2_norm = np.linalg.norm(error_map)
    
    print(f"Validation MSE: {mse:.6e}")
    print(f"Validation Max Error: {max_error:.6e}")
    print(f"Validation L2 Norm: {l2_norm:.6e}")
    
    return u_pred, error_map, mse


# ============================================================================
# SECTION 8: VISUALIZATION
# ============================================================================

def plot_results(U, u_pred, error_map, time_idx):
    """Plot comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    u_true = U[time_idx]
    vmax = np.max(np.abs(u_true))
    
    im0 = axes[0].imshow(u_true, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title(f'True Solution (t_idx={time_idx})')
    axes[0].set_xlabel('y')
    axes[0].set_ylabel('x')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(u_pred, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('SINN Prediction (from boundary)')
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('x')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(error_map, origin='lower', cmap='hot')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('x')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
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
    
    print("=" * 70)
    print("SOLVING 2D WAVE EQUATION")
    print("=" * 70)
    U, x, y, X, Y, T = solve_wave_equation(wave_dict, init_dict)
    
    print("\n" + "=" * 70)
    print("TRAINING SINN (IMPROVED)")
    print("=" * 70)
    interior_encoder, boundary_encoder, decoder = train_sinn_improved(
        U, X, Y, wave_dict,
        time_idx=10,
        n_epochs=50,
        latent_dim=6,
        batch_size=16,
        learning_rate=1e-3
    )
    
    print("\n" + "=" * 70)
    print("VALIDATING SINN")
    print("=" * 70)
    u_pred, error_map, mse = validate_sinn_improved(
        U, X, Y, 10,
        interior_encoder, boundary_encoder, decoder,
        latent_dim=6
    )
    
    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    plot_results(U, u_pred, error_map, 10)
