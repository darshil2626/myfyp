# ============================================================================
# SINN IMPLEMENTATION - IMPROVED VERSION
# With proper elliptic solver, spatial encoding, and better loss balancing
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
    """
    Solve 2D wave equation with damping using finite difference method.
    
    Parameters:
    -----------
    wave_dict : dict
        - 'c': wave speed
        - 'Lt': total simulation time
        - 'Lx', 'Ly': domain dimensions
        - 'dx', 'dy': spatial resolution
        - 'CFL': CFL number (default 0.9)
        - 'bc_type': 'dirichlet', 'neumann', or 'periodic' (default 'dirichlet')
        - 'rel_damping_width': relative damping layer width, 0-1 (default 0.2)
        - 'damping_max': maximum damping coefficient (default auto-computed)
        - 'damping_type': 'linear' or 'quadratic' (default 'quadratic')
    
    init_dict : dict
        - 'x_c', 'y_c': pulse center
        - 'width': pulse width (standard deviation)
        - 'amplitude': pulse amplitude
        - 'du0_dt_fun': initial velocity function (optional, defaults to zero)
        
    Returns:
    --------
    U : ndarray of shape (Nt+1, Nx+1, Ny+1)
        Solution array
    T : ndarray of shape (Nt+1,)
        Time array
    x : ndarray
        x grid points
    y : ndarray
        y grid points
    """
    
    # Extract wave parameters
    c = wave_dict['c']
    Lt = wave_dict['Lt']
    Lx = wave_dict['Lx']
    Ly = wave_dict['Ly']
    dx = wave_dict['dx']
    dy = wave_dict['dy']
    CFL = wave_dict.get('CFL', 0.9)
    bc_type = wave_dict.get('bc_type', 'dirichlet').lower()
    rel_damping_width = wave_dict.get('rel_damping_width', 0.2)
    damping_max = wave_dict.get('damping_max', 0.0)
    damping_type = wave_dict.get('damping_type', 'quadratic')
    
    # Extract initial condition parameters
    x_c = init_dict.get('x_c', 0.5)
    y_c = init_dict.get('y_c', 0.5)
    width = init_dict.get('width', 0.1)
    amplitude = init_dict.get('amplitude', 1.0)
    du0_dt_fun = init_dict.get('du0_dt_fun', None)
    
    # Spatial discretisation
    Nx = int(Lx / dx)
    Ny = int(Ly / dy)
    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Temporal discretisation
    dt = CFL * min(dx, dy) / c / np.sqrt(2)
    Nt = int(Lt / dt)
    T = np.linspace(0, Lt, Nt + 1)
    
    # Auto-compute damping_max if not provided
    if damping_max <= 0.0:
        damping_max = 4.0 * c / min(dx, dy)
    
    # Create damping profile
    sigma = _make_damping_profile(Nx, Ny, rel_damping_width, damping_max, damping_type)
    
    # Initial conditions
    u0 = amplitude * np.exp(-((X - x_c)**2 + (Y - y_c)**2) / (2 * width**2))
    if du0_dt_fun is None:
        du0_dt = np.zeros_like(u0)
    else:
        du0_dt = du0_dt_fun(X, Y)
    
    # Allocate solution array
    U = np.zeros((Nt + 1, Nx + 1, Ny + 1))
    U[0] = u0
    
    # Pre-factors
    alpha = (c * dt) ** 2
    sig_in = sigma[1:-1, 1:-1]
    s = 0.5 * sig_in * dt
    
    # First time step
    Lap0 = (
        (U[0, 2:, 1:-1] - 2 * U[0, 1:-1, 1:-1] + U[0, :-2, 1:-1]) / dx ** 2
        + (U[0, 1:-1, 2:] - 2 * U[0, 1:-1, 1:-1] + U[0, 1:-1, :-2]) / dy ** 2
    )
    
    U[1, 1:-1, 1:-1] = (
        U[0, 1:-1, 1:-1]
        + dt * du0_dt[1:-1, 1:-1]
        + 0.5 * alpha * Lap0
        - 0.5 * sigma[1:-1, 1:-1] * dt**2 * du0_dt[1:-1, 1:-1]
    )
    U[1] = _apply_bc(U[1], bc_type)
    
    # Time-stepping loop
    for n in range(1, Nt):
        # Laplacian
        Lap = (
            (U[n, 2:, 1:-1] - 2 * U[n, 1:-1, 1:-1] + U[n, :-2, 1:-1]) / dx ** 2
            + (U[n, 1:-1, 2:] - 2 * U[n, 1:-1, 1:-1] + U[n, 1:-1, :-2]) / dy ** 2
        )
        
        # Time step
        numer = (
            2 * U[n, 1:-1, 1:-1]
            - (1.0 - s) * U[n - 1, 1:-1, 1:-1]
            + alpha * Lap
        )
        denom = 1.0 + s
        U[n + 1, 1:-1, 1:-1] = numer / denom
        
        # Apply boundary condition
        U[n + 1] = _apply_bc(U[n + 1], bc_type)
    
    return U, T, x, y


def _make_damping_profile(Nx, Ny, rel_damping_width, damping_max, damping_type):
    """
    Create 2D damping profile.
    
    Parameters:
    -----------
    Nx, Ny : int
        Number of spatial points
    rel_damping_width : float
        Relative width of damping layer (0-1)
    damping_max : float
        Maximum damping coefficient
    damping_type : str
        'linear' or 'quadratic'
        
    Returns:
    --------
    sigma : ndarray of shape (Nx+1, Ny+1)
        Damping profile
    """
    
    # If no damping requested
    if rel_damping_width <= 0.0:
        return np.zeros((Nx + 1, Ny + 1))
    
    # If full uniform damping
    if rel_damping_width >= 0.5:
        return damping_max * np.ones((Nx + 1, Ny + 1))
    
    # Create 1D profiles
    sx = np.zeros(Nx + 1)
    sy = np.zeros(Ny + 1)
    
    # Damping in x direction
    width_x = int(rel_damping_width * Nx)
    for i in range(width_x):
        if damping_type == 'quadratic':
            coef = ((width_x - i) / width_x) ** 2
        else:  # linear
            coef = (width_x - i) / width_x
        sx[i] = damping_max * coef
        sx[-1 - i] = damping_max * coef
    
    # Damping in y direction
    width_y = int(rel_damping_width * Ny)
    for i in range(width_y):
        if damping_type == 'quadratic':
            coef = ((width_y - i) / width_y) ** 2
        else:  # linear
            coef = (width_y - i) / width_y
        sy[i] = damping_max * coef
        sy[-1 - i] = damping_max * coef
    
    # Combine into 2D profile (sum ensures corners have strong damping)
    sigma = sx[:, None] + sy[None, :]
    
    return sigma


def _apply_bc(U, bc_type):
    """
    Apply boundary conditions to solution array.
    
    Parameters:
    -----------
    U : ndarray of shape (Nx+1, Ny+1)
        Solution array
    bc_type : str
        'dirichlet', 'neumann', or 'periodic'
        
    Returns:
    --------
    U : ndarray
        Solution with BCs applied
    """
    
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
    
    elif bc_type == 'periodic':
        U[0, :] = U[-2, :]
        U[-1, :] = U[1, :]
        U[:, 0] = U[:, -2]
        U[:, -1] = U[:, 1]
    
    return U

wave_dict = {
    'c': 1.0,
    'Lt': 1.0,
    'Lx': 2.0,           # Extended domain
    'Ly': 2.0,
    'dx': 0.01,
    'dy': 0.01,
    'CFL': 0.9,
    'bc_type': 'dirichlet',
    'rel_damping_width': 0.2,   # 20% damping layer
    'damping_type': 'quadratic',
    'damping_max': 1.5
}

init_dict = {
    'x_c': 0.9,
    'y_c': 1,
    'width': np.sqrt(0.005),
    'amplitude': 1
}

# Solve
U, T, x, y = solve_wave_equation(wave_dict, init_dict)

x_train_range=(0.75, 1.25)
y_train_range=(0.75, 1.25)
# Find indices for training domain
i_min = np.argmin(np.abs(x - x_train_range[0]))
i_max = np.argmin(np.abs(x - x_train_range[1])) + 1
j_min = np.argmin(np.abs(y - y_train_range[0]))
j_max = np.argmin(np.abs(y - y_train_range[1])) + 1

# Extract training domain for ALL times
U = U[:, i_min:i_max, j_min:j_max]

# Extract training domain coordinates
x = x[i_min:i_max]
y = y[j_min:j_max]

# Create meshgrid for TRAINING domain
X, Y = np.meshgrid(x, y, indexing='ij')

# ============================================================================
# SECTION 2: MASKING
# ============================================================================

def screenshot_and_normalise(U, time_idx):
    """Standardize (z-score) to center at 0 with std=1"""
    u_screenshot = np.array(U[time_idx], dtype=np.float32)
    
    u_mean = np.mean(u_screenshot)
    u_std = np.std(u_screenshot) + 1e-8
    u_norm = (u_screenshot - u_mean) / u_std
    
    # Optional: clip to [-3, 3] for stability
    u_norm = np.clip(u_norm, -3, 3)
    
    return u_norm, u_mean, u_std

def split_interior_boundary(u_field, X, Y):
    """Extract interior and boundary data at a specific time step"""
    
    # Interior
    u_interior = u_field[1:-1, 1:-1].flatten()
    x_interior = X[1:-1, 1:-1].flatten()
    y_interior = Y[1:-1, 1:-1].flatten()
    
    # Boundary (ordered)
    u_boundary = []
    x_boundary = []
    y_boundary = []
    
    # Top edge (skip corners)
    for j in range(Y.shape[1]):
        u_boundary.append(u_field[0, j])
        x_boundary.append(X[0, j])
        y_boundary.append(Y[0, j])
    
    # Bottom edge (skip corners)
    for j in range(Y.shape[1]):
        u_boundary.append(u_field[-1, j])
        x_boundary.append(X[-1, j])
        y_boundary.append(Y[-1, j])
    
    # Left edge (skip corners)
    for i in range(1, X.shape[0] - 1):
        u_boundary.append(u_field[i, 0])
        x_boundary.append(X[i, 0])
        y_boundary.append(Y[i, 0])
    
    # Right edge (skip corners)
    for i in range(1, X.shape[0] - 1):
        u_boundary.append(u_field[i, -1])
        x_boundary.append(X[i, -1])
        y_boundary.append(Y[i, -1])
    
    u_boundary = np.array(u_boundary)
    x_boundary = np.array(x_boundary)
    y_boundary = np.array(y_boundary)
    
    return x_interior, y_interior, u_interior, x_boundary, y_boundary, u_boundary

time_idx = 30
u_field, u_mean, u_std = screenshot_and_normalise(U, time_idx)
x_interior, y_interior, u_interior, x_boundary, y_boundary, u_boundary = split_interior_boundary(u_field, X, Y)

'''
def interior_mask(u_interior, num_interior_mask_points):
    """create a mask by picking the 8 closest points to the sample point and take an average"""
    rand_x = np.random.randint(1, u_interior.shape[0]-1, size=num_interior_mask_points)
    rand_y = np.random.randint(1, u_interior.shape[1]-1, size=num_interior_mask_points)
    averages = []
    
    for i, j in zip(rand_x, rand_y):
        patch = u_interior[i-1:i+2, j-1:j+2] # 3 x 3 patch
        averages.append(patch.mean())
    return np.array(averages)
    
def boundary_mask(u_interior, u_boundary, num_boundary_mask_points):
    """take the average of neighbouring points in the boundary, use the interior points too for at least the corner"""
'''

# ============================================================================
# SECTION 3: ENCODER
# ============================================================================

def make_interior_encoder(num_latentdim, num_units):
    """Interior encoder: [x, y, u] → ℓ"""
    inputs = Input(shape=(3,), name='interior_input') # shape is 3 to have one dim each for x, y, u
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(inputs)
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.2)(x)
    outputs = Dense(num_latentdim, activation=None)(x)
    return Model(inputs, outputs, name='interior_encoder')

interior_feats = np.stack([x_interior, y_interior, u_interior], axis = -1)
interior_feats_tf = tf.constant(interior_feats, dtype=tf.float32)
interior_encoder = make_interior_encoder(num_latentdim=8, num_units=128)

def make_boundary_encoder(num_latentdim, num_units):
    """Boundary encoder: [x, y, u] → ℓ"""
    inputs = Input(shape=(3,), name='boundary_input') # shape is 3 to have one dim each for x, y, u
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(inputs)
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.2)(x)
    outputs = Dense(num_latentdim, activation=None)(x)
    return Model(inputs, outputs, name='boundary_encoder')

boundary_feats = np.stack([x_boundary, y_boundary, u_boundary], axis = -1)
boundary_feats_tf = tf.constant(boundary_feats, dtype=tf.float32)
boundary_encoder = make_boundary_encoder(num_latentdim=8, num_units=128)

# ============================================================================
# SECTION 4: PDE SOLVER - CURRENTLY NOT USED (PDE SOLVED IN LOSS FUNC)
# ============================================================================

def pde_solver(boundary_latent, A_matrix, patch_size=5, n_iters=20):
    """
    Solve generalized Laplacian: ∑ᵢⱼ A[i,j] ∂²ℓ/∂xᵢ∂xⱼ = 0
    """
    latent_dim = len(boundary_latent)
    center_latent = np.zeros(latent_dim, dtype=np.float32)
    
    # Use MUCH fewer iterations (convergence isn't needed perfectly)
    for dim in range(latent_dim):
        patch = np.ones((patch_size, patch_size), dtype=np.float32) * boundary_latent[dim]
        
        # Simple Jacobi: only a few iterations
        for _ in range(n_iters):
            interior = patch[1:-1, 1:-1]
            patch[1:-1, 1:-1] = (
                patch[:-2, 1:-1] + patch[2:, 1:-1] +
                patch[1:-1, :-2] + patch[1:-1, 2:]
            ) / 4.0
        
        center_latent[dim] = patch[patch_size // 2, patch_size // 2]
    
    return center_latent

# ============================================================================
# SECTION 5: DECODER
# ============================================================================

def make_decoder(num_latentdim, num_units):
    """Decoder: [x, y, ℓ] → u"""
    inputs = Input(shape=(num_latentdim + 2,), name='decoder_input') # shape is +2 to have the extra dim for x and y
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(inputs)
    x = Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.2)(x)
    outputs = Dense(1, activation=None)(x) # output shape is 1 for just u
    return Model(inputs, outputs, name='decoder')

decoder = make_decoder(num_latentdim=8, num_units=128)

# ============================================================================
# SECTION 6: LOSS FUNCTION
# ============================================================================

def compute_loss(interior_latent_true, boundary_latent, decoder, 
                                       interior_feats, A_matrix,
                                       latent_dim=4, alpha=1.0):
    """Loss that actually uses and learns A_matrix"""
    
    batch_size = tf.shape(interior_feats)[0]
    boundary_latent_mean = tf.reduce_mean(boundary_latent, axis=0)  # (latent_dim,)
    
    # Build generalized Laplacian with A_matrix (weighted per dimension)
    # For simplicity: diagonal elements of A weight each latent dimension
    A_diag = tf.linalg.diag_part(A_matrix)  # (latent_dim,)
    
    # Solve PDE using A_matrix weights
    ell_pred_list = []
    for dim in range(latent_dim):
        # Extract this dimension's boundary value
        bc_val = boundary_latent_mean[dim]
        
        # Weight the diffusion by A[dim, dim]
        weight = A_diag[dim]
        
        # Simple iterative solve with weighted Laplacian
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
        
        ell_pred_list.append(patch[2, 2])  # center
    
    ell_pred = tf.stack(ell_pred_list, axis=0)
    ell_pred = tf.tile(tf.expand_dims(ell_pred, 0), [batch_size, 1])
    
    # ===== LOSS =====
    loss_latent = tf.reduce_mean(tf.square(interior_latent_true - ell_pred))
    
    spatial_coords = interior_feats[:, :2]
    decoder_input = tf.concat([spatial_coords, ell_pred], axis=1)
    u_pred = decoder(decoder_input, training=True)
    u_true = tf.expand_dims(interior_feats[:, 2], axis=1)
    loss_physical = tf.reduce_mean(tf.square(u_true - u_pred))
    
    total_loss = loss_latent + alpha * loss_physical
    
    return total_loss, loss_latent, loss_physical


# ============================================================================
# SECTION 7: TRAINING
# ============================================================================

def train(interior_encoder, boundary_encoder, decoder, 
              interior_feats_tf, boundary_feats_tf,
              num_epochs=100, num_latentdim=4):
    """Fixed training loop"""
    
    # Initialize A_matrix
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
            
            # Fast loss computation
            total_loss, loss_lat, loss_phy = compute_loss(
                interior_latent, boundary_latent, decoder,
                interior_feats_tf, A_matrix, latent_dim=num_latentdim, alpha=2.0
            )
        
        # Backprop
        grads = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        
        loss_history.append(float(total_loss))
        
        if (epoch + 1) % (num_epochs / 10) == 0:
            print(f"Epoch {epoch+1}: Loss={total_loss:.6f}, Ψ₁={loss_lat:.6f}, Ψ₂={loss_phy:.6f}")
    
    return interior_encoder, boundary_encoder, decoder, A_matrix, loss_history


interior_encoder, boundary_encoder, decoder, A_matrix, loss_history = train(interior_encoder, boundary_encoder, decoder, interior_feats_tf, boundary_feats_tf, num_epochs=5000, num_latentdim=8)

# ============================================================================
# SECTION 8: VALIDATION
# ============================================================================

def validate(boundary_encoder, decoder, A_matrix, boundary_feats_tf, 
             x_boundary, y_boundary, u_mean, u_std, X, Y, U_true, 
             x, y,  # ADD: 1D coordinate arrays
             time_idx, n_iters):
    """Use actual boundary latent values at each location"""
    
    Nx, Ny = X.shape[0] - 1, X.shape[1] - 1
    latent_dim = A_matrix.shape[0]
    
    # Encode boundary
    boundary_latent = boundary_encoder(boundary_feats_tf, training=False).numpy()
    
    # Create mapping from (x,y) to latent code
    boundary_dict = {}
    for idx, (xb, yb) in enumerate(zip(x_boundary, y_boundary)):
        # Find indices using 1D arrays (not scaling!)
        xi = np.argmin(np.abs(x - xb))
        yi = np.argmin(np.abs(y - yb))
        
        key = (xi, yi)
        if key not in boundary_dict:
            boundary_dict[key] = boundary_latent[idx]
    
    # Solve PDE with spatially-varying boundary
    latent_field = np.zeros((Nx + 1, Ny + 1, latent_dim), dtype=np.float32)
    A_matrix_np = A_matrix.numpy()
    
    for dim in range(latent_dim):
        print(f"  Latent dimension {dim+1}/{latent_dim}...")
        
        ell_field = np.zeros((Nx + 1, Ny + 1), dtype=np.float32)
        boundary_mask = np.zeros((Nx + 1, Ny + 1), dtype=bool)
        
        # Set boundary values from the mapping
        for (xi, yi), latent_code in boundary_dict.items():
            if 0 <= xi < Nx+1 and 0 <= yi < Ny+1:
                ell_field[xi, yi] = latent_code[dim]
                boundary_mask[xi, yi] = True
        
        # Fill remaining edges with interpolation or nearest neighbor
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                if i == 0 or i == Nx or j == 0 or j == Ny:
                    boundary_mask[i, j] = True
                    if ell_field[i, j] == 0:  # Not set yet
                        # Find nearest boundary point
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
            
            if (iteration + 1) % 20 == 0:
                diff = np.max(np.abs(ell_field - ell_old))
                print(f"    Iteration {iteration+1}: max diff = {diff:.2e}")
        
        latent_field[:, :, dim] = ell_field
    
    # ===== DECODE LATENT FIELD =====
    print(f"\nDecoding latent field to physical space...")
    u_pred = np.zeros((Nx + 1, Ny + 1), dtype=np.float32)
    
    for i in range(Nx + 1):
        if (i + 1) % 5 == 0:
            print(f"  Row {i+1}/{Nx+1}")
        
        for j in range(Ny + 1):
            latent_code = latent_field[i, j, :]
            spatial_input = np.array([X[i, j], Y[i, j]], dtype=np.float32)
            decoder_input = np.concatenate([spatial_input, latent_code])
            decoder_input_tf = tf.constant(decoder_input[np.newaxis, :], dtype=tf.float32)
            
            u_pred_norm = decoder(decoder_input_tf, training=False).numpy().squeeze()
            u_pred[i, j] = float(u_pred_norm) * u_std + u_mean
    
    # ===== COMPUTE METRICS =====
    print(f"\nComputing validation metrics...")
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
    
    print(f"\n{'─'*70}")
    print(f"Validation Metrics (with learned A_matrix):")
    print(f"{'─'*70}")
    print(f"  MSE:                   {mse:.6e}")
    print(f"  MAE:                   {mae:.6e}")
    print(f"  RMSE:                  {rmse:.6e}")
    print(f"  Max Error:             {max_error:.6e}")
    print(f"  L² Norm:               {l2_norm:.6e}")
    print(f"  L∞ Norm:               {l_inf_norm:.6e}")
    print(f"  Relative L² Error:     {rel_l2_error:.6e}")
    print(f"  Mean (True / Pred):    {np.mean(u_true):.6f} / {np.mean(u_pred):.6f}")
    print(f"{'─'*70}\n")
    
    return u_pred, latent_field, error_map, metrics

u_pred, latent_field, error_map, metrics = validate(
    boundary_encoder, decoder, A_matrix,
    boundary_feats_tf, x_boundary, y_boundary, u_mean, u_std,
    X, Y, U, 
    x, y,  # ADD THIS - pass 1D coordinate arrays
    time_idx=time_idx, n_iters=100
)

# ============================================================================
# SECTION 8: RESULTS
# ============================================================================

def plot_validation_results(U_true, u_pred, error_map, time_idx, save_path=None):
    """
    Plot comparison: True vs Predicted vs Error
    """
    
    u_true = U_true[time_idx]
    vmax = np.max(np.abs(u_true))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # True solution
    im0 = axes[0].imshow(u_true, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0].set_title(f'True Solution (t_idx={time_idx})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('y')
    axes[0].set_ylabel('x')
    plt.colorbar(im0, ax=axes[0])
    
    vmax = np.max(np.abs(u_pred))
    
    # Predicted solution
    im1 = axes[1].imshow(u_pred, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('SINN Prediction', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('x')
    plt.colorbar(im1, ax=axes[1])
    
    # Error map
    im2 = axes[2].imshow(error_map, origin='lower', cmap='hot_r')
    axes[2].set_title('Absolute Error', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('x')
    cbar = plt.colorbar(im2, ax=axes[2])
    cbar.set_label('|Error|')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

plot_validation_results(U, u_pred, error_map, time_idx=time_idx)

def plot_training_history(loss_history):
    """
    Plot training loss over epochs
    """
    
    epochs = np.arange(1, len(loss_history) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(epochs, loss_history, 'b-', linewidth=2, label='Total Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss History', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_training_history(loss_history)

def plot_error_distribution(error_map):
    """
    Histogram of prediction errors
    """
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(error_map.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Absolute Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.axvline(np.mean(error_map), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(error_map):.2e}')
    ax.axvline(np.median(error_map), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(error_map):.2e}')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
plot_error_distribution(error_map)
    
a = 0