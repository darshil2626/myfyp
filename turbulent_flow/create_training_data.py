"""
Turbulent Flow Training Data Creator

Loads AmiraMesh files containing 2D velocity fields (u, v components)
and creates training datasets in the format expected by SINN models.

Data format: (time, y, x, 2) where 2 = [u_component, v_component]
Output format: X, Y, U, T where U can be speed magnitude or velocity components
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def load_amira_lattice_float2(path, shape=(512, 512, 1001), dtype=np.dtype('<f4')):
    """
    Loads an AmiraMesh Lattice { float[2] Data } @1
    Returns: data shaped (nz, ny, nx, 2) by default (time as z).
    
    Parameters:
    -----------
    path : str
        Path to the .am file
    shape : tuple
        (nx, ny, nz) - spatial dimensions and number of timesteps
    dtype : numpy dtype
        Data type (default: little-endian float32)
    
    Returns:
    --------
    data : np.ndarray
        Array of shape (nz, ny, nx, 2) containing velocity components
    """
    nx, ny, nz = shape  # from "define Lattice 512 512 1001"

    with open(path, "rb") as f:
        raw = f.read()

    # Find the '@1' marker, then skip to the start of binary data after the newline
    marker = raw.find(b"@1")
    if marker == -1:
        raise ValueError("Could not find '@1' data marker in file.")

    # Data starts after '@1' and the following newline(s)
    data_start = marker + 2
    while data_start < len(raw) and raw[data_start] in (ord('\n'), ord('\r'), ord(' '), ord('\t')):
        data_start += 1

    # Interpret the remaining bytes as little-endian float32
    arr = np.frombuffer(raw, dtype=dtype, offset=data_start)

    expected = nx * ny * nz * 2
    if arr.size < expected:
        raise ValueError(
            f"File truncated. Got {arr.size}, expected at least {expected}."
        )

    arr = arr[:expected]  # Ignore trailing padding

    # Amira Lattice typically stores x fastest, then y, then z
    # So reshape as (nz, ny, nx, 2)
    data = arr.reshape((nz, ny, nx, 2))
    return data


def compute_speed_magnitude(u, v):
    """
    Compute speed magnitude from velocity components.
    
    Parameters:
    -----------
    u, v : np.ndarray
        Velocity components (same shape)
    
    Returns:
    --------
    speed : np.ndarray
        Speed magnitude (same shape as u and v)
    """
    return np.sqrt(u**2 + v**2)


def create_spatial_grid(shape):
    """
    Create normalized spatial grids (0 to 1).
    
    Parameters:
    -----------
    shape : tuple
        (ny, nx)
    
    Returns:
    --------
    X, Y : np.ndarray
        2D meshgrids normalized to [0, 1]
    """
    ny, nx = shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def downsample_data(u, v, factor=2):
    """
    Downsample velocity field using stride subsampling.
    
    Parameters:
    -----------
    u, v : np.ndarray
        Velocity components, shape (nt, ny, nx)
    factor : int
        Downsampling factor
    
    Returns:
    --------
    u_ds, v_ds : np.ndarray
        Downsampled velocity fields
    """
    if factor == 1:
        return u, v
    u_ds = u[:, ::factor, ::factor]
    v_ds = v[:, ::factor, ::factor]
    return u_ds, v_ds


def temporal_downsample(data, factor=1):
    """
    Downsample in time dimension.
    
    Parameters:
    -----------
    data : np.ndarray
        Data with time as first dimension
    factor : int
        Temporal downsampling factor
    
    Returns:
    --------
    data_ds : np.ndarray
        Temporally downsampled data
    """
    if factor == 1:
        return data
    return data[::factor, ...]


def clean_velocity_data(u, v, method='percentile_clip'):
    """
    Clean velocity data by handling NaN, Inf, and extreme outliers.
    
    Parameters:
    -----------
    u, v : np.ndarray
        Velocity components
    method : str
        'remove_nan_inf': Replace NaN/Inf with 0
        'percentile_clip': Clip to 99th percentile (best for outliers)
        'std_clip': Clip to mean ± 5*std
        'interpolate_spatial': Replace with spatial mean per timestep
    
    Returns:
    --------
    u_clean, v_clean : np.ndarray
        Cleaned velocity components
    """
    u_clean = u.copy().astype(np.float32)
    v_clean = v.copy().astype(np.float32)
    
    # Count issues
    u_nan = np.isnan(u_clean).sum()
    u_inf = np.isinf(u_clean).sum()
    v_nan = np.isnan(v_clean).sum()
    v_inf = np.isinf(v_clean).sum()
    
    print(f"\nData Quality Check:")
    print(f"  u: {u_nan} NaN, {u_inf} Inf")
    print(f"  v: {v_nan} NaN, {v_inf} Inf")
    print(f"  u range: [{np.nanmin(u_clean):.6e}, {np.nanmax(u_clean):.6e}]")
    print(f"  v range: [{np.nanmin(v_clean):.6e}, {np.nanmax(v_clean):.6e}]")
    
    # Check for extreme values that would overflow
    u_valid = u_clean[~np.isnan(u_clean) & ~np.isinf(u_clean)]
    v_valid = v_clean[~np.isnan(v_clean) & ~np.isinf(v_clean)]
    
    u_extreme = (u_valid > 1e10).sum() + (u_valid < -1e10).sum()
    v_extreme = (v_valid > 1e10).sum() + (v_valid < -1e10).sum()
    
    if u_extreme > 0 or v_extreme > 0:
        print(f"  ⚠ Extreme values detected: {u_extreme} in u, {v_extreme} in v")
    
    if u_nan == 0 and u_inf == 0 and v_nan == 0 and v_inf == 0 and u_extreme == 0 and v_extreme == 0:
        print(f"  ✓ Data is clean!")
        return u_clean, v_clean
    
    if method == 'percentile_clip':
        print(f"  Clipping to 99th percentile...")
        u_p99 = np.percentile(u_valid, 99) if len(u_valid) > 0 else 1e6
        u_p01 = np.percentile(u_valid, 1) if len(u_valid) > 0 else -1e6
        v_p99 = np.percentile(v_valid, 99) if len(v_valid) > 0 else 1e6
        v_p01 = np.percentile(v_valid, 1) if len(v_valid) > 0 else -1e6
        
        print(f"    u: clipping [{u_p01:.6e}, {u_p99:.6e}]")
        print(f"    v: clipping [{v_p01:.6e}, {v_p99:.6e}]")
        
        u_clean = np.clip(u_clean, u_p01, u_p99)
        v_clean = np.clip(v_clean, v_p01, v_p99)
    
    elif method == 'std_clip':
        print(f"  Clipping to mean ± 5*std...")
        u_mean, u_std = np.nanmean(u_valid), np.nanstd(u_valid)
        v_mean, v_std = np.nanmean(v_valid), np.nanstd(v_valid)
        
        u_clean = np.clip(u_clean, u_mean - 5*u_std, u_mean + 5*u_std)
        v_clean = np.clip(v_clean, v_mean - 5*v_std, v_mean + 5*v_std)
    
    elif method == 'remove_nan_inf':
        print(f"  Replacing NaN/Inf with 0...")
        u_clean = np.nan_to_num(u_clean, nan=0.0, posinf=0.0, neginf=0.0)
        v_clean = np.nan_to_num(v_clean, nan=0.0, posinf=0.0, neginf=0.0)
    
    elif method == 'interpolate_spatial':
        print(f"  Replacing with spatial mean per timestep...")
        nt = u_clean.shape[0]
        for t in range(nt):
            u_valid_t = u_clean[t, ~np.isnan(u_clean[t]) & ~np.isinf(u_clean[t])]
            v_valid_t = v_clean[t, ~np.isnan(v_clean[t]) & ~np.isinf(v_clean[t])]
            
            u_mean = np.mean(u_valid_t) if len(u_valid_t) > 0 else 0.0
            v_mean = np.mean(v_valid_t) if len(v_valid_t) > 0 else 0.0
            
            u_clean[t, (np.isnan(u_clean[t]) | np.isinf(u_clean[t]))] = u_mean
            v_clean[t, (np.isnan(v_clean[t]) | np.isinf(v_clean[t]))] = v_mean
    
    # Final check
    u_clean = np.nan_to_num(u_clean, nan=0.0, posinf=0.0, neginf=0.0)
    v_clean = np.nan_to_num(v_clean, nan=0.0, posinf=0.0, neginf=0.0)
    
    u_nan_after = np.isnan(u_clean).sum()
    u_inf_after = np.isinf(u_clean).sum()
    v_nan_after = np.isnan(v_clean).sum()
    v_inf_after = np.isinf(v_clean).sum()
    
    print(f"  After cleaning:")
    print(f"    u: {u_nan_after} NaN, {u_inf_after} Inf, range [{u_clean.min():.6e}, {u_clean.max():.6e}]")
    print(f"    v: {v_nan_after} NaN, {v_inf_after} Inf, range [{v_clean.min():.6e}, {v_clean.max():.6e}]")
    
    return u_clean, v_clean


def create_visualisation(X, Y, snapshots, times, n_indices, vmin, vmax, title_prefix=""):
    """
    Create a grid of filled contour plots showing field evolution
    with a unified colorbar spanning all plots.
    
    Parameters:
    -----------
    X, Y : 2D arrays
        Spatial grid
    snapshots : list of 2D arrays
        Field values at different times
    times : 1D array
        Time values
    n_indices : int
        Number of snapshots to display
    vmin, vmax : float
        Minimum and maximum values for colorbar scale
    title_prefix : str
        Prefix for plot titles
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    indices = np.linspace(0, len(snapshots) - 1, n_indices, dtype=int).tolist()
    ncols = int(np.ceil(len(indices) / 2))

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, ncols + 1, width_ratios=[1]*ncols + [0.05], figure=fig)

    for i, idx in enumerate(indices):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        
        # Create filled contour plot
        ax.contourf(X, Y, snapshots[idx], levels=20,
                    cmap='viridis', vmin=vmin, vmax=vmax)
        ax.contour(X, Y, snapshots[idx], levels=8,
                   colors='black', linewidths=0.5, alpha=0.3)
        ax.set_title(f'{title_prefix} at t={idx}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')

    # Create colorbar with full range [vmin, vmax]
    cax = fig.add_subplot(gs[:, -1])
    sm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    cax.set_ylabel('Magnitude', rotation=270, labelpad=15)

    fig.suptitle(f'{title_prefix} Evolution',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


# =========================================================================
# MAIN SCRIPT
# =========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TURBULENT FLOW TRAINING DATA CREATOR")
    print("=" * 70)
    
    # -------------------------
    # Configuration
    # -------------------------
    AMIRA_FILE = "0000.am"  # Can also use "0001.am" or process both
    SPATIAL_DOWNSAMPLE = 1  # Set to 2 or 4 for faster processing
    TEMPORAL_DOWNSAMPLE = 1  # Set to 2 or 4 to reduce temporal resolution
    FIELD_TYPE = "speed"    # "speed" for magnitude, "uv" for velocity components
    
    print(f"\nConfiguration:")
    print(f"  Input file: {AMIRA_FILE}")
    print(f"  Spatial downsample factor: {SPATIAL_DOWNSAMPLE}")
    print(f"  Temporal downsample factor: {TEMPORAL_DOWNSAMPLE}")
    print(f"  Field type: {FIELD_TYPE}")
    
    # -------------------------
    # Load data
    # -------------------------
    print(f"\nLoading AmiraMesh data from {AMIRA_FILE}...")
    data = load_amira_lattice_float2(AMIRA_FILE, shape=(512, 512, 1001))
    print(f"  Raw data shape: {data.shape}")  # (1001, 512, 512, 2)
    
    # Extract velocity components
    u = data[:, :, :, 0]  # x-component
    v = data[:, :, :, 1]  # y-component
    
    print(f"  u-component shape: {u.shape}")
    print(f"  v-component shape: {v.shape}")
    print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"  v range: [{v.min():.4f}, {v.max():.4f}]")
    
    # -------------------------
    # Clean data (remove NaN/Inf and extreme outliers)
    # -------------------------
    u, v = clean_velocity_data(u, v, method='percentile_clip')
    print(f"  u range after cleaning: [{u.min():.6e}, {u.max():.6e}]")
    print(f"  v range after cleaning: [{v.min():.6e}, {v.max():.6e}]")
    
    # -------------------------
    # Downsample (optional)
    # -------------------------
    if SPATIAL_DOWNSAMPLE > 1 or TEMPORAL_DOWNSAMPLE > 1:
        print(f"\nDownsampling...")
        u, v = downsample_data(u, v, factor=SPATIAL_DOWNSAMPLE)
        u = temporal_downsample(u, factor=TEMPORAL_DOWNSAMPLE)
        v = temporal_downsample(v, factor=TEMPORAL_DOWNSAMPLE)
        print(f"  Downsampled u shape: {u.shape}")
        print(f"  Downsampled v shape: {v.shape}")
    
    # -------------------------
    # Create spatial coordinate grids
    # -------------------------
    print(f"\nCreating spatial grids...")
    nt, ny, nx = u.shape
    X, Y = create_spatial_grid((ny, nx))
    print(f"  Grid shape: {X.shape}")
    print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  Y range: [{Y.min():.2f}, {Y.max():.2f}]")
    
    # -------------------------
    # Create time array (index-based)
    # -------------------------
    T = np.arange(nt, dtype=np.float32)
    print(f"\nTime array:")
    print(f"  Time steps: {nt}")
    print(f"  T shape: {T.shape}")
    
    # -------------------------
    # Create field data
    # -------------------------
    print(f"\nProcessing field data...")
    if FIELD_TYPE == "speed":
        speed = compute_speed_magnitude(u, v)
        U = [speed[t, :, :] for t in range(nt)]
        print(f"  Using speed magnitude")
        print(f"  Speed range: [{min(np.min(s) for s in U):.4f}, {max(np.max(s) for s in U):.4f}]")
    elif FIELD_TYPE == "uv":
        # Stack u and v as separate channels (need to handle in solver)
        U = [np.stack([u[t, :, :], v[t, :, :]], axis=-1) for t in range(nt)]
        print(f"  Using velocity components (u, v)")
    else:
        raise ValueError(f"Unknown field_type: {FIELD_TYPE}")
    
    # -------------------------
    # Statistics
    # -------------------------
    print(f"\nDataset Statistics:")
    print(f"  Total timesteps: {len(U)}")
    print(f"  Spatial resolution: {ny} x {nx}")
    if FIELD_TYPE == "speed":
        all_vals = np.concatenate([U[t].flatten() for t in range(len(U))])
        print(f"  Field min: {np.min(all_vals):.6f}")
        print(f"  Field max: {np.max(all_vals):.6f}")
        print(f"  Field mean: {np.mean(all_vals):.6f}")
        print(f"  Field std: {np.std(all_vals):.6f}")
        
        # Final validation
        n_nan_final = np.isnan(all_vals).sum()
        n_inf_final = np.isinf(all_vals).sum()
        if n_nan_final > 0 or n_inf_final > 0:
            print(f"  [Warning] Final field has {n_nan_final} NaN and {n_inf_final} Inf values!")
            print(f"  Replacing with 0...")
            for t in range(len(U)):
                U[t] = np.nan_to_num(U[t], nan=0.0, posinf=0.0, neginf=0.0)
    
    # -------------------------
    # Visualize
    # -------------------------
    print(f"\nCreating visualization...")
    if FIELD_TYPE == "speed":
        vmin_speed = min(np.min(s) for s in U)
        vmax_speed = max(np.max(s) for s in U)
        fig = create_visualisation(X, Y, U, T, n_indices=8, 
                                   vmin=vmin_speed, vmax=vmax_speed,
                                   title_prefix="Speed Magnitude")
    else:
        # For velocity components, visualize speed
        speed_vis = [np.sqrt(u[t, :, :] ** 2 + v[t, :, :] ** 2) for t in range(nt)]
        vmin_speed = min(np.min(s) for s in speed_vis)
        vmax_speed = max(np.max(s) for s in speed_vis)
        fig = create_visualisation(X, Y, speed_vis, T, n_indices=8,
                                   vmin=vmin_speed, vmax=vmax_speed,
                                   title_prefix="Speed Magnitude (from u,v)")
    plt.savefig("turbulent_flow_visualization.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to turbulent_flow_visualization.png")
    plt.close()
    
    # -------------------------
    # Save training data
    # -------------------------
    print(f"\nSaving training data...")
    
    # Determine output filename
    ds_suffix = f"_ds{SPATIAL_DOWNSAMPLE}" if SPATIAL_DOWNSAMPLE > 1 else ""
    ts_suffix = f"_ts{TEMPORAL_DOWNSAMPLE}" if TEMPORAL_DOWNSAMPLE > 1 else ""
    output_path = f"training_data_{FIELD_TYPE}{ds_suffix}{ts_suffix}.pkl"
    
    data_dict = {
        'X': X,
        'Y': Y,
        'U': U,
        'T': T,
        'metadata': {
            'source_file': AMIRA_FILE,
            'field_type': FIELD_TYPE,
            'spatial_downsample': SPATIAL_DOWNSAMPLE,
            'temporal_downsample': TEMPORAL_DOWNSAMPLE,
            'original_shape': (1001, 512, 512),
            'processed_shape': (nt, ny, nx),
        }
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(data_dict, f)
    
    print(f"  ✓ Training data saved to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    
    # -------------------------
    # Verify saved data
    # -------------------------
    print(f"\nVerifying saved data...")
    with open(output_path, "rb") as f:
        loaded_data = pickle.load(f)
    
    # Check for any remaining NaN/Inf in U
    all_vals_check = np.concatenate([loaded_data['U'][t].flatten() for t in range(len(loaded_data['U']))])
    n_nan_check = np.isnan(all_vals_check).sum()
    n_inf_check = np.isinf(all_vals_check).sum()
    
    print(f"  Loaded data shape: {len(loaded_data['U'])} timesteps, {loaded_data['U'][0].shape} spatial")
    print(f"  NaN values: {n_nan_check}")
    print(f"  Inf values: {n_inf_check}")
    
    if n_nan_check == 0 and n_inf_check == 0:
        print(f"  ✓ Data is clean!")
    
    print("\n" + "=" * 70)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Load data: with open('{output_path}', 'rb') as f: data = pickle.load(f)")
    print(f"  2. Pass to SINN solver: solver = sinn(data['X'], data['Y'], data['U'], data['T'])")
    print(f"  3. Train and evaluate the model")
