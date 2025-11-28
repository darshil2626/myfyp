# NUMERICAL SOLUTION WITH ABSORBING BC

import numpy as np

def solve_wave_equation(wave_dict, init_dict):
    """
    Solves the 2D wave equation using finite difference method (FDTD)
    
    The wave equation is solved as: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
    
    Parameters:
    -----------
    wave_dict : dict
        'c': wave speed (float)
        'Lt': total time duration (float)
        'Lx': domain length in x direction (float)
        'Ly': domain length in y direction (float)
        'dx': spatial grid spacing in x (float)
        'dy': spatial grid spacing in y (float)
        'CFL': Courant-Friedrichs-Lewy stability number (float, typically 0.9)
    
    init_dict : dict
        'x_c': x-center of gaussian pulse (float)
        'y_c': y-center of gaussian pulse (float)
        'width': width parameter of gaussian pulse (float)
        'amplitude': amplitude of gaussian pulse (float)
        'pulse_period': number of time steps between pulse sources (int)
    
    Returns:
    --------
    U : ndarray of shape (Nt, Nx, Ny)
        Solution array containing displacement at each time step and spatial location
    t : ndarray of shape (Nt,)
        Time points
    x : ndarray of shape (Nx,)
        X spatial grid points
    y : ndarray of shape (Ny,)
        Y spatial grid points
    
    Notes:
    ------
    - Boundary conditions: Dirichlet (u = 0 at boundaries)
    - Initial velocity: assumed to be zero
    - Stability is ensured by CFL condition: dt <= CFL / (c * sqrt(1/dx² + 1/dy²))
    """
    
    # Extract wave parameters
    c = wave_dict['c']
    Lt = wave_dict['Lt']
    Lx = wave_dict['Lx']
    Ly = wave_dict['Ly']
    dx = wave_dict['dx']
    dy = wave_dict['dy']
    CFL = wave_dict['CFL']
    
    # Extract initialization parameters
    x_c = init_dict['x_c']
    y_c = init_dict['y_c']
    width = init_dict['width']
    amplitude = init_dict['amplitude']
    pulse_period = init_dict['pulse_period']
    
    # Create spatial grids
    x = np.arange(0, Lx + dx, dx)
    y = np.arange(0, Ly + dy, dy)
    Nx = len(x)
    Ny = len(y)
    
    # Calculate time step from CFL condition for 2D stability
    dt = CFL / (c * np.sqrt(1/dx**2 + 1/dy**2))
    t = np.arange(0, Lt + dt, dt)
    Nt = len(t)
    
    # Create 2D meshgrid for spatial coordinates
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Calculate Courant numbers for each direction
    rx = c * dt / dx  
    ry = c * dt / dy
    
    # Print information about the discretization
    print(f"=== Wave Equation 2D Solver ===")
    print(f"Grid dimensions: Nx={Nx}, Ny={Ny}, Nt={Nt}")
    print(f"Spatial resolution: dx={dx}, dy={dy}")
    print(f"Time step: dt={dt:.6f}")
    print(f"Courant numbers: rx={rx:.4f}, ry={ry:.4f}")
    print(f"Stability criterion (rx² + ry²): {rx**2 + ry**2:.4f} <= 1.0")
    
    # Initialize solution array
    U = np.zeros((Nt, Nx, Ny))
    
    # Define gaussian pulse function
    def gaussian_pulse(X, Y, x_c, y_c, width, amplitude):
        """Gaussian pulse centered at (x_c, y_c)"""
        return amplitude * np.exp(-((X - x_c)**2 + (Y - y_c)**2) / (2 * width**2))
    
    # Set initial condition at t=0
    U[0, :, :] = gaussian_pulse(X, Y, x_c, y_c, width, amplitude)
    
    # Set condition at t=dt (assume zero initial velocity)
    U[1, :, :] = U[0, :, :]
    
    # Time stepping loop using finite difference scheme
    for n in range(1, Nt - 1):
        # Add periodic gaussian pulse source every pulse_period steps
        if n % pulse_period == 0:
            source = gaussian_pulse(X, Y, x_c, y_c, width, amplitude)
        else:
            source = np.zeros((Nx, Ny))
        
        # Update interior points using finite difference approximation
        # U(t+dt, x, y) = 2*U(t, x, y) - U(t-dt, x, y) 
        #                + rx²*(U(t, x+dx, y) - 2*U(t, x, y) + U(t, x-dx, y))
        #                + ry²*(U(t, x, y+dy) - 2*U(t, x, y) + U(t, x, y-dy))
        #                + dt²*source(x, y)
        
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                U[n + 1, i, j] = (
                    2 * U[n, i, j] 
                    - U[n - 1, i, j]
                    + rx**2 * (U[n, i + 1, j] - 2 * U[n, i, j] + U[n, i - 1, j])
                    + ry**2 * (U[n, i, j + 1] - 2 * U[n, i, j] + U[n, i, j - 1])
                    + dt**2 * source[i, j]
                )
        
        # Apply Dirichlet boundary conditions (u = 0 at boundaries)
        U[n + 1, 0, :] = 0
        U[n + 1, -1, :] = 0
        U[n + 1, :, 0] = 0
        U[n + 1, :, -1] = 0
    
    return U, t, x, y
