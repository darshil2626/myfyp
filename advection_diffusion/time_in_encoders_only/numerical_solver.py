import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

class AdvectionDiffusion2D:
    """
    RK4 Solver for 2D advection-diffusion equation:
    ∂u/∂t + vx*∂u/∂x + vy*∂u/∂y = D*(∂²u/∂x² + ∂²u/∂y²)
    
    Uses Runge-Kutta 4th order for time integration (much more accurate than Forward Euler)
    """
    
    def __init__(self, Lx=10.0, Ly=10.0, nx=200, ny=200, D=0.1, cfl=0.5):
        """
        Initialize the RK4 solver
        
        Parameters:
        -----------
        Lx, Ly : float
            Domain size in x and y directions
        nx, ny : int
            Number of grid points in x and y directions
        D : float
            Diffusion coefficient
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.D = D
        self.cfl = cfl
        
        # Create grid
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize solution array
        self.u = np.zeros((ny, nx))
        
        # Velocity field (will be set by user)
        self.vx = np.zeros((ny, nx))
        self.vy = np.zeros((ny, nx))
        
        # Time stepping parameters
        self.dt = None
        
    def set_timestep(self):
        """
        Set timestep based on CFL condition
        RK4 has better stability, so we can use a larger CFL factor
        """
        vmax = max(np.max(np.abs(self.vx)), np.max(np.abs(self.vy)), 1e-10)
        dt_advection = self.cfl * min(self.dx, self.dy) / vmax
        dt_diffusion = 0.25 * min(self.dx**2, self.dy**2) / (2 * self.D)
        
        self.dt = min(dt_advection, dt_diffusion)
        print(f"Time step: dt = {self.dt:.6f}")
        
    def add_gaussian_pulse(self, x0, y0, sigma=0.5, amplitude=1.0):
        """Add a Gaussian pulse to the solution"""
        gaussian = amplitude * np.exp(-((self.X - x0)**2 + (self.Y - y0)**2) / (2 * sigma**2))
        self.u += gaussian
        
    def set_velocity_field(self, vx, vy):
        """Set the velocity field"""
        if callable(vx):
            self.vx = vx(self.X, self.Y)
        else:
            self.vx = vx * np.ones_like(self.X)
            
        if callable(vy):
            self.vy = vy(self.X, self.Y)
        else:
            self.vy = vy * np.ones_like(self.Y)
            
        self.set_timestep()
        
    def check_divergence(self):
        """Check if velocity field satisfies continuity"""
        dvx_dx = np.gradient(self.vx, self.dx, axis=1)
        dvy_dy = np.gradient(self.vy, self.dy, axis=0)
        div = dvx_dx + dvy_dy
        max_div = np.max(np.abs(div))
        print(f"Maximum divergence: {max_div:.6f}")
        if max_div > 1e-10:
            print("FLOW IS NOT INCOMPRESSIBLE!")
        return max_div
    
    def compute_rhs(self, u):
        """
        Compute the right-hand side: du/dt = f(u)
        
        This includes:
        - Advection: -vx*∂u/∂x - vy*∂u/∂y (using upwind scheme)
        - Diffusion: D*(∂²u/∂x² + ∂²u/∂y²) (using central differences)
        
        Parameters:
        -----------
        u : array
            Current concentration field
            
        Returns:
        --------
        dudt : array
            Time derivative at each point
        """
        dudt = np.zeros_like(u)
        
        # Loop over interior points (boundaries handled separately)
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                
                # ========== ADVECTION (Upwind Scheme) ==========
                # X-direction
                if self.vx[i, j] > 0:  # Flow to the right
                    dudx = (u[i, j] - u[i, j-1]) / self.dx  # Look left (upwind)
                else:  # Flow to the left
                    dudx = (u[i, j+1] - u[i, j]) / self.dx  # Look right (upwind)
                
                # Y-direction
                if self.vy[i, j] > 0:  # Flow upward
                    dudy = (u[i, j] - u[i-1, j]) / self.dy  # Look down (upwind)
                else:  # Flow downward
                    dudy = (u[i+1, j] - u[i, j]) / self.dy  # Look up (upwind)
                
                advection = -(self.vx[i, j] * dudx + self.vy[i, j] * dudy)
                
                # ========== DIFFUSION (Central Differences) ==========
                d2udx2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (self.dx**2)
                d2udy2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (self.dy**2)
                
                diffusion = self.D * (d2udx2 + d2udy2)
                
                # ========== COMBINED ==========
                dudt[i, j] = advection + diffusion
        
        # Apply boundary conditions to dudt
        dudt = self.apply_bc(dudt)
        
        return dudt
    
    def apply_bc(self, u):
        """
        Apply zero-flux boundary conditions
        
        This is crucial for RK4 because we compute intermediate states
        that also need to satisfy boundary conditions
        """
        u_bc = u.copy()
        u_bc[0, :] = u_bc[1, :]      # Bottom
        u_bc[-1, :] = u_bc[-2, :]    # Top
        u_bc[:, 0] = u_bc[:, 1]      # Left
        u_bc[:, -1] = u_bc[:, -2]    # Right
        return u_bc
    
    def step(self):
        """
        Perform one RK4 time step: uⁿ → uⁿ⁺¹
        
        RK4 Algorithm:
        k₁ = f(uⁿ)
        k₂ = f(uⁿ + dt*k₁/2)
        k₃ = f(uⁿ + dt*k₂/2)
        k₄ = f(uⁿ + dt*k₃)
        uⁿ⁺¹ = uⁿ + (dt/6)(k₁ + 2k₂ + 2k₃ + k₄)
        """
        u_n = self.u.copy()
        dt = self.dt
        
        # Stage 1: Evaluate at current state
        k1 = self.compute_rhs(u_n)
        
        # Stage 2: Evaluate at midpoint using k1
        u_temp = u_n + 0.5 * dt * k1
        u_temp = self.apply_bc(u_temp)
        k2 = self.compute_rhs(u_temp)
        
        # Stage 3: Evaluate at midpoint using k2 (better estimate)
        u_temp = u_n + 0.5 * dt * k2
        u_temp = self.apply_bc(u_temp)
        k3 = self.compute_rhs(u_temp)
        
        # Stage 4: Evaluate at endpoint using k3
        u_temp = u_n + dt * k3
        u_temp = self.apply_bc(u_temp)
        k4 = self.compute_rhs(u_temp)
        
        # Final update: weighted average of all slopes
        # Midpoints get double weight
        self.u = u_n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Apply boundary conditions to final state
        self.u = self.apply_bc(self.u)
        
    def solve(self, t_final):
        """
        Solve the equation up to time t_final using RK4
        
        Parameters:
        -----------
        t_final : float
            Final time
        n_snapshots : int
            Number of snapshots to save
            
        Returns:
        --------
        times : array
            Times at which snapshots were saved
        snapshots : list of arrays
            Solution snapshots
        """
        n_steps = int(t_final / self.dt)
        times = []
        snapshots = []
        
        print(f"\nStarting RK4 simulation...")
        print(f"Total steps: {n_steps}")
        
        for step in range(n_steps):
            times.append(step * self.dt)
            snapshots.append(self.u.copy())
            self.step()
        
        # Save final snapshot
        times.append(n_steps * self.dt)
        snapshots.append(self.u.copy())
        
        return np.array(times), snapshots


def create_velocity_field_uniform(X, Y, v_right=0.6, v_down=-0.15):
    """
    Create uniform incompressible velocity field
    """
    vx = v_right * np.ones_like(X)
    vy = v_down * np.ones_like(Y)
    return vx, vy

def create_visualisation(X, Y, snapshots, times, n_indices, vmax):
    """
    Create a grid of filled contour plots showing solution evolution
    with a unified colorbar spanning all plots
    
    Parameters:
    -----------
    X, Y : 2D arrays
        Spatial grid
    snapshots : list of 2D arrays
        Solution fields at different times
    times : 1D array
        Time values
    n_indices : int
        Number of snapshots to display
    vmax : float
        Maximum value for colorbar scale
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
                    cmap='hot', vmin=0, vmax=vmax)
        ax.contour(X, Y, snapshots[idx], levels=8,
                   colors='black', linewidths=0.5, alpha=0.3)
        ax.set_title(f't = {times[idx]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    # Create colorbar with full range [0, vmax]
    cax = fig.add_subplot(gs[:, -1])
    sm = ScalarMappable(cmap='hot', norm=Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    cax.set_ylabel('u(x,y,t)', rotation=270, labelpad=15)

    fig.suptitle('Evolution with RK4 Time Integration\n(4th order accurate in time)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('advection_diffusion_rk4_evolution.png', dpi=150, bbox_inches='tight')
    print("✓ Evolution figure saved!")



def crop_data(X, Y, snapshots, margin_x, margin_y):
    """
    Crop solution snapshots by removing margins from all sides
    
    Parameters:
    -----------
    X, Y : 2D arrays
        Spatial grid
    snapshots : list of 2D arrays
        Solution fields to crop
    margin_x, margin_y : float
        Physical distance to remove from each side in x and y
        
    Returns:
    --------
    X_crop, Y_crop : 2D arrays
        Cropped grids
    snapshots_crop : list of 2D arrays
        Cropped snapshots
    """
    # Find indices to keep
    ix_keep = np.where((X[0, :] >= margin_x) & (X[0, :] <= X[0, -1] - margin_x))[0]
    iy_keep = np.where((Y[:, 0] >= margin_y) & (Y[:, 0] <= Y[-1, 0] - margin_y))[0]
    
    X_crop = X[np.ix_(iy_keep, ix_keep)]
    Y_crop = Y[np.ix_(iy_keep, ix_keep)]
    snapshots_crop = [s[np.ix_(iy_keep, ix_keep)] for s in snapshots]
    
    return X_crop, Y_crop, snapshots_crop



if __name__ == "__main__":    
    # Initialize solver
    print("\nInitializing solver...")
    solver = AdvectionDiffusion2D(Lx=12.0, Ly=8.0, nx=150, ny=120, D=0.08, cfl=0.5)
    
    # Add two Gaussian pulses
    print("Adding two Gaussian pulses...")
    solver.add_gaussian_pulse(x0=2.0, y0=3.0, sigma=0.4, amplitude=1.0)
    solver.add_gaussian_pulse(x0=3.0, y0=5.0, sigma=0.4, amplitude=1.0)
    
    # Set uniform velocity field (incompressible)
    print("Setting up velocity field...")
    vx, vy = create_velocity_field_uniform(solver.X, solver.Y, v_right=0.6, v_down=-0.15)
    solver.set_velocity_field(vx, vy)
    
    # Check divergence
    print("\nChecking incompressibility...")
    div = solver.check_divergence()
    if div < 1e-10:
        print("✓ Flow is incompressible (∇·v ≈ 0)")
    
    # Solve
    times, snapshots = solver.solve(t_final=5.0)
    
    vmax = max([s.max() for s in snapshots])
    create_visualisation(solver.X, solver.Y, snapshots, times, 10, vmax)