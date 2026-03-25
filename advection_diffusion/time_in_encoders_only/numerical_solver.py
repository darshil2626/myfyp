import numpy as np
import matplotlib.pyplot as plt

def make_grid(Lx, Ly, nx, ny):
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return x, y, X, Y

def lowpass_random_field(nx, ny, Lx, Ly, ell=0.25, nu=2.5, seed=0):
    """
    Smooth SST-like random field using a Matérn-ish spectrum.
    Returned field has mean 0 and std ~1.
    """
    rng = np.random.default_rng(seed)

    # Fourier wavenumbers
    kx = 2*np.pi * np.fft.fftfreq(nx, d=Lx/nx)
    ky = 2*np.pi * np.fft.fftfreq(ny, d=Ly/ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K2 = KX**2 + KY**2

    kappa = np.sqrt(2*nu) / max(1e-12, ell)
    A = 1.0 / (K2 + kappa**2) ** (0.5*(nu+1))
    A[0,0] = 0.0

    Z = (rng.normal(size=(ny,nx)) + 1j*rng.normal(size=(ny,nx))) / np.sqrt(2)
    Uhat = A * Z

    u = np.fft.ifft2(Uhat).real
    u = (u - u.mean()) / (u.std() + 1e-12)
    return u

import numpy as np

def _bilinear_periodic(u, x, y, Lx, Ly):
    """
    Sample u(y,x) at continuous (x,y) points with periodic wrap using bilinear interpolation.
    u: (ny,nx)
    x,y: arrays same shape (ny,nx) in [0,Lx), [0,Ly) (not required but assumed periodic)
    """
    ny, nx = u.shape
    dx = Lx / nx
    dy = Ly / ny

    # map to grid indices
    gx = (x / dx) % nx
    gy = (y / dy) % ny

    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    x1 = (x0 + 1) % nx
    y1 = (y0 + 1) % ny

    sx = gx - x0
    sy = gy - y0

    u00 = u[y0, x0]
    u10 = u[y0, x1]
    u01 = u[y1, x0]
    u11 = u[y1, x1]

    return (1-sx)*(1-sy)*u00 + sx*(1-sy)*u10 + (1-sx)*sy*u01 + sx*sy*u11


class AdvDiffSpectralSL2D:
    """
    Periodic 2D advection–diffusion:
        u_t + v·∇u = kappa Δu + forcing
    with:
      - semi-Lagrangian advection (stable)
      - exact spectral diffusion step (stable)
      - optional incompressible velocity from a streamfunction
    """

    def __init__(self, nx=128, ny=128, Lx=1.0, Ly=1.0, kappa=0.01):
        self.nx, self.ny = int(nx), int(ny)
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.kappa = float(kappa)

        self.x = np.linspace(0.0, self.Lx, self.nx, endpoint=False)
        self.y = np.linspace(0.0, self.Ly, self.ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")

        # wavenumbers for diffusion filter
        kx = 2*np.pi * np.fft.fftfreq(self.nx, d=self.Lx/self.nx)
        ky = 2*np.pi * np.fft.fftfreq(self.ny, d=self.Ly/self.ny)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        self.K2 = KX**2 + KY**2

    def _velocity_field(self, t, kind="taylor_green", U0=1.0, omega=0.0):
        """
        Return incompressible velocity (vx, vy) on grid.
        - taylor_green: classic smooth cellular flow
        - omega: optional time modulation (vx,vy multiplied by cos(omega t))
        """
        X, Y = self.X, self.Y
        Lx, Ly = self.Lx, self.Ly

        if kind == "taylor_green":
            # streamfunction psi = (U0/(2π)) sin(2πx/Lx) sin(2πy/Ly)
            # v = ( dpsi/dy, -dpsi/dx )
            psi = (U0/(2*np.pi)) * np.sin(2*np.pi*X/Lx) * np.sin(2*np.pi*Y/Ly)
            vx = (U0) * np.sin(2*np.pi*X/Lx) * np.cos(2*np.pi*Y/Ly)
            vy = -(U0) * np.cos(2*np.pi*X/Lx) * np.sin(2*np.pi*Y/Ly)
        else:
            raise ValueError(f"Unknown velocity kind '{kind}'")

        if omega and omega != 0.0:
            vx = vx * np.cos(omega*t)
            vy = vy * np.cos(omega*t)

        return vx.astype(np.float64), vy.astype(np.float64)

    def evolve(
        self,
        u0,
        nt=100,
        t0=0.0,
        t1=1.0,
        vel_kind="taylor_green",
        U0=1.0,
        vel_omega=0.0,
        forcing=None,
        forcing_amp=0.0,
        forcing_rho=0.99,
        seed=0,
    ):
        """
        u0: (ny,nx) initial condition
        forcing: None or "smooth_random" (expects you to reuse your lowpass_random_field() to build base)
        """
        rng = np.random.default_rng(seed)
        T = np.linspace(float(t0), float(t1), int(nt))
        dt = float(T[1] - T[0])

        u = u0.copy().astype(np.float64)
        snapshots = [u.copy()]

        # Optional smooth forcing in Fourier space (same idea as your current solver)
        Fhat = None
        if forcing == "smooth_random" and forcing_amp > 0:
            # build a smooth random forcing in physical space, then fft it
            from numerical_solver import lowpass_random_field  # or import locally if same file
            base = lowpass_random_field(self.nx, self.ny, self.Lx, self.Ly, ell=0.35, nu=3.0, seed=seed+123)
            Fhat = np.fft.fft2(base)

        # diffusion multiplier for exact step
        diff_mult = np.exp(-self.kappa * self.K2 * dt)

        for k in range(1, len(T)):
            t = float(T[k-1])

            # --- 1) Semi-Lagrangian advection ---
            vx, vy = self._velocity_field(t, kind=vel_kind, U0=U0, omega=vel_omega)

            # backtrace
            Xb = (self.X - dt * vx) % self.Lx
            Yb = (self.Y - dt * vy) % self.Ly

            u_adv = _bilinear_periodic(u, Xb, Yb, self.Lx, self.Ly)

            # --- 2) Exact diffusion step (spectral) ---
            Uhat = np.fft.fft2(u_adv)
            Uhat = diff_mult * Uhat
            u = np.fft.ifft2(Uhat).real

            # --- 3) Forcing (optional, Euler add-on) ---
            if Fhat is not None:
                rho = float(forcing_rho)
                rho = np.clip(rho, 0.0, 0.999999)
                s = np.sqrt(max(0.0, 1 - rho*rho))
                noise = (rng.normal(size=Fhat.shape) + 1j*rng.normal(size=Fhat.shape)) / np.sqrt(2)
                Fhat = rho * Fhat + s * noise

                f = np.fft.ifft2(Fhat).real
                f = f / (f.std() + 1e-12)
                u = u + dt * float(forcing_amp) * f

            snapshots.append(u.copy())

        return T, snapshots

def create_visualisation(X, Y, snapshots, T, n_indices=6, title="Heat-equation SST-like"):
    idxs = np.linspace(0, len(snapshots)-1, int(n_indices)).astype(int)
    fig, axes = plt.subplots(1, len(idxs), figsize=(3.2*len(idxs), 3.2), constrained_layout=True)
    if len(idxs) == 1:
        axes = [axes]

    vmin = min(np.min(snapshots[i]) for i in idxs)
    vmax = max(np.max(snapshots[i]) for i in idxs)

    for ax, i in zip(axes, idxs):
        im = ax.imshow(
            snapshots[i], origin="lower",
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            aspect="auto", vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"t={T[i]:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")

    fig.suptitle(title)
    fig.colorbar(im, ax=axes, shrink=0.8)
    return fig