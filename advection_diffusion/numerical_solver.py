import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#  Initial condition: two pulses on the left
# ============================================================

def initial_condition(X, Y):
    """
    Two positive Gaussian pulses on the left half of the domain.
    They will be advected rightwards and pulled towards a meeting point.
    """
    g1 = np.exp(-(((X - 0.25)**2 + (Y - 0.25)**2) / 0.09))  # bottom-left
    g2 = np.exp(-(((X - 0.75)**2 + (Y - 0.75)**2) / 0.09))  # top-left
    return g1 + g2


# ============================================================
#  Solver: 2D advection–diffusion with spatially varying v(x,y)
# ============================================================

def solve_advection_diffusion_2d(
    nx=101,
    ny=101,
    Lx=1.0,
    Ly=1.0,
    v_bg=0.3,        # uniform rightward drift
    alpha=1.0,       # strength of convergence towards (x_c, y_c)
    kappa=0.001,     # diffusion coefficient
    T=1.0,           # final time
    dt=None,         # if None, chosen automatically for stability
    save_every=5,    # save every N time steps
):
    """
    Solve u_t + v·∇u = kappa ∆u on [0,Lx]×[0,Ly] with u=0 on boundaries.

    Velocity field:
        v(x,y) = (v_bg, 0) - alpha * (x - x_c, y - y_c)
    where (x_c, y_c) is an off-centre "meeting point".

    Returns
    -------
    x, y : 1D arrays
    t_snap : (nt_snap,) array of times
    U_snap : (nt_snap, ny, nx) array of solution snapshots
    """

    # ---------------- Grid ----------------
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    # ---------------- Velocity field ----------------
    x_c = 0.7 * Lx
    y_c = 0.5 * Ly

    vx = v_bg - alpha * (X - x_c)
    vy =      - alpha * (Y - y_c)

    # ---------------- Time step (stability) ----------------
    vmax_x = np.max(np.abs(vx))
    vmax_y = np.max(np.abs(vy))

    adv_term = vmax_x / dx + vmax_y / dy
    diff_term = 2.0 * kappa * (1.0 / dx**2 + 1.0 / dy**2)

    if dt is None:
        denom = adv_term + diff_term
        if denom == 0.0:
            dt = 0.4 * min(dx, dy)**2 / (4 * kappa + 1e-12)
        else:
            dt = 0.4 / denom

    nt = int(np.ceil(T / dt))
    dt = T / nt  # adjust so that we land exactly on T

    print(f"Using dt = {dt:.3e}, nt = {nt}")

    # ---------------- Initial condition ----------------
    u = initial_condition(X, Y)

    # Dirichlet boundaries u = 0
    u[0, :]  = 0.0
    u[-1, :] = 0.0
    u[:, 0]  = 0.0
    u[:, -1] = 0.0

    # ---------------- Time integration ----------------
    U_list = []
    t_list = []

    # save initial state
    U_list.append(u.copy())
    t_list.append(0.0)

    for n in range(1, nt + 1):
        un = u.copy()

        # interior slices
        u_c = un[1:-1, 1:-1]
        u_l = un[1:-1, 0:-2]
        u_r = un[1:-1, 2:]
        u_d = un[0:-2, 1:-1]
        u_u = un[2:,   1:-1]

        vx_c = vx[1:-1, 1:-1]
        vy_c = vy[1:-1, 1:-1]

        # upwind in x
        du_dx = np.where(
            vx_c >= 0.0,
            (u_c - u_l) / dx,     # wind from left -> backward difference
            (u_r - u_c) / dx,     # wind from right -> forward difference
        )

        # upwind in y
        du_dy = np.where(
            vy_c >= 0.0,
            (u_c - u_d) / dy,     # wind from below
            (u_u - u_c) / dy,     # wind from above
        )

        # diffusion (central differences)
        d2u_dx2 = (u_r - 2.0 * u_c + u_l) / dx**2
        d2u_dy2 = (u_u - 2.0 * u_c + u_d) / dy**2

        # update interior
        u[1:-1, 1:-1] = u_c + dt * (
            -vx_c * du_dx
            -vy_c * du_dy
            + kappa * (d2u_dx2 + d2u_dy2)
        )

        # re-apply Dirichlet BCs
        u[0, :]  = 0.0
        u[-1, :] = 0.0
        u[:, 0]  = 0.0
        u[:, -1] = 0.0

        # save snapshot
        if (n % save_every == 0) or (n == nt):
            U_list.append(u.copy())
            t_list.append(n * dt)

    U_snap = np.stack(U_list, axis=0)   # (nt_snap, ny, nx)
    t_snap = np.array(t_list)           # (nt_snap,)

    return x, y, t_snap, U_snap


# ============================================================
#  Plot snapshots at evenly spaced times with shared colorbar
# ============================================================

def plot_snapshots(x, y, t, U, n_plots=5):
    """
    Plot n_plots snapshots of U with a constant color scale and shared colorbar.
    """
    NT = len(U)
    k_frames = np.linspace(0, NT - 1, n_plots, dtype=int)

    fig, axes = plt.subplots(1, n_plots, figsize=(20, 3), constrained_layout=True)

    # global colour limits
    umin = U.min()
    umax = U.max()

    # if you want symmetric limits around 0, uncomment:
    # m = max(abs(umin), abs(umax))
    # umin, umax = -m, m

    for ax, k in zip(axes, k_frames):
        # use extent so axes are in physical coordinates
        cs = ax.imshow(
            U[k],
            origin='lower',
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect='auto',
            cmap='RdBu_r',
            vmin=umin,
            vmax=umax,
        )
        ax.set_title(f"t = {t[k]:.2f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # one shared colorbar for all subplots
    cbar = fig.colorbar(cs, ax=axes.ravel().tolist(), location='right')
    cbar.set_label("u(x,y,t)")

    fig.suptitle("2D Advection–Diffusion: converging pulses + rightward drift")
    plt.tight_layout()
    plt.show()


# ============================================================
#  Run as a script
# ============================================================

if __name__ == "__main__":
    x, y, t, U = solve_advection_diffusion_2d(
        nx=101,
        ny=101,
        Lx=1.0,
        Ly=1.0,
        v_bg=0.3,
        alpha=1.0,
        kappa=0.001,
        T=3.0,
        dt=None,
        save_every=5,
    )

    plot_snapshots(x, y, t, U, n_plots=5)
