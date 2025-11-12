# NUMERICAL SOLUTION WITH ABSORBING BC

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def apply_absorbing_bc(U, U_prev, wave_dict):
    c = wave_dict['c']
    dt = wave_dict['dt']
    dx = wave_dict['dx']
    dy = wave_dict['dy']
    kx = c * dt / dx
    ky = c * dt / dy
    Nx = U.shape[0] - 1
    Ny = U.shape[1] - 1

    # Left and right x-boundaries
    U[0, 1:-1]  = U_prev[1, 1:-1] + (kx - 1) / (kx + 1) * (U[1, 1:-1] - U_prev[0, 1:-1])
    U[Nx, 1:-1] = U_prev[Nx - 1, 1:-1] + (kx - 1) / (kx + 1) * (U[Nx - 1, 1:-1] - U_prev[Nx, 1:-1])
    # Top and bottom y-boundaries
    U[1:-1, 0]  = U_prev[1:-1, 1] + (ky - 1) / (ky + 1) * (U[1:-1, 1] - U_prev[1:-1, 0])
    U[1:-1, Ny] = U_prev[1:-1, Ny - 1] + (ky - 1) / (ky + 1) * (U[1:-1, Ny - 1] - U_prev[1:-1, Ny])
    # Corners
    U[0, 0] = U[1, 1]
    U[0, Ny] = U[1, Ny - 1]
    U[Nx, 0] = U[Nx - 1, 1]
    U[Nx, Ny] = U[Nx - 1, Ny - 1]
    return U

def solve_wave_equation_abc(wave_dict, init_dict):
    Nx = int(wave_dict['Lx'] / wave_dict['dx'])
    Ny = int(wave_dict['Ly'] / wave_dict['dy'])
    x = np.linspace(0, wave_dict['Lx'], Nx + 1)
    y = np.linspace(0, wave_dict['Ly'], Ny + 1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dt = wave_dict['CFL'] * min(wave_dict['dx'], wave_dict['dy']) / wave_dict['c'] / np.sqrt(2)
    Nt = int(wave_dict['Lt'] / dt)
    T = np.linspace(0, wave_dict['Lt'], Nt + 1)
    wave_dict['dt'] = dt  # Add dt for boundary function

    print(f"Wave Solver (ABCs): Grid {Nx} × {Ny}, Nt={Nt}, dt={dt:.6f}")

    x_c = np.atleast_1d(init_dict['x_c'])
    y_c = np.atleast_1d(init_dict['y_c'])
    width = np.atleast_1d(init_dict['width'])
    amplitude = np.atleast_1d(init_dict['amplitude'])

    u0 = np.zeros_like(X, dtype=float)
    for xx_c, yy_c, w, a in zip(x_c, y_c, width, amplitude):
        u0 += a * np.exp(-((X - xx_c) ** 2 + (Y - yy_c) ** 2) / (2 * w ** 2))

    U = np.zeros((Nt + 1, Nx + 1, Ny + 1))
    U[0] = u0

    alpha = (wave_dict['c'] * dt) ** 2

    Lap0 = (
        (U[0, 2:, 1:-1] - 2 * U[0, 1:-1, 1:-1] + U[0, :-2, 1:-1]) / wave_dict['dx'] ** 2
        + (U[0, 1:-1, 2:] - 2 * U[0, 1:-1, 1:-1] + U[0, 1:-1, :-2]) / wave_dict['dy'] ** 2
    )
    U[1, 1:-1, 1:-1] = U[0, 1:-1, 1:-1] + 0.5 * alpha * Lap0
    U[1] = apply_absorbing_bc(U[1], U[0], wave_dict)

    for n in range(1, Nt):
        Lap = (
            (U[n, 2:, 1:-1] - 2 * U[n, 1:-1, 1:-1] + U[n, :-2, 1:-1]) / wave_dict['dx'] ** 2
            + (U[n, 1:-1, 2:] - 2 * U[n, 1:-1, 1:-1] + U[n, 1:-1, :-2]) / wave_dict['dy'] ** 2
        )
        U[n + 1, 1:-1, 1:-1] = (
            2 * U[n, 1:-1, 1:-1] - U[n - 1, 1:-1, 1:-1] + alpha * Lap
        )
        U[n + 1] = apply_absorbing_bc(U[n + 1], U[n], wave_dict)

    print(f"Wave solution computed: U.shape = {U.shape}")
    return U, x, y, X, Y, T

wave_dict = {
    'c': 1.5,
    'Lt': 5.0,
    'Lx': 1.0,
    'Ly': 1.0,
    'dx': 0.01,
    'dy': 0.01,
    'CFL': 0.9
}
init_dict = {
    'x_c': 0.25,
    'y_c': 0.5,
    'width': np.sqrt(0.05),
    'amplitude': 2.0
}

U, x, y, X, Y, T = solve_wave_equation_abc(wave_dict, init_dict)

U_down = U.copy()
NT, NX, NY = U_down.shape

fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
cs = axs[0].imshow(
    U_down[0],
    origin='lower',
    extent=[0, wave_dict['Lx'], 0, wave_dict['Ly']],
    cmap='RdBu_r',
    aspect='auto'
)
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
title = axs[0].set_title(f"t = {T[0]:.2f}s")

plot_x_idx = int(init_dict["x_c"] / wave_dict['dx'])
(line,) = axs[1].plot(np.linspace(0, wave_dict['Ly'], NY), U_down[0, plot_x_idx, :], color='k')
axs[1].set_xlabel("y")
axs[1].set_ylabel("u(t, x=$x_c$, y)")
axs[1].set_ylim(0.9*np.min(U_down), 1.1*np.max(U_down))

def update(frame):
    cs.set_data(U_down[frame])
    line.set_ydata(U_down[frame, plot_x_idx, :])
    title.set_text(f"t = {T[frame]:.2f}s")
    return cs, line, title

ani = animation.FuncAnimation(fig, update, frames=NT, interval=50, blit=False)

plt.show()