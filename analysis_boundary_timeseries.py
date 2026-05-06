"""
Boundary SST time series analysis near the complex (high-error) region.

Identifies the highest-error region from the spatial error field,
picks boundary points adjacent to it, and plots their SST evolution
over 2022–2024. Checks for oscillatory behaviour.

Addresses supervisor feedback:
  "Observe how the boundary value near the complex region changes over
   time and if it is oscillatory then it means there is a flux of flow
   through the boundary."

Usage:
    python analysis_boundary_timeseries.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from config import (
    B_THICK, FIGURES_DIR, RESULTS_DIR,
    N_DAYS_2022, N_DAYS_2023, N_DAYS_2024, ensure_dirs,
)
from utils import load_data

ensure_dirs()


def find_high_error_boundary_points(error_field, obs_mask, n_points=4):
    """
    Find boundary points adjacent to the highest-error interior region.
    
    Strategy: find the interior point with highest time-averaged error,
    then pick the nearest boundary points.
    """
    interior = ~obs_mask
    err = error_field.copy()
    err[obs_mask] = 0.0  # zero out boundary

    # Smooth to find the region, not a single pixel
    from scipy.ndimage import uniform_filter
    err_smooth = uniform_filter(err, size=10)
    err_smooth[obs_mask] = 0.0

    # Find peak
    peak_idx = np.unravel_index(np.argmax(err_smooth), err_smooth.shape)
    peak_y, peak_x = peak_idx
    print(f"  Peak error region centred at (y={peak_y}, x={peak_x})")

    # Find nearest boundary points
    bnd_y, bnd_x = np.where(obs_mask)
    dists = np.sqrt((bnd_y - peak_y)**2 + (bnd_x - peak_x)**2)
    nearest_idx = np.argsort(dists)[:n_points * 4]  # oversample then pick diverse

    # Pick points spread across different boundary edges
    selected = []
    edges_used = set()
    for i in nearest_idx:
        y, x = int(bnd_y[i]), int(bnd_x[i])
        # Determine which edge
        edge = None
        if y < B_THICK: edge = "top"
        elif y >= error_field.shape[0] - B_THICK: edge = "bottom"
        elif x < B_THICK: edge = "left"
        elif x >= error_field.shape[1] - B_THICK: edge = "right"
        
        if edge and edge not in edges_used:
            selected.append((y, x, edge))
            edges_used.add(edge)
        elif len(selected) < n_points:
            selected.append((y, x, edge or "interior"))
        
        if len(selected) >= n_points:
            break

    # If we didn't get enough, just take the nearest
    while len(selected) < n_points:
        i = nearest_idx[len(selected)]
        selected.append((int(bnd_y[i]), int(bnd_x[i]), "nearest"))

    return selected[:n_points], (peak_y, peak_x)


def main():
    data = load_data()
    U = np.asarray(data["U"], dtype=np.float32)
    nt, ny, nx = U.shape

    # Load spatial error field
    try:
        npz = np.load(f"{RESULTS_DIR}/spatial_error_fields.npz")
        s1_error_avg = npz["s1_error_avg"]
    except FileNotFoundError:
        print("ERROR: Run analysis_spatial_error.py first to generate spatial_error_fields.npz")
        return

    obs_mask = np.zeros((ny, nx), dtype=bool)
    obs_mask[:B_THICK, :] = True; obs_mask[-B_THICK:, :] = True
    obs_mask[:, :B_THICK] = True; obs_mask[:, -B_THICK:] = True

    # Find points
    points, peak = find_high_error_boundary_points(s1_error_avg, obs_mask, n_points=4)
    print(f"  Selected boundary points: {points}")

    # Time axis with calendar dates
    days = np.arange(nt)
    # Markers for year boundaries
    year_boundaries = [0, N_DAYS_2022, N_DAYS_2022 + N_DAYS_2023]
    year_labels_pos = [
        N_DAYS_2022 // 2,
        N_DAYS_2022 + N_DAYS_2023 // 2,
        N_DAYS_2022 + N_DAYS_2023 + N_DAYS_2024 // 2,
    ]

    colours = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    # ---- Figure: time series ----
    fig, axes = plt.subplots(len(points) + 1, 1, figsize=(14, 3 * (len(points) + 1)),
                              sharex=True)

    for i, (py, px, edge) in enumerate(points):
        ax = axes[i]
        sst = U[:, py, px]
        ax.plot(days, sst, color=colours[i], lw=1.0, alpha=0.9)
        ax.set_ylabel("SST (°C)", fontsize=10)
        ax.set_title(f"Boundary point ({py}, {px}) — {edge} edge", fontsize=11)
        ax.axvline(N_DAYS_2022, color="black", ls="--", lw=0.8, alpha=0.6)
        ax.axvline(N_DAYS_2022 + N_DAYS_2023, color="black", ls="--", lw=0.8, alpha=0.6)
        ax.axvspan(0, N_DAYS_2022, alpha=0.05, color="green", label="Train (2022)")
        ax.axvspan(N_DAYS_2022, nt, alpha=0.05, color="red", label="Test (2023-24)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")

    # Bottom panel: power spectrum of first point to check oscillatory nature
    ax = axes[-1]
    sst_full = U[:, points[0][0], points[0][1]]
    # Detrend
    sst_detrend = sst_full - np.convolve(sst_full, np.ones(30)/30, mode="same")
    freqs, psd = welch(sst_detrend, fs=1.0, nperseg=min(256, nt // 2))
    ax.semilogy(1.0 / freqs[1:], psd[1:], "k-", lw=1.5)
    ax.set_xlabel("Period (days)", fontsize=11)
    ax.set_ylabel("Power spectral density", fontsize=10)
    ax.set_title(f"PSD of boundary SST at ({points[0][0]}, {points[0][1]}) — "
                 f"oscillatory if strong peaks exist", fontsize=11)
    ax.axvline(365, color="red", ls=":", lw=1, alpha=0.7, label="Annual (365d)")
    ax.axvline(182, color="blue", ls=":", lw=1, alpha=0.7, label="Semi-annual (182d)")
    ax.set_xlim(5, 500)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Boundary SST Near High-Error Region (2022–2024)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/boundary_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved boundary_timeseries.png")

    # ---- Location map ----
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(np.nan_to_num(s1_error_avg, nan=0), cmap="hot",
                    origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax, label="Time-averaged |error| (°C)")
    for i, (py, px, edge) in enumerate(points):
        ax.plot(px, py, "o", color=colours[i], markersize=10,
                markeredgecolor="white", markeredgewidth=2,
                label=f"({py},{px}) — {edge}")
    ax.plot(peak[1], peak[0], "x", color="cyan", markersize=15, markeredgewidth=3,
            label="Error peak")
    ax.set_title("Selected Boundary Points Near High-Error Region", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/boundary_points_location.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved boundary_points_location.png")


if __name__ == "__main__":
    main()
