"""
SST Interannual Variability Across ENSO Phases (2022–2024)
===========================================================
Characterises SST conditions across a complete ENSO cycle
in the South Pacific ROI to justify train/test split.

This figure goes in Chapter 3 (Problem Formulation & Data).

Usage:
    python plot_sst_3year_comparison.py
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATA_PATH, FIGURES_DIR,
    N_DAYS_2022, N_DAYS_2023, N_DAYS_2024, N_TOTAL,
    YEAR_LABELS, ensure_dirs,
)

ensure_dirs()

# ============================================================
# Load data
# ============================================================
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

U = np.array(data["U"])
print(f"Loaded U with shape: {U.shape}")
assert U.shape[0] == N_TOTAL, f"Expected {N_TOTAL} timesteps, got {U.shape[0]}"

idx = 0
U_by_year = {}
n_days = {2022: N_DAYS_2022, 2023: N_DAYS_2023, 2024: N_DAYS_2024}
for year, n in n_days.items():
    U_by_year[year] = U[idx:idx + n]
    idx += n

# ============================================================
# Compute daily spatial statistics
# ============================================================
mean_by_year = {}
std_by_year = {}
q25_by_year = {}
q75_by_year = {}
for year, U_yr in U_by_year.items():
    mean_by_year[year] = np.array([np.nanmean(U_yr[t]) for t in range(n_days[year])])
    std_by_year[year]  = np.array([np.nanstd(U_yr[t])  for t in range(n_days[year])])
    q25_by_year[year]  = np.array([np.nanpercentile(U_yr[t], 25) for t in range(n_days[year])])
    q75_by_year[year]  = np.array([np.nanpercentile(U_yr[t], 75) for t in range(n_days[year])])

doy_by_year = {year: np.arange(1, n + 1) for year, n in n_days.items()}

# ============================================================
# Pairwise stats
# ============================================================
n_common = 365
years = [2022, 2023, 2024]

stats = {}
for i in range(len(years)):
    for j in range(i + 1, len(years)):
        y1, y2 = years[i], years[j]
        m1 = mean_by_year[y1][:n_common]
        m2 = mean_by_year[y2][:n_common]
        diff = m2 - m1
        r = np.corrcoef(m1, m2)[0, 1]
        rmsd = np.sqrt(np.mean(diff**2))
        mean_diff = np.mean(diff)
        stats[(y1, y2)] = {"r": r, "rmsd": rmsd, "mean_diff": mean_diff}

print(f"\n{'='*55}")
print(f" Pairwise Interannual Comparison (first {n_common} days)")
print(f"{'='*55}")
print(f"{'Pair':<15} {'Pearson r':>10} {'RMSD (°C)':>10} {'Mean Δ (°C)':>12}")
print(f"{'-'*55}")
for (y1, y2), s in stats.items():
    print(f"{y1} vs {y2:<7} {s['r']:>10.4f} {s['rmsd']:>10.3f} {s['mean_diff']:>+12.3f}")

# ============================================================
# Combined years comparison
# ============================================================
# Combine all three years' mean and std data
combined_mean = np.concatenate([mean_by_year[year][:n_common] for year in years])
combined_std = np.concatenate([std_by_year[year][:n_common] for year in years])

combined_mean_overall = np.nanmean(combined_mean)
combined_std_overall = np.nanstd(combined_std)

print(f"\n{'='*55}")
print(f" Each Year vs Combined (first {n_common} days)")
print(f"{'='*55}")
print(f"{'Year':<15} {'Mean Δ vs Combined (°C)':>25} {'Std Dev Ratio':>15}")
print(f"{'-'*55}")
for year in years:
    year_mean = np.nanmean(mean_by_year[year][:n_common])
    year_std_mean = np.nanmean(std_by_year[year][:n_common])
    mean_change = year_mean - combined_mean_overall
    std_ratio = year_std_mean / combined_std_overall
    print(f"{year:<15} {mean_change:>+25.3f} {std_ratio:>15.4f}")

enso_labels = {2022: "La Niña", 2023: "El Niño", 2024: "Neutral"}

# ============================================================
# Plot
# ============================================================
colours = {2022: "#4daf4a", 2023: "#2166ac", 2024: "#b2182b"}
phase_labels = {
    2022: "2022 — La Niña (training)",
    2023: "2023 — El Niño (test)",
    2024: "2024 — ENSO-neutral (test)",
}

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                          gridspec_kw={"height_ratios": [3, 2, 1.5]})

month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --- Panel 1: SST across ENSO phases ---
ax = axes[0]
for year in years:
    ax.plot(doy_by_year[year], mean_by_year[year],
            color=colours[year], lw=1.8, label=phase_labels[year], alpha=0.9)
    ax.fill_between(doy_by_year[year],
                    q25_by_year[year], q75_by_year[year],
                    color=colours[year], alpha=0.12)

stats_text = (
    f"Pairwise Interannual Statistics\n"
    f"  2022 vs 2023 (La Niña / El Niño):   r = {stats[(2022,2023)]['r']:.3f},  RMSD = {stats[(2022,2023)]['rmsd']:.3f} °C\n"
    f"  2022 vs 2024 (La Niña / Neutral):   r = {stats[(2022,2024)]['r']:.3f},  RMSD = {stats[(2022,2024)]['rmsd']:.3f} °C\n"
    f"  2023 vs 2024 (El Niño / Neutral):   r = {stats[(2023,2024)]['r']:.3f},  RMSD = {stats[(2023,2024)]['rmsd']:.3f} °C"
)
ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=9.5,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))

ax.set_ylabel("SST (°C)", fontsize=12)
ax.set_title("Daily Spatial-Mean SST Across ENSO Phases in South Pacific ROI",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

# --- Panel 2: Test year deviations from training year ---
ax = axes[1]
doy_common = np.arange(1, n_common + 1)
diff_23 = mean_by_year[2023][:n_common] - mean_by_year[2022][:n_common]
diff_24 = mean_by_year[2024][:n_common] - mean_by_year[2022][:n_common]

ax.plot(doy_common, diff_23, color=colours[2023], lw=1.5, alpha=0.85,
        label=f"2023 − 2022 (RMSD = {stats[(2022,2023)]['rmsd']:.3f} °C)")
ax.plot(doy_common, diff_24, color=colours[2024], lw=1.5, alpha=0.85,
        label=f"2024 − 2022 (RMSD = {stats[(2022,2024)]['rmsd']:.3f} °C)")
ax.fill_between(doy_common, diff_23, 0, color=colours[2023], alpha=0.12)
ax.fill_between(doy_common, diff_24, 0, color=colours[2024], alpha=0.12)
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_ylabel("ΔSST from training year (°C)", fontsize=11)
ax.set_title("Test Year SST Anomaly Relative to Training Conditions",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
ax.grid(True, alpha=0.3)

# --- Panel 3: Spatial variability ---
ax = axes[2]
for year in years:
    ax.plot(doy_by_year[year], std_by_year[year],
            color=colours[year], lw=1.2, label=f"{year} ({enso_labels[year]})", alpha=0.8)
ax.set_ylabel("Spatial σ (°C)", fontsize=10)
ax.set_xlabel("Day of Year", fontsize=12)
ax.set_title("Spatial SST Variability Within ROI Across ENSO Phases", fontsize=11)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)
ax.set_xlim(1, 366)

plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/sst_enso_phase_comparison.png", dpi=300, bbox_inches="tight")
print(f"\nSaved sst_enso_phase_comparison.png")

# ============================================================
# Figure: Seasonal SST fields — 2022 training year
# ============================================================
X_grid = np.array(data["X"])   # (ny, nx) longitudes
Y_grid = np.array(data["Y"])   # (ny, nx) latitudes

U_2022 = U_by_year[2022]  # (365, ny, nx)

# 0-based day-of-year index ranges for each austral season within 2022
# (2022 is not a leap year: Feb has 28 days)
_season_idx = {
    "DJF": list(range(0, 59)) + list(range(334, 365)),   # Jan + Feb + Dec
    "MAM": list(range(59, 151)),
    "JJA": list(range(151, 243)),
    "SON": list(range(243, 334)),
}
_season_titles = {
    "DJF": "DJF — Austral Summer",
    "MAM": "MAM — Austral Autumn",
    "JJA": "JJA — Austral Winter",
    "SON": "SON — Austral Spring",
}

season_fields = {s: np.nanmean(U_2022[idx], axis=0) for s, idx in _season_idx.items()}

all_vals = np.concatenate([f.ravel() for f in season_fields.values()])
vmin_s = np.nanpercentile(all_vals, 2)
vmax_s = np.nanpercentile(all_vals, 98)

lon_min_roi = float(X_grid.min())
lon_max_roi = float(X_grid.max())
lat_min_roi = float(Y_grid.min())
lat_max_roi = float(Y_grid.max())

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
axes2_flat = axes2.ravel()

seasons_order = ["DJF", "MAM", "JJA", "SON"]
im_s = None
for i, s in enumerate(seasons_order):
    ax = axes2_flat[i]
    im_s = ax.pcolormesh(X_grid, Y_grid, season_fields[s],
                         cmap="RdYlBu_r", vmin=vmin_s, vmax=vmax_s, shading="auto")
    ax.set_title(_season_titles[s], fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.tick_params(labelsize=8)

cbar2 = fig2.colorbar(im_s, ax=axes2_flat.tolist(),
                      orientation="vertical", shrink=0.8, pad=0.02)
cbar2.set_label("SST (°C)", fontsize=11)

fig2.suptitle("Representative SST Fields — 2022 Training Year (Austral Seasons)",
              fontsize=13, fontweight="bold")

plt.savefig(f"{FIGURES_DIR}/sst_seasonal_2022.png", dpi=300, bbox_inches="tight")
print(f"Saved sst_seasonal_2022.png")
