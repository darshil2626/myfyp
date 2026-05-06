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
# Compute daily spatial-mean and spatial-std
# ============================================================
mean_by_year = {}
std_by_year = {}
for year, U_yr in U_by_year.items():
    mean_by_year[year] = np.array([np.nanmean(U_yr[t]) for t in range(n_days[year])])
    std_by_year[year] = np.array([np.nanstd(U_yr[t]) for t in range(n_days[year])])

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
                    mean_by_year[year] - std_by_year[year],
                    mean_by_year[year] + std_by_year[year],
                    color=colours[year], alpha=0.08)

stats_text = (
    f"Correlation with training year (2022)\n"
    f"  2023 (El Niño):    r = {stats[(2022,2023)]['r']:.3f},  RMSD = {stats[(2022,2023)]['rmsd']:.3f} °C\n"
    f"  2024 (Neutral):    r = {stats[(2022,2024)]['r']:.3f},  RMSD = {stats[(2022,2024)]['rmsd']:.3f} °C"
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
