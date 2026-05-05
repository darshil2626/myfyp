"""
SST Annual Comparison: 2023 vs 2024
====================================
Plots daily spatial-mean SST in the ROI for both years overlaid.
Quantifies interannual similarity to justify train/test split.

Usage:
    python plot_sst_annual_comparison.py

Expects:
    sst_2023_2024_combined.pkl  in the same directory (or edit DATA_PATH below)
    containing keys: X, Y, U (shape: [731, ny, nx]), T
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import date, timedelta

# ============================================================
# CONFIG — edit these
# ============================================================
DATA_PATH = r"c:\Users\darsh\Documents\fyp\myfyp\time_in_encoders_only\sst\sst_2023_2024_combined.pkl"
SAVE_PATH = "sst_2023_vs_2024_comparison.png"

N_DAYS_2023 = 365
N_DAYS_2024 = 366  # leap year

# ============================================================
# Load data
# ============================================================
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

U = np.array(data["U"])  # shape: (nt, ny, nx)
print(f"Loaded U with shape: {U.shape}")
assert U.shape[0] == N_DAYS_2023 + N_DAYS_2024, (
    f"Expected {N_DAYS_2023 + N_DAYS_2024} timesteps, got {U.shape[0]}"
)

U_2023 = U[:N_DAYS_2023]   # (365, ny, nx)
U_2024 = U[N_DAYS_2023:]   # (366, ny, nx)

# ============================================================
# Compute daily spatial-mean SST (ignoring NaNs if any)
# ============================================================
mean_2023 = np.array([np.nanmean(U_2023[t]) for t in range(N_DAYS_2023)])
mean_2024 = np.array([np.nanmean(U_2024[t]) for t in range(N_DAYS_2024)])

# Day-of-year axes
doy_2023 = np.arange(1, N_DAYS_2023 + 1)
doy_2024 = np.arange(1, N_DAYS_2024 + 1)

# Calendar dates for x-axis labels
dates_2023 = [date(2023, 1, 1) + timedelta(days=int(d)) for d in range(N_DAYS_2023)]
dates_2024 = [date(2024, 1, 1) + timedelta(days=int(d)) for d in range(N_DAYS_2024)]

# ============================================================
# Compute similarity statistics
# ============================================================
# Truncate to common length (365 days) for direct comparison
n_common = min(N_DAYS_2023, N_DAYS_2024)
diff = mean_2024[:n_common] - mean_2023[:n_common]
corr = np.corrcoef(mean_2023[:n_common], mean_2024[:n_common])[0, 1]
rmsd = np.sqrt(np.mean(diff**2))
max_diff = np.max(np.abs(diff))
mean_diff = np.mean(diff)

print(f"\n--- Interannual comparison (spatial-mean SST) ---")
print(f"  Pearson correlation:      {corr:.4f}")
print(f"  RMSD:                     {rmsd:.3f} °C")
print(f"  Mean difference (24-23):  {mean_diff:+.3f} °C")
print(f"  Max absolute difference:  {max_diff:.3f} °C")
print(f"  2023 range:               {mean_2023.min():.2f} – {mean_2023.max():.2f} °C")
print(f"  2024 range:               {mean_2024.min():.2f} – {mean_2024.max():.2f} °C")

# ============================================================
# Also compute spatial std per day (shows variability within ROI)
# ============================================================
std_2023 = np.array([np.nanstd(U_2023[t]) for t in range(N_DAYS_2023)])
std_2024 = np.array([np.nanstd(U_2024[t]) for t in range(N_DAYS_2024)])

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

# --- Panel 1: Overlaid annual SST curves ---
ax = axes[0]
ax.plot(doy_2023, mean_2023, color="#2166ac", lw=1.5, label="2023 (training)", alpha=0.9)
ax.plot(doy_2024, mean_2024, color="#b2182b", lw=1.5, label="2024 (testing)", alpha=0.9)
ax.fill_between(doy_2023, mean_2023 - std_2023, mean_2023 + std_2023,
                color="#2166ac", alpha=0.12, label="2023 ± 1σ (spatial)")
ax.fill_between(doy_2024, mean_2024 - std_2024, mean_2024 + std_2024,
                color="#b2182b", alpha=0.12, label="2024 ± 1σ (spatial)")
ax.set_ylabel("SST (°C)", fontsize=12)
ax.set_title("Daily Spatial-Mean SST in ROI: 2023 vs 2024", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Add stats text box
stats_text = (
    f"Pearson r = {corr:.4f}\n"
    f"RMSD = {rmsd:.3f} °C\n"
    f"Mean Δ (2024−2023) = {mean_diff:+.3f} °C"
)
ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4",
        facecolor="white", edgecolor="gray", alpha=0.9))

# --- Panel 2: Daily difference ---
ax = axes[1]
ax.bar(doy_2023, diff, width=1.0, color=np.where(diff > 0, "#b2182b", "#2166ac"), alpha=0.7)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Δ SST (°C)\n(2024 − 2023)", fontsize=10)
ax.set_title("Daily Difference in Spatial-Mean SST", fontsize=11)
ax.grid(True, alpha=0.3)

# --- Panel 3: Spatial standard deviation ---
ax = axes[2]
ax.plot(doy_2023, std_2023, color="#2166ac", lw=1.2, label="2023", alpha=0.8)
ax.plot(doy_2024, std_2024, color="#b2182b", lw=1.2, label="2024", alpha=0.8)
ax.set_ylabel("Spatial σ (°C)", fontsize=10)
ax.set_xlabel("Day of Year", fontsize=12)
ax.set_title("Spatial Variability Within ROI", fontsize=11)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

# Month labels on x-axis
month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ax.set_xticks(month_starts)
ax.set_xticklabels(month_labels)
ax.set_xlim(1, 366)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
print(f"\nSaved figure to: {SAVE_PATH}")
plt.show()
