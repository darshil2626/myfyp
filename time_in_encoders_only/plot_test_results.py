"""
Plot test reconstruction results (MSE, MAE, Max Error) against test timesteps.

This script visualizes the reconstruction quality metrics across held-out test
time steps to assess temporal interpolation capability.

Usage:
    python plot_test_results.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------
PKL_PATH = "test_reconstruction_results_sst.pkl"
PLOT_PATH = "test_results_plot_turb_flow_500t.png"


# -------------------------
# Load data
# -------------------------
print(f"Loading test results from {PKL_PATH}...")
try:
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
except FileNotFoundError:
    print(f"Error: {PKL_PATH} not found!")
    print("Make sure to run the solver with test evaluation first.")
    exit(1)

test_time_indices = np.array(data["test_time_indices"])
results = data["results"]

if not results:
    print("Error: No test results found in pickle file!")
    exit(1)

print(f"Loaded {len(results)} test timesteps")

# -------------------------
# Extract metrics
# -------------------------
mse_values = [r["mse"] for r in results]
mae_values = [r["mae"] for r in results]
max_error_values = [r["max_error"] for r in results]

# -------------------------
# Create figure with subplots
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: MSE over test timesteps
ax = axes[0, 0]
ax.plot(test_time_indices, mse_values, 'b-o', linewidth=2, markersize=6)
ax.set_xlabel("Test Timestep Index", fontsize=11)
ax.set_ylabel("MSE", fontsize=11)
ax.set_title("Mean Squared Error", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

# Plot 2: MAE over test timesteps
ax = axes[0, 1]
ax.plot(test_time_indices, mae_values, 'g-o', linewidth=2, markersize=6)
ax.set_xlabel("Test Timestep Index", fontsize=11)
ax.set_ylabel("MAE", fontsize=11)
ax.set_title("Mean Absolute Error", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

# Plot 3: Max Error over test timesteps
ax = axes[1, 0]
ax.plot(test_time_indices, max_error_values, 'r-o', linewidth=2, markersize=6)
ax.set_xlabel("Test Timestep Index", fontsize=11)
ax.set_ylabel("Max Error", fontsize=11)
ax.set_title("Maximum Absolute Error", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

# Plot 4: All metrics together for comparison
ax = axes[1, 1]
ax.plot(test_time_indices, mse_values, 'b-o', linewidth=2, markersize=6, label="MSE", alpha=0.8)
ax.plot(test_time_indices, mae_values, 'g-o', linewidth=2, markersize=6, label="MAE", alpha=0.8)
ax.plot(test_time_indices, max_error_values, 'r-o', linewidth=2, markersize=6, label="Max Error", alpha=0.8)
ax.set_xlabel("Test Timestep Index", fontsize=11)
ax.set_ylabel("Error", fontsize=11)
ax.set_title("All Error Metrics", fontsize=12, fontweight="bold")
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

plt.tight_layout()

# -------------------------
# Save and display info
# -------------------------
print(f"\nTest Results Summary:")
print(f"  Total test timesteps: {len(results)}")
print(f"\n  MSE:        min={min(mse_values):.6e}, max={max(mse_values):.6e}, mean={np.mean(mse_values):.6e}")
print(f"  MAE:        min={min(mae_values):.6e}, max={max(mae_values):.6e}, mean={np.mean(mae_values):.6e}")
print(f"  Max Error:  min={min(max_error_values):.6e}, max={max(max_error_values):.6e}, mean={np.mean(max_error_values):.6e}")

print(f"\nSaving plot to {PLOT_PATH}...")
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
print("Done!")

plt.show()
