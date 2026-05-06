"""
Animate test-time reconstructions as a 3-panel GIF (True | Predicted | % Error).

Reads the pkl file produced by save_test_results.py and creates an animated
GIF showing how the reconstruction evolves across held-out time steps.

Usage:
    python animate_test_results.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# -------------------------
# Config
# -------------------------
PKL_PATH = "test_reconstruction_results_500t.pkl"
GIF_PATH = "test_reconstruction_turb_500t.gif"
FPS = 20


# -------------------------
# Load data
# -------------------------
print(f"Loading results from {PKL_PATH}...")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

test_time_indices = data["test_time_indices"]
results = data["results"]
num_frames = len(results)
print(f"Loaded {num_frames} test time steps")

# -------------------------
# Precompute global colour limits (so scale is consistent across frames)
# -------------------------
global_vmin = np.percentile(np.concatenate([r["u_true"].ravel() for r in results]), 2)
global_vmax = np.percentile(np.concatenate([r["u_true"].ravel() for r in results]), 98)

# For absolute error, use percentile-based scaling
abs_errors = []
for r in results:
    abs_err = np.abs(r["u_error"])
    abs_err = np.nan_to_num(abs_err, nan=0.0, posinf=0.0, neginf=0.0)
    abs_errors.append(abs_err)
error_vmin = np.percentile(np.concatenate([a.ravel() for a in abs_errors]), 5)
error_vmax = np.percentile(np.concatenate([a.ravel() for a in abs_errors]), 95)

# -------------------------
# Create figure
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Initial frame
r0 = results[0]
u_true_0 = r0["u_true"]
u_pred_0 = r0["u_pred"]
abs_err_0 = abs_errors[0]
boundary_mask_0 = r0["boundary_mask"]
boundary_coords = np.argwhere(boundary_mask_0)

# Panel 1: True field
im1 = axes[0].imshow(u_true_0, cmap="viridis", origin="lower",
                      vmin=global_vmin, vmax=global_vmax, aspect="auto")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
scat1 = axes[0].scatter(boundary_coords[:, 1], boundary_coords[:, 0],
                         c="red", s=1, alpha=0.3)
fig.colorbar(im1, ax=axes[0]).set_label("u")

# Panel 2: Predicted field
im2 = axes[1].imshow(u_pred_0, cmap="viridis", origin="lower",
                      vmin=global_vmin, vmax=global_vmax, aspect="auto")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
scat2 = axes[1].scatter(boundary_coords[:, 1], boundary_coords[:, 0],
                         c="red", s=1, alpha=0.3)
fig.colorbar(im2, ax=axes[1]).set_label("u")

# Panel 3: Absolute error
im3 = axes[2].imshow(abs_err_0, cmap="hot", origin="lower",
                      vmin=error_vmin, vmax=error_vmax, aspect="auto")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
scat3 = axes[2].scatter(boundary_coords[:, 1], boundary_coords[:, 0],
                         c="cyan", s=1, alpha=0.5)
fig.colorbar(im3, ax=axes[2]).set_label("error")

# Stats text box
stats_text = axes[2].text(
    0.02, 0.98, "",
    transform=axes[2].transAxes, fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Titles (updated each frame)
title1 = axes[0].set_title("", fontsize=13, fontweight="bold")
title2 = axes[1].set_title("", fontsize=13, fontweight="bold")
title3 = axes[2].set_title("Absolute Error", fontsize=13, fontweight="bold")
suptitle = fig.suptitle("", fontsize=15, fontweight="bold", y=1)

plt.tight_layout()


# -------------------------
# Update function
# -------------------------
def update(frame_index):
    r = results[frame_index]
    t_idx = r["t_index"]

    u_true = r["u_true"]
    u_pred = r["u_pred"]
    abs_err = abs_errors[frame_index]

    im1.set_data(u_true)
    im2.set_data(u_pred)
    im3.set_data(abs_err)

    title1.set_text(f"True Field (t={t_idx})")
    title2.set_text(f"Predicted Field (t={t_idx})")
    suptitle.set_text(f"Test Reconstruction at t={t_idx}  "
                      f"[{frame_index+1}/{num_frames}]")

    stats_text.set_text(
        f"MSE: {r['mse']:.4e}\n"
        f"MAE: {r['mae']:.4e}\n"
        f"Max: {r['max_error']:.4e}"
    )

    return im1, im2, im3, stats_text


# -------------------------
# Build animation
# -------------------------
print("Building animation...")
anim = animation.FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=100,
    blit=False,
)

# -------------------------
# Save as GIF
# -------------------------
print(f"Saving GIF to {GIF_PATH} ({num_frames} frames, {FPS} fps)...")
anim.save(GIF_PATH, writer="pillow", fps=FPS)
plt.close(fig)
print("Done!")