"""
Animate two-stage test reconstructions as a 2x3 panel GIF.

Row 1: True Field | Stage 1 Prediction | Stage 1 Error
Row 2: CNN Correction | Combined Prediction | Combined Error

Reads the pkl file produced by twostage_sinn.py.

Usage:
    python twostage_animator.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# -------------------------
# Config
# -------------------------
PKL_PATH = "test_results_twostage.pkl"
GIF_PATH = "twostage_reconstruction.gif"
FPS = 15


# -------------------------
# Load data
# -------------------------
print(f"Loading results from {PKL_PATH}...")
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

results = data["results"]
num_frames = len(results)
print(f"Loaded {num_frames} test time steps")
print(f"Stage 1 mean MAE: {data['stage1_test_mae']:.4f}")
print(f"Combined mean MAE: {data['combined_test_mae']:.4f}")

# -------------------------
# Precompute global colour limits
# -------------------------
all_true = np.concatenate([r["u_true"].ravel() for r in results])
global_vmin = np.percentile(all_true, 2)
global_vmax = np.percentile(all_true, 98)

# Stage 1 error limits
s1_errors = []
combined_errors = []
for r in results:
    s1_err = np.abs(r["u_pred_stage1"] - r["u_true"])
    s1_errors.append(np.nan_to_num(s1_err, nan=0.0))
    comb_err = np.abs(r["u_error"])
    combined_errors.append(np.nan_to_num(comb_err, nan=0.0))

error_vmax = np.percentile(
    np.concatenate([e.ravel() for e in s1_errors]), 95
)

# Correction field limits (symmetric around 0)
corr_lim = np.percentile(
    np.concatenate([np.abs(r["correction"]).ravel() for r in results]), 95
)

# -------------------------
# Create figure
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(20, 11))

r0 = results[0]
s1_err_0 = s1_errors[0]
comb_err_0 = combined_errors[0]

# Row 1: True | Stage 1 | Stage 1 Error
im_true = axes[0, 0].imshow(r0["u_true"], cmap="viridis", origin="lower",
                              vmin=global_vmin, vmax=global_vmax, aspect="auto")
fig.colorbar(im_true, ax=axes[0, 0], fraction=0.046, pad=0.04)

im_s1 = axes[0, 1].imshow(r0["u_pred_stage1"], cmap="viridis", origin="lower",
                            vmin=global_vmin, vmax=global_vmax, aspect="auto")
fig.colorbar(im_s1, ax=axes[0, 1], fraction=0.046, pad=0.04)

im_s1err = axes[0, 2].imshow(s1_err_0, cmap="hot", origin="lower",
                               vmin=0, vmax=error_vmax, aspect="auto")
cb_s1err = fig.colorbar(im_s1err, ax=axes[0, 2], fraction=0.046, pad=0.04)
cb_s1err.set_label("Absolute Error (°C)")

# Row 2: Correction | Combined | Combined Error
im_corr = axes[1, 0].imshow(r0["correction"], cmap="RdBu_r", origin="lower",
                              vmin=-corr_lim, vmax=corr_lim, aspect="auto")
cb_corr = fig.colorbar(im_corr, ax=axes[1, 0], fraction=0.046, pad=0.04)
cb_corr.set_label("Correction (°C)")

im_comb = axes[1, 1].imshow(r0["u_pred"], cmap="viridis", origin="lower",
                              vmin=global_vmin, vmax=global_vmax, aspect="auto")
fig.colorbar(im_comb, ax=axes[1, 1], fraction=0.046, pad=0.04)

im_comberr = axes[1, 2].imshow(comb_err_0, cmap="hot", origin="lower",
                                 vmin=0, vmax=error_vmax, aspect="auto")
cb_comberr = fig.colorbar(im_comberr, ax=axes[1, 2], fraction=0.046, pad=0.04)
cb_comberr.set_label("Absolute Error (°C)")

# Stats text boxes
stats_s1 = axes[0, 2].text(
    0.02, 0.98, "", transform=axes[0, 2].transAxes, fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
stats_comb = axes[1, 2].text(
    0.02, 0.98, "", transform=axes[1, 2].transAxes, fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Titles
title_true = axes[0, 0].set_title("", fontsize=12, fontweight="bold")
title_s1 = axes[0, 1].set_title("", fontsize=12, fontweight="bold")
title_s1err = axes[0, 2].set_title("Stage 1 Error", fontsize=12, fontweight="bold")
title_corr = axes[1, 0].set_title("CNN Correction", fontsize=12, fontweight="bold")
title_comb = axes[1, 1].set_title("", fontsize=12, fontweight="bold")
title_comberr = axes[1, 2].set_title("Combined Error", fontsize=12, fontweight="bold")
suptitle = fig.suptitle("", fontsize=15, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])


# -------------------------
# Update function
# -------------------------
def update(frame_index):
    r = results[frame_index]
    t = r["t_index"]

    s1_err = s1_errors[frame_index]
    comb_err = combined_errors[frame_index]

    im_true.set_data(r["u_true"])
    im_s1.set_data(r["u_pred_stage1"])
    im_s1err.set_data(s1_err)
    im_corr.set_data(r["correction"])
    im_comb.set_data(r["u_pred"])
    im_comberr.set_data(comb_err)

    title_true.set_text(f"True Field (t={t})")
    title_s1.set_text(f"Stage 1 (MAE {r['mae_stage1']:.3f})")
    title_comb.set_text(f"Combined (MAE {r['mae']:.3f})")

    improvement = (1 - r["mae"] / r["mae_stage1"]) * 100
    suptitle.set_text(
        f"Two-Stage Reconstruction at t={t}  "
        f"[{frame_index+1}/{num_frames}]  |  "
        f"Improvement: {improvement:+.1f}%"
    )

    stats_s1.set_text(
        f"MAE: {r['mae_stage1']:.4f}\n"
        f"MSE: —"
    )
    stats_comb.set_text(
        f"MAE: {r['mae']:.4f}\n"
        f"MSE: {r['mse']:.4f}\n"
        f"Max: {r['max_error']:.4f}"
    )

    return (im_true, im_s1, im_s1err, im_corr, im_comb, im_comberr,
            stats_s1, stats_comb)


# -------------------------
# Build and save animation
# -------------------------
print(f"Building animation ({num_frames} frames)...")
anim = animation.FuncAnimation(
    fig, update, frames=num_frames, interval=1000 // FPS, blit=False,
)

print(f"Saving GIF to {GIF_PATH} at {FPS} fps...")
anim.save(GIF_PATH, writer="pillow", fps=FPS)
plt.close(fig)
print("Done!")
