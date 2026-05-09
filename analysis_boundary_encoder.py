"""
Boundary encoder accuracy check.

Tests whether the boundary encoder is doing its job correctly by comparing:
  - Latent values produced by the BOUNDARY encoder at boundary points
  - Latent values produced by the INTERIOR encoder at the same boundary points

If these agree closely, the bottleneck is definitively in the elliptic
solve (not in the encoding step), isolating the maximum principle as the
root cause of oversmoothing.

Produces:
  - Scatter plot: boundary encoder latent vs interior encoder latent (per dim)
  - Correlation coefficient and RMSE per latent dimension
  - Figure: figures/boundary_encoder_check.png

Usage:
    python analysis_boundary_encoder.py
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEFAULT_SEED, FIGURES_DIR, SINN_V3_MODULE, B_THICK, ensure_dirs,
)
from utils import load_sinn_class, load_data, setup_sinn, load_sinn_weights

ensure_dirs()


def main():
    data = load_data()

    sinn_class = load_sinn_class(SINN_V3_MODULE)
    solver = setup_sinn(sinn_class, data, seed=DEFAULT_SEED)
    load_sinn_weights(solver, f"v3_seed{DEFAULT_SEED}")
    print("[info] v3 weights loaded")

    # Use a handful of timesteps spread across train and test sets
    n_times = 20
    train_sample = solver.train_time_indices[
        np.linspace(0, len(solver.train_time_indices)-1, n_times//2, dtype=int)
    ]
    test_sample = solver.test_time_indices[
        np.linspace(0, len(solver.test_time_indices)-1, n_times//2, dtype=int)
    ]
    timesteps = np.concatenate([train_sample, test_sample])

    boundary_indices = solver._boundary_indices
    interior_indices = solver._interior_indices
    bnd_y = boundary_indices[:, 0].astype(np.int32)
    bnd_x = boundary_indices[:, 1].astype(np.int32)

    bnd_latents_all  = []   # from boundary encoder
    int_latents_all  = []   # from interior encoder at same points

    print(f"Processing {len(timesteps)} timesteps...")
    for t in timesteps:
        t = int(t)
        boundary_t = np.full(len(bnd_y), t, dtype=np.int32)

        # Boundary encoder path
        idx_bnd = np.stack([boundary_t, bnd_y, bnd_x], axis=1).astype(np.int32)
        bnd_feats = solver._stack_mask_patch_features_from_idx(idx_bnd, apply_obs_mask=True)
        bnd_lat = solver.boundary_encoder(
            tf.constant(bnd_feats, tf.float32), training=False).numpy()

        # Interior encoder path — applied at the same boundary coordinates
        # (apply_obs_mask=False so it uses the full patch, not the masked version)
        int_feats = solver._stack_mask_patch_features_from_idx(idx_bnd, apply_obs_mask=False)
        int_lat = solver.interior_encoder(
            tf.constant(int_feats, tf.float32), training=False).numpy()

        bnd_latents_all.append(bnd_lat)
        int_latents_all.append(int_lat)

    bnd_latents_all = np.concatenate(bnd_latents_all, axis=0)   # (N_total, r)
    int_latents_all = np.concatenate(int_latents_all, axis=0)   # (N_total, r)

    latent_dim = bnd_latents_all.shape[1]

    print(f"\n{'='*60}")
    print("Boundary encoder vs Interior encoder at boundary points")
    print(f"{'='*60}")
    print(f"{'Dim':>4} {'Pearson r':>10} {'RMSE':>10} {'Mean |diff|':>12}")
    print(f"{'-'*40}")

    corrs, rmses = [], []
    for d in range(latent_dim):
        b = bnd_latents_all[:, d]
        i = int_latents_all[:, d]
        valid = ~(np.isnan(b) | np.isnan(i) | np.isinf(b) | np.isinf(i))
        if valid.sum() < 10:
            continue
        r = np.corrcoef(b[valid], i[valid])[0, 1]
        rmse = np.sqrt(np.mean((b[valid] - i[valid])**2))
        mean_diff = np.mean(np.abs(b[valid] - i[valid]))
        corrs.append(r); rmses.append(rmse)
        print(f"  {d:2d}  {r:>10.4f}  {rmse:>10.4f}  {mean_diff:>12.4f}")

    print(f"\n  Mean Pearson r: {np.mean(corrs):.4f}")
    print(f"  Mean RMSE:      {np.mean(rmses):.4f}")
    if np.mean(corrs) > 0.9:
        print("\n  → Boundary encoder is accurate (r > 0.9)")
        print("    Bottleneck is definitively in the elliptic solve,")
        print("    not in the encoding step. Max principle argument is solid.")
    elif np.mean(corrs) > 0.7:
        print("\n  → Boundary encoder has moderate accuracy (0.7 < r < 0.9)")
        print("    Some encoding error present but elliptic solve still dominant.")
    else:
        print("\n  → Boundary encoder is inaccurate (r < 0.7)")
        print("    Encoding quality is a contributing factor to oversmoothing.")

    # ---- Scatter plot ----
    n_cols = min(4, latent_dim)
    n_rows = (latent_dim + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for d in range(latent_dim):
        ax = axes[d // n_cols][d % n_cols]
        b = bnd_latents_all[:, d]
        i = int_latents_all[:, d]
        valid = ~(np.isnan(b) | np.isnan(i) | np.isinf(b) | np.isinf(i))
        ax.scatter(b[valid], i[valid], s=1, alpha=0.15, c="steelblue", rasterized=True)
        lo = min(b[valid].min(), i[valid].min())
        hi = max(b[valid].max(), i[valid].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y=x")
        r = corrs[d] if d < len(corrs) else float("nan")
        ax.set_title(f"Dim {d}  (r={r:.3f})", fontsize=10)
        ax.set_xlabel("Boundary encoder", fontsize=9)
        ax.set_ylabel("Interior encoder", fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx in range(latent_dim, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(
        "Boundary Encoder Accuracy Check\n"
        "Scatter: boundary encoder latent vs interior encoder latent at boundary pixels\n"
        f"High r → encoding is accurate → bottleneck = elliptic solve (max principle)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = f"{FIGURES_DIR}/boundary_encoder_check.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
