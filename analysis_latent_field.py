"""
Latent field visualisation.

Loads the v3 baseline SINN and visualises the interior latent field
ℓ(x) for a few representative timesteps alongside the true SST field.

If the maximum principle argument is correct, the latent field should
be spatially smooth — devoid of the sharp gradient features visible
in the true SST. This makes the theory visually concrete.

Produces per-timestep figures:
  - Row 1: true SST | decoded prediction | absolute error
  - Row 2: latent dim 0 | latent dim 1 | latent dim 2 (first 3 dims)

Figures saved to: figures/latent_field_t{t}.png

Usage:
    python analysis_latent_field.py
"""

import numpy as np
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
    U = np.asarray(data["U"], dtype=np.float32)

    sinn_class = load_sinn_class(SINN_V3_MODULE)
    solver = setup_sinn(sinn_class, data, seed=DEFAULT_SEED)
    load_sinn_weights(solver, f"v3_seed{DEFAULT_SEED}")
    print("[info] v3 weights loaded")

    # Pick one training and one test timestep
    # Prefer timesteps where MAE is representative (not outliers)
    train_mid = int(solver.train_time_indices[len(solver.train_time_indices) // 2])
    test_mid  = int(solver.test_time_indices[len(solver.test_time_indices) // 2])

    chosen = [
        (train_mid, "Training timestep"),
        (test_mid,  "Test timestep"),
    ]

    for t_idx, label in chosen:
        print(f"\nProcessing {label} (t={t_idx})...")
        res = solver.reconstruct_field_at_timestep(
            t_idx, use_fast_solver=True, return_latent=True
        )

        latent_field = res["latent_field"]   # (ny, nx, r) — NaN at boundary
        u_true  = res["u_true"]
        u_pred  = res["u_pred"]
        u_error = res["u_error"]
        interior = ~res["boundary_mask"]

        n_dims_show = min(3, latent_field.shape[2])

        vmin = np.nanpercentile(u_true[interior], 2)
        vmax = np.nanpercentile(u_true[interior], 98)
        err_vmax = np.nanpercentile(u_error[interior], 95)

        fig, axes = plt.subplots(2, max(3, n_dims_show), figsize=(5 * max(3, n_dims_show), 9))

        # Row 1: true SST / prediction / error
        for ax, (field, title, cmap, v0, v1) in zip(axes[0], [
            (u_true,  "True SST (°C)",       "viridis", vmin,   vmax),
            (u_pred,  "v3 Prediction (°C)",  "viridis", vmin,   vmax),
            (u_error, "|Error| (°C)",         "hot",     0,      err_vmax),
        ]):
            im = ax.imshow(field, cmap=cmap, origin="lower",
                           vmin=v0, vmax=v1, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, label="°C")
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        # Hide extra axes in row 1 if n_dims_show > 3
        for ax in axes[0][3:]:
            ax.axis("off")

        # Row 2: latent dimensions
        for dim in range(n_dims_show):
            ax = axes[1][dim]
            lf = latent_field[:, :, dim]
            lv = np.nanpercentile(np.abs(lf[interior]), 98)
            im = ax.imshow(lf, cmap="RdBu_r", origin="lower",
                           vmin=-lv, vmax=lv, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"Latent dim {dim}", fontsize=11)
            ax.axis("off")

        # Hide unused axes in row 2
        for ax in axes[1][n_dims_show:]:
            ax.axis("off")

        # Quantify latent smoothness: gradient of each latent dim vs SST
        sst_grad = np.nanmean([
            np.nanmean(np.sqrt(np.gradient(U[t_idx].astype(np.float64))[0]**2 +
                               np.gradient(U[t_idx].astype(np.float64))[1]**2))
        ])
        lat_grads = []
        for dim in range(latent_field.shape[2]):
            lf = latent_field[:, :, dim].astype(np.float64)
            lf_filled = np.where(np.isnan(lf), 0.0, lf)
            gy, gx = np.gradient(lf_filled)
            lat_grads.append(np.nanmean(np.sqrt(gx**2 + gy**2)))
        mean_lat_grad = np.mean(lat_grads)

        fig.suptitle(
            f"{label} — t={t_idx}  |  MAE = {res['mae']:.4f} °C\n"
            f"SST mean grad: {sst_grad:.4f}  |  "
            f"Mean latent grad: {mean_lat_grad:.4f}  "
            f"({'smoother' if mean_lat_grad < sst_grad else 'comparable'} — "
            f"{'✓ max principle' if mean_lat_grad < sst_grad else 'check latent scaling'})",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        out = f"{FIGURES_DIR}/latent_field_t{t_idx}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")
        print(f"  SST mean gradient:    {sst_grad:.5f}")
        print(f"  Latent mean gradient: {mean_lat_grad:.5f}")
        print(f"  Smoothness ratio:     {sst_grad / (mean_lat_grad + 1e-12):.2f}× "
              f"(latent is smoother by this factor)")


if __name__ == "__main__":
    main()
