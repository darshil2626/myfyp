"""
Small domain experiment — supervisor feedback Version A.

Crops the SST domain to a sub-region centred on the high-error area,
retrains the v3 baseline SINN on this smaller domain, and compares
performance in the complex region between the full-domain and
cropped-domain models.

If the cropped model performs significantly better in the complex region,
the full-domain model's failure was partly due to training dynamics
(the complex region was too small to influence the loss). If performance
is similar, the limitation is structural (maximum principle).

Usage:
    python run_small_domain.py
    python run_small_domain.py --crop-size 80   # change crop size
    python run_small_domain.py --load            # skip training
"""

import argparse
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
from utils import (
    set_seed, load_sinn_class, load_data,
    apply_year_split, save_sinn_weights, load_sinn_weights,
    save_results,
)

ensure_dirs()


def find_high_error_centre(error_field, obs_mask):
    """Find the centre of the highest-error interior region."""
    from scipy.ndimage import uniform_filter
    err = error_field.copy()
    err[obs_mask] = 0.0
    err_smooth = uniform_filter(err, size=15)
    err_smooth[obs_mask] = 0.0
    peak = np.unravel_index(np.argmax(err_smooth), err_smooth.shape)
    return int(peak[0]), int(peak[1])


def crop_data(data, y_start, y_end, x_start, x_end):
    """Crop the SST data dict to a spatial sub-region."""
    U = np.asarray(data["U"], dtype=np.float32)
    X = np.asarray(data["X"], dtype=np.float32)
    Y = np.asarray(data["Y"], dtype=np.float32)
    T = data["T"]

    U_crop = U[:, y_start:y_end, x_start:x_end].copy()
    X_crop = X[x_start:x_end].copy()
    Y_crop = Y[y_start:y_end].copy()

    return {"U": U_crop, "X": X_crop, "Y": Y_crop, "T": T}


def evaluate_region(solver, time_indices, region_y, region_x,
                    full_ny=None, full_nx=None, offset_y=0, offset_x=0):
    """
    Evaluate MAE only within a specific spatial region.
    region_y, region_x are in the solver's local coordinates.
    """
    maes = []
    for t in time_indices:
        res = solver.reconstruct_field_at_timestep(int(t), use_fast_solver=True)
        u_pred = res["u_pred"]
        u_true = res["u_true"]
        err = np.abs(u_pred[region_y, region_x] - u_true[region_y, region_x])
        maes.append(float(np.mean(err)))
    return maes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--y-start", type=int, default=115,  help="Crop y start index")
    parser.add_argument("--y-end",   type=int, default=140, help="Crop y end index")
    parser.add_argument("--x-start", type=int, default=50,  help="Crop x start index")
    parser.add_argument("--x-end",   type=int, default=175, help="Crop x end index")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    y_start, y_end = args.y_start, args.y_end
    x_start, x_end = args.x_start, args.x_end
    crop_sz = max(y_end - y_start, x_end - x_start)
    prefix = f"small_domain_{crop_sz}_seed{args.seed}"

    # ---- Step 1: Find the high-error region ----
    print("Step 1: Locating high-error region...")
    try:
        npz = np.load(f"{RESULTS_DIR}/spatial_error_fields.npz")
        s1_error_avg = npz["s1_error_avg"]
    except FileNotFoundError:
        print("ERROR: Run analysis_spatial_error.py first.")
        print("  Need spatial_error_fields.npz to locate the complex region.")
        return

    data = load_data()
    U_full = np.asarray(data["U"], dtype=np.float32)
    nt, ny_full, nx_full = U_full.shape

    obs_mask_full = np.zeros((ny_full, nx_full), dtype=bool)
    obs_mask_full[:B_THICK, :] = True; obs_mask_full[-B_THICK:, :] = True
    obs_mask_full[:, :B_THICK] = True; obs_mask_full[:, -B_THICK:] = True

    # ---- Step 2: Define crop bounds ----
    cy = (y_start + y_end) // 2
    cx = (x_start + x_end) // 2
    print(f"  Crop bounds: y=[{y_start}:{y_end}], x=[{x_start}:{x_end}]")

    # Ensure dimensions are even (for U-Net compatibility if needed later)
    actual_ny = y_end - y_start
    actual_nx = x_end - x_start
    if actual_ny % 2 != 0:
        y_end -= 1
        actual_ny -= 1
    if actual_nx % 2 != 0:
        x_end -= 1
        actual_nx -= 1

    print(f"  Crop region: y=[{y_start}:{y_end}], x=[{x_start}:{x_end}]")
    print(f"  Crop size: {actual_ny} × {actual_nx}")
    print(f"  Full domain: {ny_full} × {nx_full}")
    print(f"  Crop fraction: {(actual_ny * actual_nx) / (ny_full * nx_full) * 100:.1f}% of full domain")

    # ---- Step 3: Crop and train ----
    data_crop = crop_data(data, y_start, y_end, x_start, x_end)
    print(f"  Cropped U shape: {np.asarray(data_crop['U']).shape}")

    sinn_class = load_sinn_class(SINN_V3_MODULE)
    set_seed(args.seed)

    X_c, Y_c, U_c, T_c = data_crop["X"], data_crop["Y"], data_crop["U"], data_crop["T"]
    solver = sinn_class(X_c, Y_c, U_c, T_c, debug=False)
    apply_year_split(solver)
    solver.standardise_u(time_indices=solver.train_time_indices)
    solver.split_interior_boundary(B_THICK, INCLUDE_T0, INCLUDE_TT)

    # Use r=1 since that was optimal in the ablation
    solver.build_models(
        num_latentdim=1,
        num_units=NUM_UNITS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        l2_reg=L2_REG,
        lr=LR,
        n_past_steps=N_PAST_STEPS,
    )

    if args.load:
        load_sinn_weights(solver, prefix)
    else:
        print(f"\n{'='*60}")
        print(f"TRAINING v3 baseline on CROPPED domain ({actual_ny}×{actual_nx})")
        print(f"{'='*60}")

        # Adjust patch size if crop is smaller than default patch
        patch_dim = [min(p, d - 2) for p, d in
                     zip(PATCH_DIM, [nt, actual_ny, actual_nx])]
        patch_dim = [max(p, 4) for p in patch_dim]  # minimum 4
        print(f"  Adjusted patch_dim: {patch_dim}")

        loss_history = solver.train(STAGE1_EPOCHS, patch_dim, NUM_PATCHES)
        print(f"  Final loss: {loss_history['total'][-1]:.6e}")
        save_sinn_weights(solver, prefix)

    # ---- Step 4: Evaluate cropped model ----
    print("\nEvaluating cropped model on test set...")
    test_summary = solver.evaluate_on_test_timesteps()
    crop_mae = test_summary["mae"]
    crop_per_step = [r["mae"] for r in test_summary["per_step"]]
    print(f"  Cropped model MAE (full crop): {crop_mae:.4f} °C")

    # ---- Step 5: Load full-domain results for comparison ----
    print("\nLoading full-domain results for comparison...")

    # Try to load full-domain per-step results
    full_domain_results = None
    for pkl_name in [
        f"test_results_v3_seed{args.seed}.pkl",
        f"test_results_twostage_seed{args.seed}.pkl",
    ]:
        try:
            full_data = pickle.load(open(f"{RESULTS_DIR}/{pkl_name}", "rb"))
            full_domain_results = full_data["results"]
            print(f"  Loaded full-domain results from {pkl_name}")
            break
        except FileNotFoundError:
            continue

    # Compute full-domain MAE restricted to the SAME spatial region
    full_region_maes = None
    if full_domain_results is not None:
        full_region_maes = []
        for r in full_domain_results:
            u_pred = r["u_pred"]
            u_true = r["u_true"]
            # Extract the crop region from full-domain prediction
            u_pred_region = u_pred[y_start:y_end, x_start:x_end]
            u_true_region = u_true[y_start:y_end, x_start:x_end]
            # Interior of the crop (exclude boundary ring)
            int_mask_region = np.ones((actual_ny, actual_nx), dtype=bool)
            int_mask_region[:B_THICK, :] = False; int_mask_region[-B_THICK:, :] = False
            int_mask_region[:, :B_THICK] = False; int_mask_region[:, -B_THICK:] = False
            iy, ix = np.where(int_mask_region)
            mae = float(np.mean(np.abs(u_pred_region[iy, ix] - u_true_region[iy, ix])))
            full_region_maes.append(mae)

        print(f"  Full-domain model MAE (in crop region): {np.mean(full_region_maes):.4f} °C")
        print(f"  Cropped model MAE (in crop region):     {crop_mae:.4f} °C")
        diff = np.mean(full_region_maes) - crop_mae
        pct = diff / np.mean(full_region_maes) * 100
        print(f"  Difference: {diff:+.4f} °C ({pct:+.1f}%)")

    # ---- Step 6: Save results ----
    payload = {
        "crop_region": {"y_start": y_start, "y_end": y_end,
                        "x_start": x_start, "x_end": x_end},
        "crop_size": (actual_ny, actual_nx),
        "error_centre": (cy, cx),
        "crop_mae": crop_mae,
        "crop_per_step_mae": crop_per_step,
        "full_region_maes": full_region_maes,
        "full_region_mean_mae": float(np.mean(full_region_maes)) if full_region_maes else None,
        "seed": args.seed,
    }
    save_results(payload, f"small_domain_results_{crop_sz}.pkl")

    # ---- Step 7: Figures ----
    test_indices = np.arange(TEST_START, TEST_END, dtype=np.int32)

    # Figure 1: MAE over time comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_indices, crop_per_step, "b-", lw=1.5, alpha=0.8,
            label=f"Cropped domain {actual_ny}×{actual_nx} (MAE={crop_mae:.3f} °C)")
    if full_region_maes is not None:
        ax.plot(test_indices, full_region_maes, "r--", lw=1.5, alpha=0.7,
                label=f"Full domain (same region) (MAE={np.mean(full_region_maes):.3f} °C)")
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("MAE (°C)", fontsize=12)
    ax.set_title(f"Small Domain Experiment: {actual_ny}×{actual_nx} Crop vs Full {ny_full}×{nx_full} Domain\n"
                 f"Centred on high-error region (y={cy}, x={cx})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/small_domain_comparison_{crop_sz}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved small_domain_comparison_{crop_sz}.png")

    # Figure 2: Show the crop region on the error map
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    err_display = np.nan_to_num(s1_error_avg, nan=0.0)
    im = ax.imshow(err_display, cmap="hot", origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax, label="Time-averaged |error| (°C)")
    rect = plt.Rectangle((x_start, y_start), actual_nx, actual_ny,
                           linewidth=2, edgecolor="cyan", facecolor="none")
    ax.add_patch(rect)
    ax.plot(cx, cy, "x", color="cyan", markersize=15, markeredgewidth=3)
    ax.set_title("Full Domain Error with Crop Region", fontsize=12, fontweight="bold")

    # Show a representative field in the crop
    ax = axes[1]
    mid_t = TEST_START + (TEST_END - TEST_START) // 2
    im2 = ax.imshow(U_full[mid_t, y_start:y_end, x_start:x_end],
                     cmap="viridis", origin="lower", aspect="auto")
    fig.colorbar(im2, ax=ax, label="SST (°C)")
    ax.set_title(f"Cropped SST Field (t={mid_t})", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/small_domain_region_{crop_sz}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved small_domain_region_{crop_sz}.png")

    # ---- Interpretation guide ----
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    if full_region_maes is not None:
        full_mae = np.mean(full_region_maes)
        if crop_mae < full_mae * 0.85:
            print("  RESULT: Cropped model is significantly better (>15% improvement).")
            print("  → Supports TRAINING DYNAMICS hypothesis: the complex region was")
            print("    too small for the full-domain loss to prioritise.")
        elif crop_mae < full_mae * 0.95:
            print("  RESULT: Cropped model is modestly better (5-15% improvement).")
            print("  → Mixed evidence: both training dynamics and structural limitation")
            print("    likely contribute.")
        else:
            print("  RESULT: Cropped model performs similarly (<5% improvement).")
            print("  → Supports STRUCTURAL LIMITATION hypothesis: even with the complex")
            print("    region dominating the loss, the maximum principle still constrains")
            print("    reconstruction quality.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
