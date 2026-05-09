"""
Train vs test reconstruction comparison.

Loads the v3 baseline SINN and evaluates it on both training timesteps
(which the model has seen) and test timesteps. Produces:
  1. A 2×2 figure: train smooth / train complex / test smooth / test complex
  2. Printed summary of mean train MAE vs mean test MAE

This is the single most informative diagnostic for distinguishing
generalisation failure from structural (maximum principle) failure.
If oversmoothing appears even on training data in the complex region,
the bottleneck is the elliptic system, not insufficient training.

Section 3.3 of thesis.

Usage:
    python analysis_train_reconstruction.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

from config import (
    DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR, SINN_V3_MODULE,
    N_DAYS_2022, B_THICK, ensure_dirs,
)
from utils import load_sinn_class, load_data, setup_sinn, load_sinn_weights, load_results

ensure_dirs()


def find_smooth_and_complex_timesteps(U, time_indices, n_pick=2):
    """
    Among given time_indices, find timesteps with lowest and highest
    spatial SST gradient magnitude (smooth vs complex).
    Returns (smooth_indices, complex_indices) each of length n_pick.
    """
    grad_mags = []
    for t in time_indices:
        gy, gx = np.gradient(U[t].astype(np.float64))
        grad_mags.append(np.nanmean(np.sqrt(gx**2 + gy**2)))
    grad_mags = np.array(grad_mags)
    order = np.argsort(grad_mags)
    smooth = [time_indices[i] for i in order[:n_pick]]
    complex_ = [time_indices[i] for i in order[-n_pick:]]
    return smooth, complex_


def main():
    data = load_data()
    U = np.asarray(data["U"], dtype=np.float32)
    nt, ny, nx = U.shape

    # Load existing test results for test MAE (avoids re-running full test set)
    try:
        test_data = load_results(f"test_results_v3_seed{DEFAULT_SEED}.pkl")
        test_mae_mean = float(np.mean([r["mae"] for r in test_data["results"]]))
        print(f"[info] Loaded test results — mean test MAE: {test_mae_mean:.4f} °C")
    except FileNotFoundError:
        test_data = None
        test_mae_mean = None
        print("[warn] No test results pkl found — will compute from scratch")

    # Load v3 model
    sinn_class = load_sinn_class(SINN_V3_MODULE)
    solver = setup_sinn(sinn_class, data, seed=DEFAULT_SEED)
    load_sinn_weights(solver, f"v3_seed{DEFAULT_SEED}")
    print("[info] v3 weights loaded")

    # ---- Sample training timesteps (spread evenly across 2022) ----
    train_indices = solver.train_time_indices  # days 0–364
    sample_train = train_indices[np.linspace(0, len(train_indices)-1, 20, dtype=int)]

    # ---- Sample test timesteps ----
    sample_test = solver.test_time_indices[
        np.linspace(0, len(solver.test_time_indices)-1, 20, dtype=int)
    ]

    # ---- Find smooth / complex examples ----
    train_smooth, train_complex = find_smooth_and_complex_timesteps(U, sample_train, n_pick=1)
    test_smooth,  test_complex  = find_smooth_and_complex_timesteps(U, sample_test,  n_pick=1)

    # ---- Reconstruct the four chosen timesteps ----
    cases = [
        ("Train — smooth region",   train_smooth[0],  "train"),
        ("Train — complex region",  train_complex[0], "train"),
        ("Test — smooth region",    test_smooth[0],   "test"),
        ("Test — complex region",   test_complex[0],  "test"),
    ]

    results = []
    for label, t_idx, split in cases:
        res = solver.reconstruct_field_at_timestep(int(t_idx), use_fast_solver=True)
        results.append((label, t_idx, split, res))
        print(f"  {label:35s}  t={t_idx:4d}  MAE={res['mae']:.4f} °C")

    # ---- Compute mean train MAE across all sample train timesteps ----
    print("\nComputing mean train MAE over sample...")
    train_maes = []
    for t_idx in sample_train:
        r = solver.reconstruct_field_at_timestep(int(t_idx), use_fast_solver=True)
        train_maes.append(r["mae"])
    train_mae_mean = float(np.mean(train_maes))

    if test_mae_mean is None:
        print("Computing mean test MAE over sample...")
        test_maes = []
        for t_idx in sample_test:
            r = solver.reconstruct_field_at_timestep(int(t_idx), use_fast_solver=True)
            test_maes.append(r["mae"])
        test_mae_mean = float(np.mean(test_maes))

    print(f"\n{'='*50}")
    print(f"  Mean train MAE (2022):      {train_mae_mean:.4f} °C")
    print(f"  Mean test  MAE (2023+2024): {test_mae_mean:.4f} °C")
    ratio = test_mae_mean / train_mae_mean if train_mae_mean > 0 else float("inf")
    print(f"  Test/train MAE ratio:       {ratio:.2f}x")
    if ratio > 1.5:
        print("  → Significant generalisation gap — encoder cannot extrapolate")
    else:
        print("  → Small generalisation gap — failure is structural, not generalisation")
    print(f"{'='*50}\n")

    # ---- 2×2 figure ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    interior = ~results[0][3]["boundary_mask"]
    vmin_global = min(np.nanpercentile(r["u_true"][interior], 2)  for _, _, _, r in results)
    vmax_global = max(np.nanpercentile(r["u_true"][interior], 98) for _, _, _, r in results)
    err_vmax    = max(np.nanpercentile(r["u_error"][interior], 95) for _, _, _, r in results)

    positions = [(0,0), (0,1), (1,0), (1,1)]
    colours   = {"train": "#4daf4a", "test": "#e41a1c"}

    for ax, (pos, (label, t_idx, split, res)) in zip(axes.flat, zip(positions, results)):
        # Show absolute error (most informative)
        im = ax.imshow(res["u_error"], cmap="hot", origin="lower",
                       vmin=0, vmax=err_vmax, aspect="auto")
        fig.colorbar(im, ax=ax, label="|Error| (°C)", fraction=0.046)
        title_col = colours[split]
        ax.set_title(f"{label}\nt={t_idx}  MAE={res['mae']:.3f} °C",
                     fontsize=11, color=title_col, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"Train vs Test Reconstruction — v3 Baseline\n"
        f"Mean train MAE: {train_mae_mean:.3f} °C  |  Mean test MAE: {test_mae_mean:.3f} °C  "
        f"  (ratio {ratio:.2f}×)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out = f"{FIGURES_DIR}/train_vs_test_reconstruction.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # ---- Also save a 3-panel comparison (true / pred / error) for the complex train case ----
    label, t_idx, split, res = results[1]  # train complex
    interior_mask = ~res["boundary_mask"]
    vmin = np.nanpercentile(res["u_true"][interior_mask], 2)
    vmax = np.nanpercentile(res["u_true"][interior_mask], 98)

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (field, title) in zip(axes2, [
        (res["u_true"],  "True SST"),
        (res["u_pred"],  "v3 Prediction"),
        (res["u_error"], "|Error|"),
    ]):
        cmap = "viridis" if title != "|Error|" else "hot"
        v0, v1 = (vmin, vmax) if title != "|Error|" else (0, err_vmax)
        im = ax.imshow(field, cmap=cmap, origin="lower", vmin=v0, vmax=v1, aspect="auto")
        fig2.colorbar(im, ax=ax, label="°C", fraction=0.046)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig2.suptitle(
        f"Training Timestep — Complex Region (t={t_idx}, Day {t_idx+1} of 2022)\n"
        f"MAE = {res['mae']:.4f} °C — oversmoothing on seen data → maximum principle",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out2 = f"{FIGURES_DIR}/train_complex_reconstruction.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
