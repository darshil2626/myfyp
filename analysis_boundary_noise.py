"""
Boundary noise robustness experiment.

Adds Gaussian noise to boundary SST values at inference time and measures
how MAE degrades for both the v3 SINN and the two-stage model. Addresses
the "perfect boundary assumption" — quantifies how sensitive the model is
to boundary measurement error.

σ tested: 0.0, 0.1, 0.25, 0.5, 1.0 °C

Produces:
  - figures/boundary_noise_robustness.png

Usage:
    python analysis_boundary_noise.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle, os

from config import (
    DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR,
    SINN_V3_MODULE, SINN_V8_MODULE, B_THICK, ensure_dirs,
)
from utils import load_sinn_class, load_data, setup_sinn, load_sinn_weights, load_results

ensure_dirs()

NOISE_SIGMAS = [0.0, 0.1, 0.25, 0.5, 1.0]
N_EVAL_STEPS = 50    # subset of test steps to keep runtime manageable


def eval_with_noise(solver, test_indices, sigma, n_steps):
    maes = []
    for t in test_indices[:n_steps]:
        res = solver.reconstruct_field_at_timestep(
            int(t), use_fast_solver=True, boundary_noise_sigma=sigma
        )
        maes.append(res["mae"])
    return float(np.mean(maes))


def main():
    data = load_data()

    # ---- v3 baseline ----
    sinn_class_v3 = load_sinn_class(SINN_V3_MODULE)
    solver_v3 = setup_sinn(sinn_class_v3, data, seed=DEFAULT_SEED)
    load_sinn_weights(solver_v3, f"v3_seed{DEFAULT_SEED}")
    print("[info] v3 weights loaded")

    # ---- v8 bypass (Stage 1 of twostage) ----
    sinn_class_v8 = load_sinn_class(SINN_V8_MODULE)
    solver_v8 = setup_sinn(sinn_class_v8, data, seed=DEFAULT_SEED)
    # Load twostage Stage 1 weights (same as v8 bypass weights)
    try:
        load_sinn_weights(solver_v8, f"twostage_s1_seed{DEFAULT_SEED}")
        print("[info] twostage s1 weights loaded for v8")
    except Exception:
        load_sinn_weights(solver_v8, f"v8_seed{DEFAULT_SEED}")
        print("[info] v8 weights loaded (twostage s1 not found)")

    test_indices = solver_v3.test_time_indices
    # Spread evaluation evenly across the test period
    eval_indices = test_indices[np.linspace(0, len(test_indices)-1, N_EVAL_STEPS, dtype=int)]

    results = {"v3": {}, "v8_bypass": {}}

    for sigma in NOISE_SIGMAS:
        print(f"\nσ = {sigma:.2f} °C:")
        mae_v3 = eval_with_noise(solver_v3, eval_indices, sigma, N_EVAL_STEPS)
        mae_v8 = eval_with_noise(solver_v8, eval_indices, sigma, N_EVAL_STEPS)
        results["v3"][sigma] = mae_v3
        results["v8_bypass"][sigma] = mae_v8
        print(f"  v3 baseline:  MAE = {mae_v3:.4f} °C")
        print(f"  v8 bypass:    MAE = {mae_v8:.4f} °C")

    # Save results
    out_pkl = os.path.join(RESULTS_DIR, "boundary_noise_results.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {out_pkl}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colours = {"v3": "#e41a1c", "v8_bypass": "#377eb8"}
    labels  = {"v3": "v3 Baseline SINN", "v8_bypass": "v8 Bypass SINN (Stage 1)"}

    # Absolute MAE vs sigma
    ax = axes[0]
    for model in ["v3", "v8_bypass"]:
        maes = [results[model][s] for s in NOISE_SIGMAS]
        ax.plot(NOISE_SIGMAS, maes, "o-", color=colours[model],
                lw=2, markersize=8, label=labels[model])
    ax.set_xlabel("Boundary noise σ (°C)", fontsize=12)
    ax.set_ylabel("Mean MAE (°C)", fontsize=12)
    ax.set_title("MAE vs Boundary Noise Level", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Relative degradation (normalised to σ=0)
    ax = axes[1]
    for model in ["v3", "v8_bypass"]:
        maes = [results[model][s] for s in NOISE_SIGMAS]
        base = maes[0]
        rel  = [m / base for m in maes]
        ax.plot(NOISE_SIGMAS, rel, "o-", color=colours[model],
                lw=2, markersize=8, label=labels[model])
    ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Boundary noise σ (°C)", fontsize=12)
    ax.set_ylabel("Relative MAE (normalised to σ=0)", fontsize=12)
    ax.set_title("Relative MAE Degradation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Boundary Noise Robustness — How Sensitive Is the Model to Boundary Errors?",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out_fig = f"{FIGURES_DIR}/boundary_noise_robustness.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_fig}")

    # Summary
    print(f"\n{'='*50}")
    print("Summary at σ=0.5 °C (moderate noise):")
    for model in ["v3", "v8_bypass"]:
        base = results[model][0.0]
        noisy = results[model][0.5]
        print(f"  {labels[model]}: {base:.4f} → {noisy:.4f} °C "
              f"(+{(noisy-base):.4f}, ×{noisy/base:.2f})")


if __name__ == "__main__":
    main()
