"""
A-matrix diagnostic analysis.

Compares the learned latent operator matrix A across model variants:
  - v3 baseline SINN
  - v8 bypass SINN
  - Ablation variants (different r)

Produces eigenvalue spectrum plots and Frobenius distance from identity.
This is the centrepiece of Section 4.2 in the thesis.

Usage:
    python analysis_a_matrix.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    ABLATION_LATENT_DIMS, DEFAULT_SEED, FIGURES_DIR, RESULTS_DIR, ensure_dirs,
)

ensure_dirs()


def analyse_A(A, label):
    """Print diagnostics for a single A matrix."""
    r = A.shape[0]
    eigvals = np.linalg.eigvalsh(A)
    frob_I = np.linalg.norm(A - np.eye(r))
    frob_scaled_I = np.linalg.norm(A - np.mean(eigvals) * np.eye(r))
    cond = eigvals[-1] / eigvals[0] if eigvals[0] > 0 else float("inf")

    print(f"\n  {label}:")
    print(f"    Eigenvalues: {eigvals}")
    print(f"    Mean eigenvalue: {np.mean(eigvals):.4f}")
    print(f"    Eigenvalue range: [{eigvals[0]:.4f}, {eigvals[-1]:.4f}]")
    print(f"    Condition number: {cond:.4f}")
    print(f"    ||A - I||_F: {frob_I:.4f}")
    print(f"    ||A - λ̄I||_F: {frob_scaled_I:.4f}")

    # Off-diagonal magnitude
    off_diag = A - np.diag(np.diag(A))
    off_mag = np.linalg.norm(off_diag, "fro")
    print(f"    Off-diagonal ||·||_F: {off_mag:.4f}")

    return eigvals


def main():
    print(f"{'='*60}")
    print("A-MATRIX DIAGNOSTIC ANALYSIS")
    print(f"{'='*60}")

    models = {}

    # Load available A matrices
    for label, filename in [
        ("v3 baseline", f"A_matrix_v3_seed{DEFAULT_SEED}.npy"),
        ("v8 bypass", f"A_matrix_v8_seed{DEFAULT_SEED}.npy"),
        ("twostage", f"A_matrix_twostage_seed{DEFAULT_SEED}.npy"),
    ]:
        try:
            A = np.load(f"{RESULTS_DIR}/{filename}")
            models[label] = A
        except FileNotFoundError:
            print(f"  Skipping {label} (not found)")

    # Load ablation A matrices
    for r in ABLATION_LATENT_DIMS:
        try:
            A = np.load(f"{RESULTS_DIR}/A_matrix_ablation_r{r}.npy")
            models[f"ablation r={r}"] = A
        except FileNotFoundError:
            pass

    if not models:
        print("No A matrices found. Run training scripts first.")
        return

    # Analyse each
    all_eigvals = {}
    for label, A in models.items():
        eigvals = analyse_A(A, label)
        all_eigvals[label] = eigvals

    # ---- Figure 1: Eigenvalue comparison (v3 vs v8) ----
    main_models = {k: v for k, v in all_eigvals.items()
                   if k in ["v3 baseline", "v8 bypass", "twostage"]}

    if main_models:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Bar chart of eigenvalues
        ax = axes[0]
        n_models = len(main_models)
        width = 0.8 / n_models
        colours = {"v3 baseline": "#e41a1c", "v8 bypass": "#377eb8", "twostage": "#4daf4a"}

        for i, (label, eigvals) in enumerate(main_models.items()):
            r = len(eigvals)
            x = np.arange(r) + i * width
            ax.bar(x, eigvals, width, label=label, color=colours.get(label, "gray"), alpha=0.8)

        ax.set_xlabel("Latent dimension index", fontsize=11)
        ax.set_ylabel("Eigenvalue of A", fontsize=11)
        ax.set_title("Eigenvalue Spectrum of Learned A Matrix", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Heatmap of A matrix (first available)
        ax = axes[1]
        first_label = list(main_models.keys())[0]
        A_show = models[first_label]
        im = ax.imshow(A_show, cmap="RdBu_r", aspect="equal",
                        vmin=-np.max(np.abs(A_show)), vmax=np.max(np.abs(A_show)))
        fig.colorbar(im, ax=ax)
        ax.set_title(f"A Matrix Heatmap ({first_label})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Latent dim"); ax.set_ylabel("Latent dim")

        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/a_matrix_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nSaved a_matrix_comparison.png")

    # ---- Figure 2: Ablation eigenvalues ----
    ablation_models = {k: v for k, v in all_eigvals.items() if k.startswith("ablation")}
    if ablation_models:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, eigvals in sorted(ablation_models.items()):
            r = len(eigvals)
            ax.plot(range(r), eigvals, "o-", label=label, markersize=6)

        ax.set_xlabel("Latent dimension index", fontsize=11)
        ax.set_ylabel("Eigenvalue", fontsize=11)
        ax.set_title("Eigenvalue Spectra Across Latent Dimensions", fontsize=12)
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/a_matrix_ablation.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved a_matrix_ablation.png")


if __name__ == "__main__":
    main()
