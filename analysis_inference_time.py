"""
Inference time measurement.

Times reconstruction for Stage 1 (v3, v8 bypass) and combined
Stage 1 + Stage 2 (twostage). Reports mean ± std per timestep.

Results go in the methodology chapter.

Usage:
    python analysis_inference_time.py
"""

import time
import numpy as np

from config import (
    DEFAULT_SEED, RESULTS_DIR, SINN_V3_MODULE, SINN_V8_MODULE,
    B_THICK, ensure_dirs,
)
from utils import load_sinn_class, load_data, setup_sinn, load_sinn_weights
from cnn_corrector import ResidualCorrectorCNN

try:
    from utils import load_cnn_weights
except ImportError:
    load_cnn_weights = None

ensure_dirs()

N_TIMING_STEPS = 30   # enough for stable mean


def time_model(solver, test_indices, n_steps, cnn=None, bnd_mask=None, int_mask=None):
    times_s1 = []
    times_combined = []
    indices = test_indices[np.linspace(0, len(test_indices)-1, n_steps, dtype=int)]

    for t in indices:
        t0 = time.perf_counter()
        res = solver.reconstruct_field_at_timestep(int(t), use_fast_solver=True)
        t1 = time.perf_counter()
        times_s1.append(t1 - t0)

        if cnn is not None:
            u_pred_s1 = res["u_pred"]
            u_true    = res["u_true"]
            bnd_vals  = u_true * bnd_mask.astype(np.float32)
            t2 = time.perf_counter()
            correction      = cnn.predict_correction(u_pred_s1, bnd_vals, bnd_mask.astype(np.float32))
            u_pred_combined = u_pred_s1 + correction
            t3 = time.perf_counter()
            times_combined.append((t1 - t0) + (t3 - t2))

    return np.array(times_s1), np.array(times_combined) if times_combined else None


def main():
    data = load_data()

    print("Loading models...")
    # v3 baseline
    sinn_class_v3 = load_sinn_class(SINN_V3_MODULE)
    solver_v3 = setup_sinn(sinn_class_v3, data, seed=DEFAULT_SEED)
    load_sinn_weights(solver_v3, f"v3_seed{DEFAULT_SEED}")

    # v8 bypass (stage 1 of twostage)
    sinn_class_v8 = load_sinn_class(SINN_V8_MODULE)
    solver_v8 = setup_sinn(sinn_class_v8, data, seed=DEFAULT_SEED)
    try:
        load_sinn_weights(solver_v8, f"twostage_s1_seed{DEFAULT_SEED}")
    except Exception:
        load_sinn_weights(solver_v8, f"v8_seed{DEFAULT_SEED}")

    # CNN corrector
    ny, nx = solver_v8.ny, solver_v8.nx
    obs_mask = np.zeros((ny, nx), dtype=bool)
    obs_mask[:B_THICK, :] = True; obs_mask[-B_THICK:, :] = True
    obs_mask[:, :B_THICK] = True; obs_mask[:, -B_THICK:] = True
    interior_mask = ~obs_mask

    cnn = ResidualCorrectorCNN(ny, nx, interior_mask)
    try:
        from utils import load_cnn_weights
        load_cnn_weights(cnn, f"twostage_s2_seed{DEFAULT_SEED}")
        has_cnn = True
    except Exception:
        has_cnn = False
        print("[warn] CNN weights not found — timing Stage 1 only")

    test_indices = solver_v3.test_time_indices

    print(f"\nTiming over {N_TIMING_STEPS} test timesteps...")

    t_v3, _  = time_model(solver_v3, test_indices, N_TIMING_STEPS)
    t_v8, t_ts = time_model(
        solver_v8, test_indices, N_TIMING_STEPS,
        cnn=(cnn if has_cnn else None),
        bnd_mask=obs_mask, int_mask=interior_mask
    )

    print(f"\n{'='*55}")
    print("Inference Time Summary")
    print(f"{'='*55}")
    print(f"{'Model':<35} {'Mean (s)':>8} {'Std (s)':>8} {'per day'}")
    print(f"{'-'*55}")
    print(f"  v3 Baseline SINN           {np.mean(t_v3):>8.3f} {np.std(t_v3):>8.3f}")
    print(f"  v8 Bypass SINN (Stage 1)   {np.mean(t_v8):>8.3f} {np.std(t_v8):>8.3f}")
    if t_ts is not None:
        print(f"  Two-Stage (S1 + CNN)       {np.mean(t_ts):>8.3f} {np.std(t_ts):>8.3f}")
        print(f"  CNN overhead               {np.mean(t_ts - t_v8):>8.3f} {np.std(t_ts - t_v8):>8.3f}")

    print(f"\n  Note: timings on {'CPU' if True else 'GPU'}, "
          f"grid size {ny}×{nx} = {ny*nx:,} pixels")

    import json, os
    timing = {
        "v3_mean": float(np.mean(t_v3)), "v3_std": float(np.std(t_v3)),
        "v8_mean": float(np.mean(t_v8)), "v8_std": float(np.std(t_v8)),
        "n_steps": N_TIMING_STEPS,
        "grid": (ny, nx),
    }
    if t_ts is not None:
        timing["twostage_mean"] = float(np.mean(t_ts))
        timing["twostage_std"]  = float(np.std(t_ts))
    with open(os.path.join(RESULTS_DIR, "inference_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(f"\nSaved inference_timing.json")


if __name__ == "__main__":
    main()
