"""
Utility functions shared across all runner and analysis scripts.
Handles: seed control, weight save/load, dataset split, class loading, config saving.

Reconstruction error metrics live in metrics.py.
"""

import os
import random
import json
import pickle
import importlib.util
import numpy as np
import tensorflow as tf

from config import (
    WEIGHTS_DIR, RESULTS_DIR, TRAIN_START, TRAIN_END, TEST_START, TEST_END,
    MASK_RADIUS,
)


# =============================================================================
# SEED CONTROL
# =============================================================================
def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[seed] All seeds set to {seed}")


# =============================================================================
# DATASET SPLIT
# =============================================================================
def apply_year_split(solver):
    """
    Override the solver's train/test split with year-based indices.
    Train: 2022 (indices 0..364)
    Test:  2023+2024 (indices 365..1095)
    """
    solver.train_time_indices = np.arange(TRAIN_START, TRAIN_END, dtype=np.int32)
    solver.test_time_indices = np.arange(TEST_START, TEST_END, dtype=np.int32)
    print(f"[split] Year-based split applied")
    print(f"  Train: {len(solver.train_time_indices)} days "
          f"(t={solver.train_time_indices[0]}..{solver.train_time_indices[-1]})")
    print(f"  Test:  {len(solver.test_time_indices)} days "
          f"(t={solver.test_time_indices[0]}..{solver.test_time_indices[-1]})")


# =============================================================================
# SINN CLASS LOADER
# =============================================================================
def load_sinn_class(module_path: str):
    """Import the sinn class from a .py file."""
    name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sinn


# =============================================================================
# WEIGHT SAVE / LOAD - works for both v3 and v8
# =============================================================================
def save_sinn_weights(solver, prefix: str, directory: str = None):
    """Save all SINN trainable weights to disk.

    Args:
        solver: trained sinn instance (v3 or v8)
        prefix: e.g. "v3_seed42" - files saved as {prefix}_interior_encoder.weights.h5 etc.
        directory: folder to save in (defaults to WEIGHTS_DIR)
    """
    d = directory or WEIGHTS_DIR
    os.makedirs(d, exist_ok=True)

    solver.interior_encoder.save_weights(os.path.join(d, f"{prefix}_interior_encoder.weights.h5"))
    solver.boundary_encoder.save_weights(os.path.join(d, f"{prefix}_boundary_encoder.weights.h5"))
    solver.decoder.save_weights(os.path.join(d, f"{prefix}_decoder.weights.h5"))
    np.save(os.path.join(d, f"{prefix}_a_chol_raw.npy"), solver.a_chol_raw.numpy())
    print(f"[save] SINN weights saved with prefix '{prefix}' to {d}/")


def load_sinn_weights(solver, prefix: str, directory: str = None):
    """Load SINN weights from disk. The solver must already have build_models() called.

    Args:
        solver: sinn instance with models built but untrained
        prefix: same prefix used in save_sinn_weights
        directory: folder to load from
    """
    d = directory or WEIGHTS_DIR

    solver.interior_encoder.load_weights(os.path.join(d, f"{prefix}_interior_encoder.weights.h5"))
    solver.boundary_encoder.load_weights(os.path.join(d, f"{prefix}_boundary_encoder.weights.h5"))
    solver.decoder.load_weights(os.path.join(d, f"{prefix}_decoder.weights.h5"))
    a_chol = np.load(os.path.join(d, f"{prefix}_a_chol_raw.npy"))
    solver.a_chol_raw.assign(a_chol)
    print(f"[load] SINN weights loaded from prefix '{prefix}' in {d}/")


def save_cnn_weights(corrector, prefix: str, directory: str = None):
    """Save CNN corrector weights and normalisation stats."""
    d = directory or WEIGHTS_DIR
    os.makedirs(d, exist_ok=True)

    corrector.model.save_weights(os.path.join(d, f"{prefix}_cnn.weights.h5"))
    stats = {
        "x_mean_ch0": corrector.x_mean_ch0,
        "x_std_ch0": corrector.x_std_ch0,
        "x_mean_ch1": corrector.x_mean_ch1,
        "x_std_ch1": corrector.x_std_ch1,
        "y_mean": corrector.y_mean,
        "y_std": corrector.y_std,
    }
    with open(os.path.join(d, f"{prefix}_cnn_stats.json"), "w") as f:
        json.dump(stats, f)
    print(f"[save] CNN weights + stats saved with prefix '{prefix}' to {d}/")


def load_cnn_weights(corrector, prefix: str, directory: str = None):
    """Load CNN corrector weights and normalisation stats."""
    d = directory or WEIGHTS_DIR

    corrector.model.load_weights(os.path.join(d, f"{prefix}_cnn.weights.h5"))
    with open(os.path.join(d, f"{prefix}_cnn_stats.json"), "r") as f:
        stats = json.load(f)
    corrector.x_mean_ch0 = stats["x_mean_ch0"]
    corrector.x_std_ch0 = stats["x_std_ch0"]
    corrector.x_mean_ch1 = stats["x_mean_ch1"]
    corrector.x_std_ch1 = stats["x_std_ch1"]
    corrector.y_mean = stats["y_mean"]
    corrector.y_std = stats["y_std"]
    print(f"[load] CNN weights + stats loaded from prefix '{prefix}' in {d}/")


# =============================================================================
# RESULTS SAVE / LOAD
# =============================================================================
def save_results(payload: dict, filename: str, directory: str = None):
    """Save results dict as pickle."""
    d = directory or RESULTS_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, filename)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[save] Results saved to {path}")


def load_results(filename: str, directory: str = None) -> dict:
    """Load results dict from pickle."""
    d = directory or RESULTS_DIR
    path = os.path.join(d, filename)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[load] Results loaded from {path}")
    return data


# =============================================================================
# CONFIG SNAPSHOT
# =============================================================================
def save_config_snapshot(extra: dict = None, filename: str = "config_snapshot.json",
                          directory: str = None):
    """Save a JSON snapshot of all config values used in this run."""
    import config as cfg

    snapshot = {
        "DATA_PATH": cfg.DATA_PATH,
        "N_DAYS": {"2022": cfg.N_DAYS_2022, "2023": cfg.N_DAYS_2023, "2024": cfg.N_DAYS_2024},
        "TRAIN_RANGE": [cfg.TRAIN_START, cfg.TRAIN_END],
        "TEST_RANGE": [cfg.TEST_START, cfg.TEST_END],
        "B_THICK": cfg.B_THICK,
        "INCLUDE_T0": cfg.INCLUDE_T0,
        "INCLUDE_TT": cfg.INCLUDE_TT,
        "MASK_RADIUS": cfg.MASK_RADIUS,
        "NUM_LATENTDIM": cfg.NUM_LATENTDIM,
        "NUM_UNITS": cfg.NUM_UNITS,
        "NUM_LAYERS": cfg.NUM_LAYERS,
        "DROPOUT": cfg.DROPOUT,
        "L2_REG": cfg.L2_REG,
        "LR": cfg.LR,
        "PATCH_DIM": cfg.PATCH_DIM,
        "NUM_PATCHES": cfg.NUM_PATCHES,
        "N_PAST_STEPS": cfg.N_PAST_STEPS,
        "STAGE1_EPOCHS": cfg.STAGE1_EPOCHS,
        "STAGE2_EPOCHS": cfg.STAGE2_EPOCHS,
        "STAGE2_BATCH_SIZE": cfg.STAGE2_BATCH_SIZE,
        "EPS_STD": cfg.EPS_STD,
        "EPS_CHOL": cfg.EPS_CHOL,
        "EPS_EIG": cfg.EPS_EIG,
        "NAN_CLAMP": cfg.NAN_CLAMP,
        "INF_CLAMP_U": cfg.INF_CLAMP_U,
        "GRAD_CLIP_NORM": cfg.GRAD_CLIP_NORM,
        "CG_TOL_FAST": cfg.CG_TOL_FAST,
        "CG_MAXITER_FAST": cfg.CG_MAXITER_FAST,
        "CG_TOL_ACCURATE": cfg.CG_TOL_ACCURATE,
        "CG_MAXITER_ACCURATE": cfg.CG_MAXITER_ACCURATE,
    }
    if extra:
        snapshot.update(extra)

    d = directory or RESULTS_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, filename)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[save] Config snapshot saved to {path}")


# =============================================================================
# SETUP HELPER - standard SINN initialisation sequence
# =============================================================================
def setup_sinn(sinn_class, data, seed=None, num_latentdim=None):
    """Standard initialisation: load data -> split -> standardise -> build models.
    Returns a solver ready for training or weight loading.

    Args:
        sinn_class: the sinn class (from load_sinn_class)
        data: dict with keys X, Y, U, T
        seed: random seed (set before building)
        num_latentdim: override latent dims (for ablation)
    """
    import config as cfg

    if seed is not None:
        set_seed(seed)

    X, Y, U, T = data["X"], data["Y"], data["U"], data["T"]
    solver = sinn_class(X, Y, U, T, debug=False)

    apply_year_split(solver)
    solver.standardise_u(time_indices=solver.train_time_indices)
    solver.split_interior_boundary(cfg.B_THICK, cfg.INCLUDE_T0, cfg.INCLUDE_TT)

    r = num_latentdim if num_latentdim is not None else cfg.NUM_LATENTDIM
    solver.build_models(
        num_latentdim=r,
        num_units=cfg.NUM_UNITS,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        l2_reg=cfg.L2_REG,
        lr=cfg.LR,
        n_past_steps=cfg.N_PAST_STEPS,
    )

    return solver


def load_data():
    """Load the 3-year SST pickle."""
    import config as cfg
    with open(cfg.DATA_PATH, "rb") as f:
        data = pickle.load(f)
    U = np.asarray(data["U"])
    print(f"[data] Loaded SST data: U shape = {U.shape}")
    assert U.shape[0] == cfg.N_TOTAL, (
        f"Expected {cfg.N_TOTAL} timesteps, got {U.shape[0]}"
    )
    return data
