"""
Central configuration for all SINN experiments.
Single source of truth for paths, hyperparameters, and dataset split.
Edit paths here for your machine, then every script imports from this file.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# PATHS — relative to this config file so they resolve regardless of cwd
# =============================================================================
DATA_PATH = os.path.join(_THIS_DIR, "sst_2022_2023_2024_combined.pkl")
SINN_V3_MODULE = os.path.join(_THIS_DIR, "v3_baseline_sinn.py")
SINN_V8_MODULE = os.path.join(_THIS_DIR, "v8_bypass_sinn.py")

# Output directories (created automatically)
WEIGHTS_DIR = os.path.join(_THIS_DIR, "weights")
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
FIGURES_DIR = os.path.join(_THIS_DIR, "figures")

# =============================================================================
# DATASET SPLIT — year-based
# =============================================================================
N_DAYS_2022 = 365
N_DAYS_2023 = 365
N_DAYS_2024 = 366  # leap year
N_TOTAL = N_DAYS_2022 + N_DAYS_2023 + N_DAYS_2024  # 1096

# Train: 2022 (days 0–364)    Test: 2023+2024 (days 365–1095)
TRAIN_START = 0
TRAIN_END = N_DAYS_2022          # exclusive
TEST_START = N_DAYS_2022
TEST_END = N_TOTAL               # exclusive

# ENSO labels for figures
YEAR_LABELS = {2022: "La Niña (train)", 2023: "El Niño (test)", 2024: "Neutral (test)"}

# =============================================================================
# SINN HYPERPARAMETERS
# =============================================================================
B_THICK = 1
INCLUDE_T0 = True
INCLUDE_TT = True
NUM_LATENTDIM = 10
NUM_UNITS = 128
NUM_LAYERS = 3
DROPOUT = 0.0
L2_REG = 1e-5
LR = 1e-3
PATCH_DIM = [10, 10, 10]
NUM_PATCHES = 100
N_PAST_STEPS = 5
MASK_RADIUS = 1

# Training
STAGE1_EPOCHS = 75
STAGE2_EPOCHS = 200
STAGE2_BATCH_SIZE = 8

# =============================================================================
# MULTI-SEED
# =============================================================================
SEEDS = [42, 123, 456]
DEFAULT_SEED = 42

# =============================================================================
# ABLATION — latent dimensions to test
# =============================================================================
ABLATION_LATENT_DIMS = [1, 3, 5, 10, 15]

# =============================================================================
# NUMERICAL CONSTANTS — single source of truth (was scattered as magic numbers)
# =============================================================================
# Floor for any std-divisor (avoid div-by-zero in standardisation, normalisation).
EPS_STD = 1e-8
# Floor for Cholesky / eigenvalue safety (a_chol_raw, A-matrix eigvals, etc.).
EPS_CHOL = 1e-6
# Eigenvalue clip threshold for elliptic eigendecomp solves.
EPS_EIG = 1e-8
# Replacement clamp for nan/inf in latent encodings and RHS vectors.
NAN_CLAMP = 1e3
# Posinf/neginf clamp during U standardisation (raw values can have outliers).
INF_CLAMP_U = 1e6
# Gradient clip global L2 norm.
GRAD_CLIP_NORM = 1.0
# CG tolerances at inference time.
CG_TOL_FAST = 1e-2
CG_MAXITER_FAST = 1000
CG_TOL_ACCURATE = 1e-4
CG_MAXITER_ACCURATE = 5000

# =============================================================================
# HELPER
# =============================================================================
def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [WEIGHTS_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
