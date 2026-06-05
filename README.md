# SINN — Structured Implicit Neural Networks for SST Interpolation

**Author:** Darshil Gajjar  
**Degree:** Final Year Project (BEng/MEng), May 2026

---

## Overview

This project applies **Structured Implicit Neural Networks (SINNs)** to reconstruct interior **Sea Surface Temperature (SST)** fields across a South Pacific domain, given only boundary observations. The core idea is to embed an elliptic PDE constraint (`div(A ∇ℓ) = 0`) in a learned latent space, giving the network physically plausible inductive bias while remaining fully data-driven.

The study spans three years of daily SST data (2022–2024) that happen to cover three distinct ENSO phases (La Niña → El Niño → Neutral), creating a challenging temporal generalisation test: train on one climate regime, evaluate on two others.

### Models

| Name | Description |
|------|-------------|
| **Baseline SINN** | Encoder → latent PDE solve → decoder |
| **Bypass SINN** | Same as v3, plus a skip connection from harmonic boundary interpolation into the decoder |
| **Two-stage (combined)** | bypass SINN (Stage 1) + residual CNN corrector (Stage 2) |
| **Baselines** | Boundary interpolation (cubic), persistence, pure CNN (U-Net), interpolation + CNN corrector |

---

## Installation

**Requirements:** Python 3.11, pip

```bash
# Clone the repo and set up a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

**Dependencies** (see [requirements.txt](requirements.txt)):

| Package | Version |
|---------|---------|
| numpy | 1.26.2 |
| scipy | 1.11.4 |
| matplotlib | 3.8.2 |
| tensorflow | 2.20.0 |
| keras | 3.13.2 |

> The `LEGACY/` directory contains older experiments (1D PDEs, atmospheric data, turbulent flow) that additionally require `torch`, `xarray`, `pandas`, `earthaccess`, and `requests`. These are not needed to reproduce the FYP results.

---

## Configuration

All paths and hyperparameters live in a single file: [config.py](config.py). Edit paths for your machine — every other script imports from here.

```python
# The only values you should need to change:
DATA_PATH    = "sst_2022_2023_2024_combined.pkl"   # path to raw SST pickle
WEIGHTS_DIR  = "weights/"                           # saved model weights
RESULTS_DIR  = "results/"                           # metrics & predictions
FIGURES_DIR  = "figures/"                           # output PNGs
```

Key hyperparameters (defaults):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_LATENTDIM` | 10 | Latent space dimension r |
| `NUM_UNITS` | 128 | Hidden units per MLP layer |
| `NUM_LAYERS` | 3 | MLP depth |
| `PATCH_DIM` | [10, 10, 10] | Patch size (t, y, x) |
| `NUM_PATCHES` | 100 | Patches sampled per batch |
| `N_PAST_STEPS` | 5 | Temporal context for boundary encoder |
| `STAGE1_EPOCHS` | 75 | SINN training epochs |
| `STAGE2_EPOCHS` | 200 | CNN corrector training epochs |
| `SEEDS` | [42, 123, 456] | Multi-seed reproducibility |

---

## Dataset

The raw dataset (`sst_2022_2023_2024_combined.pkl`) contains 1096 daily SST snapshots over a South Pacific domain:

| Split | Year | Days | ENSO Phase |
|-------|------|------|-----------|
| Train | 2022 | 365 | La Niña |
| Test | 2023 | 365 | El Niño |
| Test | 2024 | 366 | Neutral |

---

## Execution Order

### Phase 0 — Data verification

```bash
python plot_sst_3year_comparison.py
```

Produces the ENSO phase comparison figure and confirms the dataset loads correctly.

---

### Phase 1 — Training

Run these independently (they can run in parallel on separate GPUs/sessions):

```bash
python run_v3_baseline.py     # baseline SINN
python run_v8_bypass.py       # bypass SINN
python run_twostage.py        # two-stage: bypass SINN + CNN corrector
python run_baselines.py       # all baselines (interpolation, persistence, pure CNN)
```

Each script saves weights to `weights/` and metrics/predictions to `results/`.

To resume a two-stage run from a saved Stage 1 checkpoint:

```bash
python run_twostage.py --load-stage1
```

---

### Phase 2 — Analysis

Run after Phase 1 completes:

```bash
python analysis_spatial_error.py        # spatial error heatmaps
python analysis_boundary_timeseries.py  # boundary SST time series + PSD
python analysis_seasonal_error.py       # MAE over time with calendar dates
python analysis_a_matrix.py             # A-matrix eigenvalue diagnostics
python analysis_results_table.py        # full results table (CSV + LaTeX)
```

---

### Phase 3 — Ablations and multi-seed (run overnight)

```bash
python run_ablation_latent_dim.py   # r = 1, 3, 5, 10, 15
python run_multiseed.py             # seeds 42, 123, 456 for v3, v8, twostage
```

After these finish, re-run the analysis scripts to include ablation and multi-seed results:

```bash
python analysis_results_table.py
python analysis_a_matrix.py
```

---

## Project Structure

```
myfyp/
├── config.py                        # Central config (paths, hyperparameters)
├── requirements.txt                 # Runtime dependencies
│
├── v3_baseline_sinn.py              # Baseline SINN model class
├── v8_bypass_sinn.py                # Bypass SINN model class (skip connection)
├── cnn_corrector.py                 # ResidualCorrectorCNN + PureCNN architectures
├── utils.py                         # Shared utilities (data loading, seeding, I/O)
├── metrics.py                       # MAE, RMSE, R², bias, PCC computation
│
├── run_v3_baseline.py               # Train & evaluate SINN
├── run_v8_bypass.py                 # Train & evaluate bypass SINN
├── run_twostage.py                  # Train & evaluate two-stage pipeline
├── run_baselines.py                 # Train & evaluate all baselines
├── run_ablation_latent_dim.py       # Ablation: latent dimension r
├── run_multiseed.py                 # Multi-seed robustness study
│
├── plot_sst_3year_comparison.py     # ENSO phase characterisation figure
├── analysis_spatial_error.py        # Spatial error decomposition
├── analysis_boundary_timeseries.py  # Boundary oscillation analysis
├── analysis_seasonal_error.py       # Temporal error + seasonal structure
├── analysis_a_matrix.py             # Learned operator A diagnostics
├── analysis_results_table.py        # Aggregated results table
├── analysis_latent_field.py         # Latent field visualisation
├── analysis_latent_dims_report.py   # Report-quality latent field figures
├── analysis_boundary_encoder.py     # Boundary encoder representation study
├── analysis_boundary_noise.py       # Robustness to boundary noise
├── analysis_small_domain_field.py   # Sub-domain analysis
├── analysis_inference_time.py       # Inference wall-clock benchmarks
├── analysis_train_reconstruction.py # Reconstruction on training set
├── analysis_standardisation_bias.py # Bias from standardisation procedure
├── animate_results.py               # Animated GIF of predictions
│
├── weights/                         # Saved model weights (auto-created)
├── results/                         # Metrics, predictions, tables (auto-created)
├── figures/                         # Output PNGs for thesis (auto-created)
├── figures_for_report/              # Curated report figures + generation scripts
│   ├── generate_report_figures.py
│   ├── report_style.py
│   └── *.png
│
├── sst_2022_2023_2024_combined.pkl  # Raw SST dataset (3 years)
└── LEGACY/                          # Older experiments (not needed for FYP)
```

---

## Key Scripts and Thesis Mapping

| Script | Output | Thesis Section |
|--------|--------|----------------|
| `plot_sst_3year_comparison.py` | ENSO comparison figure | Ch 3 (Data) |
| `analysis_a_matrix.py` | A-matrix eigenvalue plots | §4.2 (Diagnostics) |
| `analysis_seasonal_error.py` | Temporal MAE plot | §4.2 + Ch 5 |
| `analysis_spatial_error.py` | Spatial error heatmaps | Ch 5 (Results) |
| `analysis_boundary_timeseries.py` | Boundary SST + PSD | Ch 5 (Results) |
| `analysis_results_table.py` | Summary table (CSV + LaTeX) | Ch 5 (Results) |
| `run_ablation_latent_dim.py` | MAE vs r ablation plot | Ch 5 (Results) |
| `run_multiseed.py` | Mean ± std across seeds | Ch 5 (Results) |

---

## Output Artifacts

After a full run, the following are produced:

- `weights/` — encoder, decoder, and A-matrix weights for each model variant
- `results/` — per-timestep metrics and full field predictions (`.pkl`), plus `results_table.csv` and `results_table.tex`
- `figures/` — all thesis figures as PNGs
- `figures_for_report/` — curated subset formatted for the report

---

## Notes

- All scripts use `matplotlib.use("Agg")` for headless/server operation.
- The `--load-stage1` flag on `run_twostage.py` skips Stage 1 training and loads saved weights.
- Results are reproducible: seeds are controlled globally via `utils.set_seed()`.
- `figures_for_report/report_style.py` sets consistent matplotlib styling across all report figures.
