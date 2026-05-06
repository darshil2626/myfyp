# SINN Experiment Scripts — Execution Guide
## Darshil Gajjar FYP, May 2026

### Setup
1. Place all these scripts in your working directory alongside your existing model files:
   - `v3_baseline_sinn.py` (your existing file — DO NOT modify)
   - `v8_bypass_sinn.py` (your existing file — DO NOT modify)

2. Edit `config.py`:
   - Set `DATA_PATH` to point at your `sst_2022_2023_2024_combined.pkl`
   - Set `SINN_V3_MODULE` and `SINN_V8_MODULE` to the filenames of your model classes
   - Adjust `STAGE1_EPOCHS` if needed (75 is default)

3. Your existing model classes work as-is. These scripts import them via `importlib`.

### Execution Order

**Phase 0 — verify data:**
```
python plot_sst_3year_comparison.py
```
Produces the ENSO comparison figure. Confirms data loads correctly.

**Phase 1 — training runs (start these, then write while they run):**
```
python run_v3_baseline.py          # v3 baseline SINN
python run_v8_bypass.py            # v8 bypass SINN  
python run_twostage.py             # bypass SINN + CNN corrector
python run_baselines.py            # boundary interp, persistence, pure CNN
```

Each saves weights to `weights/` and results to `results/`.
If a run crashes, restart with `--load` to skip training:
```
python run_twostage.py --load-stage1   # reload Stage 1, retrain Stage 2 only
```

**Phase 2 — analysis (run after Phase 1 completes):**
```
python analysis_spatial_error.py       # spatial error decomposition
python analysis_boundary_timeseries.py # boundary SST near complex region
python analysis_seasonal_error.py      # MAE over time with calendar dates
python analysis_a_matrix.py            # A-matrix eigenvalue comparison
python analysis_results_table.py       # compile all results into table
```

**Phase 3 — ablation and multi-seed (run overnight):**
```
python run_ablation_latent_dim.py      # r = 1, 3, 5, 10, 15
python run_multiseed.py                # 3 seeds for v3, v8, twostage
```

**After multi-seed completes:**
```
python analysis_results_table.py       # re-run to include multi-seed stats
python analysis_a_matrix.py            # re-run to include ablation A matrices
```

### Output Directories
- `weights/` — saved model weights (for reloading without retraining)
- `results/` — pickle files with metrics, CSV/LaTeX table
- `figures/` — all PNG figures for the thesis

### Key Files
| Script | Produces | Thesis Section |
|--------|----------|---------------|
| `plot_sst_3year_comparison.py` | ENSO comparison figure | Ch 3 (Data) |
| `analysis_a_matrix.py` | A-matrix eigenvalue plots | §4.2 (Diagnostics) |
| `analysis_seasonal_error.py` | v3 temporal degradation plot | §4.2 + Ch 5 |
| `analysis_spatial_error.py` | Spatial error heatmaps | Ch 5 (Results) |
| `analysis_boundary_timeseries.py` | Boundary SST time series + PSD | Ch 5 (Results) |
| `analysis_results_table.py` | Summary table (CSV + LaTeX) | Ch 5 (Results) |
| `run_ablation_latent_dim.py` | MAE vs r plot | Ch 5 (Results) |

### Notes
- Your existing `twostage_animator.py` still works — just point `PKL_PATH` at 
  `results/test_results_twostage_seed42.pkl`
- All scripts use `matplotlib.use("Agg")` for headless operation
- Seeds are controlled globally — results are reproducible
- The `--load` flag on any runner skips training and loads saved weights
