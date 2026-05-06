"""
Compile all results into a clean summary table.

Loads every result pickle, recomputes a richer metric set (MAE, RMSE, bias,
max |err|, P99 |err|, R^2, pattern correlation) from the saved per-timestep
predictions where available, and writes:
  1. Console output with a formatted table
  2. CSV for thesis copy-paste
  3. LaTeX table fragment

All metrics are reported in °C — the saved per-step u_pred / u_true are in
the original physical units (un-standardised before MAE was computed).

Usage:
    python analysis_results_table.py
"""

import numpy as np
import csv

from config import (
    B_THICK, DEFAULT_SEED, RESULTS_DIR, SEEDS, ABLATION_LATENT_DIMS,
    ensure_dirs,
)
from utils import load_results, load_data
from metrics import aggregate_per_step_metrics


METRIC_KEYS = ("mae", "rmse", "bias", "max_abs", "p99_abs", "r2", "pcc")
METRIC_LABELS = {
    "mae": "MAE (°C)",
    "rmse": "RMSE (°C)",
    "bias": "Bias (°C)",
    "max_abs": "Max |err| (°C)",
    "p99_abs": "P99 |err| (°C)",
    "r2": "R²",
    "pcc": "Pattern r",
}


def _interior_mask_from_data():
    """Build the 2D interior mask matching what the runners use."""
    data = load_data()
    U = np.asarray(data["U"])
    ny, nx = U.shape[1], U.shape[2]
    obs_mask = np.zeros((ny, nx), dtype=bool)
    obs_mask[:B_THICK, :] = True; obs_mask[-B_THICK:, :] = True
    obs_mask[:, :B_THICK] = True; obs_mask[:, -B_THICK:] = True
    return ~obs_mask, U


def _metrics_from_records(records, interior_mask):
    """Mean of per-timestep compute_metrics over a list of records."""
    _, means = aggregate_per_step_metrics(records, interior_mask)
    return means


def _metrics_from_pred_array(u_pred_per_t, U, test_indices, interior_mask):
    """Build records on the fly from per-timestep prediction arrays.

    Used for baselines that didn't store full fields. u_pred_per_t may be
    None in which case only MAE is recoverable from the saved array.
    """
    if u_pred_per_t is None:
        return None
    records = []
    for u_pred, t in zip(u_pred_per_t, test_indices):
        records.append({"u_pred": u_pred, "u_true": U[int(t)]})
    _, means = aggregate_per_step_metrics(records, interior_mask)
    return means


def _empty_metrics():
    return {k: float("nan") for k in METRIC_KEYS}


def _mean_with_std(values_per_seed):
    """values_per_seed: list of dicts {metric: value}.
    Returns dict of {metric: (mean, std)}."""
    out = {}
    for k in METRIC_KEYS:
        vs = [d.get(k, float("nan")) for d in values_per_seed]
        vs = [v for v in vs if not (v is None or (isinstance(v, float) and np.isnan(v)))]
        if vs:
            out[k] = (float(np.mean(vs)), float(np.std(vs)) if len(vs) > 1 else None)
        else:
            out[k] = (float("nan"), None)
    return out


def _collect_seeded_model(pkl_template, interior_mask, results_key="results"):
    """Aggregate a model's per-seed metrics. Returns (per_seed_means, n_seeds)."""
    per_seed = []
    for seed in SEEDS:
        try:
            data = load_results(pkl_template.format(seed=seed))
        except FileNotFoundError:
            continue
        records = data.get(results_key, [])
        if not records:
            continue
        per_seed.append(_metrics_from_records(records, interior_mask))
    if not per_seed:
        # Fall back to single-seed default
        try:
            data = load_results(pkl_template.format(seed=DEFAULT_SEED))
        except FileNotFoundError:
            return [], 0
        records = data.get(results_key, [])
        if not records:
            return [], 0
        per_seed.append(_metrics_from_records(records, interior_mask))
    return per_seed, len(per_seed)


def main():
    ensure_dirs()
    interior_mask, U = _interior_mask_from_data()

    print(f"\n{'='*100}")
    print("RESULTS COMPILATION  (interior-only metrics, °C unless noted)")
    print(f"{'='*100}")

    rows = []  # list of (name, notes, dict_of_(metric: (mean, std_or_None)))

    # ---- Baselines ----
    try:
        bl = load_results(f"baseline_results_seed{DEFAULT_SEED}.pkl")
        test_indices = bl["test_time_indices"]

        # Boundary interp and persistence: full records now saved, enabling all metrics.
        # Pure CNN: only MAE saved; falls back to MAE-only row.
        for label, records_key, mae_key in [
            ("Boundary Interpolation", "interp_results", "interp_maes"),
            ("Persistence", "persist_results", "persist_maes"),
        ]:
            if records_key in bl and bl[records_key]:
                means = _metrics_from_records(bl[records_key], interior_mask)
                rows.append((label, "1 seed",
                             {k: (means[k], None) for k in METRIC_KEYS}))
            elif mae_key in bl:
                # Legacy pkl without full records — fallback to MAE-only
                m = _empty_metrics()
                m["mae"] = float(np.mean(bl[mae_key]))
                rows.append((label, "1 seed (MAE only)",
                             {k: (m[k], None) for k in METRIC_KEYS}))

        # Pure CNN — only MAE saved
        if "cnn_maes" in bl:
            m = _empty_metrics()
            m["mae"] = float(np.mean(bl["cnn_maes"]))
            rows.append(("Pure CNN", "1 seed (MAE only)",
                         {k: (m[k], None) for k in METRIC_KEYS}))

        # Boundary-interp + CNN corrector — has full records.
        if "interp_cnn_results" in bl:
            means = _metrics_from_records(bl["interp_cnn_results"], interior_mask)
            rows.append(("Interp + CNN corrector", "1 seed",
                         {k: (means[k], None) for k in METRIC_KEYS}))
    except FileNotFoundError:
        print("  Baselines not found")

    # ---- v3 baseline SINN ----
    per_seed_v3, n_v3 = _collect_seeded_model(
        "test_results_v3_seed{seed}.pkl", interior_mask)
    if per_seed_v3:
        rows.append(("v3 Baseline SINN", f"{n_v3} seed{'s' if n_v3 > 1 else ''}",
                     _mean_with_std(per_seed_v3)))

    # ---- v8 bypass SINN ----
    per_seed_v8, n_v8 = _collect_seeded_model(
        "test_results_v8_seed{seed}.pkl", interior_mask)
    if per_seed_v8:
        rows.append(("v8 Bypass SINN", f"{n_v8} seed{'s' if n_v8 > 1 else ''}",
                     _mean_with_std(per_seed_v8)))

    # ---- twostage (SINN + CNN) ----
    per_seed_ts_combined = []
    per_seed_ts_s1 = []
    for seed in SEEDS:
        try:
            ts = load_results(f"test_results_twostage_seed{seed}.pkl")
        except FileNotFoundError:
            continue
        recs = ts["results"]
        per_seed_ts_combined.append(_metrics_from_records(recs, interior_mask))
        # Stage 1 metrics: synthesise records using u_pred_stage1 instead.
        s1_recs = [{"u_pred": r["u_pred_stage1"], "u_true": r["u_true"]}
                   for r in recs if "u_pred_stage1" in r]
        if s1_recs:
            per_seed_ts_s1.append(_metrics_from_records(s1_recs, interior_mask))
    if not per_seed_ts_combined:
        try:
            ts = load_results(f"test_results_twostage_seed{DEFAULT_SEED}.pkl")
            recs = ts["results"]
            per_seed_ts_combined.append(_metrics_from_records(recs, interior_mask))
            s1_recs = [{"u_pred": r["u_pred_stage1"], "u_true": r["u_true"]}
                       for r in recs if "u_pred_stage1" in r]
            if s1_recs:
                per_seed_ts_s1.append(_metrics_from_records(s1_recs, interior_mask))
        except FileNotFoundError:
            pass
    if per_seed_ts_s1:
        n = len(per_seed_ts_s1)
        rows.append(("Bypass SINN (Stage 1 of twostage)",
                     f"{n} seed{'s' if n > 1 else ''}",
                     _mean_with_std(per_seed_ts_s1)))
    if per_seed_ts_combined:
        n = len(per_seed_ts_combined)
        rows.append(("Bypass SINN + CNN (Combined)",
                     f"{n} seed{'s' if n > 1 else ''}",
                     _mean_with_std(per_seed_ts_combined)))

    # ---- Ablation ----
    try:
        abl = load_results("ablation_latent_dim_results.pkl")
        for r in sorted(abl.keys()):
            entry = abl[r]
            # The ablation runner saved aggregate mae/mse but no u_pred fields,
            # so non-MAE metrics aren't recoverable here.
            m = _empty_metrics()
            m["mae"] = float(entry.get("mean_mae", float("nan")))
            m["rmse"] = (float(np.sqrt(entry["mean_mse"]))
                          if "mean_mse" in entry else float("nan"))
            rows.append((f"Bypass SINN (r={r})", "ablation",
                         {k: (m[k], None) for k in METRIC_KEYS}))
    except FileNotFoundError:
        pass

    # ---- Print table ----
    if not rows:
        print("No results found.")
        return

    header = f"{'Model':<38} {'Notes':>10}  " + "  ".join(
        f"{METRIC_LABELS[k]:>13}" for k in METRIC_KEYS)
    print()
    print(header)
    print("-" * len(header))

    # Find best MAE for highlighting
    best_mae = min(d["mae"][0] for _, _, d in rows
                   if not np.isnan(d["mae"][0]))

    for name, notes, d in rows:
        cells = []
        for k in METRIC_KEYS:
            mean, std = d[k]
            if np.isnan(mean):
                cells.append(f"{'—':>13}")
            elif std is not None:
                cells.append(f"{mean:>6.3f}±{std:5.3f}")
            else:
                cells.append(f"{mean:>13.4f}")
        marker = " ◄" if d["mae"][0] == best_mae else ""
        print(f"  {name:<36} {notes:>10}  " + "  ".join(cells) + marker)

    # ---- Relative MAE improvements ----
    print(f"\n{'='*60}")
    print("RELATIVE MAE IMPROVEMENTS")
    print(f"{'='*60}")
    mae_dict = {name: d["mae"][0] for name, _, d in rows
                if not np.isnan(d["mae"][0])}
    baseline_ref = None
    for ref_name in ["Boundary Interpolation", "v3 Baseline SINN"]:
        if ref_name in mae_dict:
            baseline_ref = ref_name
            break
    if baseline_ref:
        ref_mae = mae_dict[baseline_ref]
        print(f"  Reference: {baseline_ref} (MAE = {ref_mae:.4f} °C)")
        for name, mae in mae_dict.items():
            if name != baseline_ref:
                imp = (1 - mae / ref_mae) * 100
                print(f"  {name:<38} {imp:>+8.1f}% vs reference")

    # ---- Save CSV ----
    csv_path = f"{RESULTS_DIR}/results_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_row = ["Model", "Notes"]
        for k in METRIC_KEYS:
            header_row.extend([METRIC_LABELS[k], f"{METRIC_LABELS[k]} std"])
        writer.writerow(header_row)
        for name, notes, d in rows:
            row = [name, notes]
            for k in METRIC_KEYS:
                mean, std = d[k]
                row.append(f"{mean:.6f}" if not np.isnan(mean) else "")
                row.append(f"{std:.6f}" if std is not None else "")
            writer.writerow(row)
    print(f"\nSaved {csv_path}")

    # ---- Save LaTeX fragment (MAE / RMSE / R^2 / P99 — readable subset) ----
    tex_path = f"{RESULTS_DIR}/results_table.tex"
    show_keys = ("mae", "rmse", "p99_abs", "r2")
    show_labels = {
        "mae": "MAE (°C)",
        "rmse": "RMSE (°C)",
        "p99_abs": "P99 $|$err$|$ (°C)",
        "r2": "$R^2$",
    }
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Model performance on the test set (2023--2024). "
                "MAE, RMSE and 99th-percentile absolute error reported in $^\\circ$C; "
                "$R^2$ is the coefficient of determination. Lower is better, except $R^2$.}\n")
        f.write("\\label{tab:results}\n")
        cols = "l" + "c" * len(show_keys)
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\hline\n")
        f.write("Model & " + " & ".join(show_labels[k] for k in show_keys)
                + " \\\\\n\\hline\n")
        for name, notes, d in rows:
            cells = [name]
            for k in show_keys:
                mean, std = d[k]
                if np.isnan(mean):
                    cells.append("---")
                elif std is not None:
                    cells.append(f"{mean:.4f} $\\pm$ {std:.4f}")
                else:
                    cells.append(f"{mean:.4f}")
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
