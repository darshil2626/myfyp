"""
Reconstruction error metrics for SINN test-set evaluation.

All metrics work on (ny, nx) numpy arrays plus a boolean interior mask
that selects which pixels participate. u_pred and u_true must be in the
same physical units; for the SINN pipeline that's degrees Celsius after
the un-standardisation step inside reconstruct_field_at_timestep.

Usage:
    from metrics import compute_metrics, aggregate_per_step_metrics
"""

import numpy as np


METRIC_KEYS = ("mae", "rmse", "bias", "max_abs", "p99_abs", "r2", "pcc")


def compute_metrics(u_pred, u_true, interior_mask):
    """Per-timestep error metrics over the interior.

    Returns a dict with:
        mae       - mean absolute error
        rmse      - root mean squared error
        bias      - mean signed error (u_pred - u_true)
        max_abs   - maximum absolute error
        p99_abs   - 99th percentile of absolute error
        r2        - coefficient of determination (1 - SS_res / SS_tot)
        pcc       - Pearson pattern correlation between u_pred and u_true
    """
    int_y, int_x = np.where(interior_mask)
    p = np.asarray(u_pred, dtype=np.float64)[int_y, int_x]
    t = np.asarray(u_true, dtype=np.float64)[int_y, int_x]
    err = p - t
    abs_err = np.abs(err)

    if t.size == 0:
        nan = float("nan")
        return {k: nan for k in METRIC_KEYS}

    ss_tot = float(np.sum((t - t.mean()) ** 2))
    ss_res = float(np.sum(err ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    p_std = p.std()
    t_std = t.std()
    if p_std > 0 and t_std > 0:
        pcc = float(np.corrcoef(p, t)[0, 1])
    else:
        pcc = float("nan")

    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(err.mean()),
        "max_abs": float(abs_err.max()),
        "p99_abs": float(np.percentile(abs_err, 99)),
        "r2": r2,
        "pcc": pcc,
    }


def aggregate_per_step_metrics(per_step_results, interior_mask):
    """Run compute_metrics on every record. Returns (per_metric_lists, mean_dict).

    per_step_results: iterable of dicts each containing 'u_pred' and 'u_true'.
    Records missing either field are skipped. Empty input produces NaN means.
    """
    per = {k: [] for k in METRIC_KEYS}
    for r in per_step_results:
        u_pred = r.get("u_pred")
        u_true = r.get("u_true")
        if u_pred is None or u_true is None:
            continue
        m = compute_metrics(u_pred, u_true, interior_mask)
        for k in METRIC_KEYS:
            per[k].append(m[k])
    means = {k: (float(np.nanmean(v)) if len(v) > 0 else float("nan"))
             for k, v in per.items()}
    return per, means
