"""
generate_report_figures.py
===========================
Reproduce every figure used in the thesis report with one consistent,
title-free, large-font visual style (see report_style.py).

Run from anywhere:
    python figures_for_report/generate_report_figures.py            # all figures
    python figures_for_report/generate_report_figures.py psd field  # subset by keyword

Outputs are written to  myfyp/figures_for_report/.
Data come from  myfyp/results/  and the SST pickle; the latent-component
figure additionally loads the trained v3 weights (needs tensorflow).
"""

import os
import sys
import pickle
import numpy as np

# make myfyp importable (config, utils) regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_MYFYP = os.path.dirname(_HERE)
sys.path.insert(0, _MYFYP)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import report_style as S
from config import (
    RESULTS_DIR, DATA_PATH, DEFAULT_SEED, B_THICK,
    N_DAYS_2022, N_DAYS_2023, N_DAYS_2024, N_TOTAL,
    TEST_START, TEST_END,
)

S.apply_style()


# ---------------------------------------------------------------------------
# small loaders
# ---------------------------------------------------------------------------
_CACHE = {}


def _load(name):
    if name not in _CACHE:
        with open(os.path.join(RESULTS_DIR, name), "rb") as f:
            _CACHE[name] = pickle.load(f)
    return _CACHE[name]


def _data():
    if "_U" not in _CACHE:
        with open(DATA_PATH, "rb") as f:
            d = pickle.load(f)
        _CACHE["_U"] = (np.asarray(d["U"], np.float32),
                        np.asarray(d["X"], np.float32),
                        np.asarray(d["Y"], np.float32))
    return _CACHE["_U"]


def _sorted_series(results, key):
    """Return (t_index, value) sorted by t_index for a results list."""
    pairs = sorted((r["t_index"], r[key]) for r in results)
    t = np.array([p[0] for p in pairs])
    v = np.array([p[1] for p in pairs])
    return t, v


def _year_dividers(ax, t):
    """Vertical dashed lines + year labels for a test-period x axis (t in days)."""
    yr2024 = N_DAYS_2022 + N_DAYS_2023   # = 730, first day of 2024
    ax.axvline(yr2024, color=S.GREY, ls="--", lw=1.0, alpha=0.7)
    ymin, ymax = ax.get_ylim()
    ypos = ymin + 0.06 * (ymax - ymin)
    ax.text((TEST_START + yr2024) / 2, ypos, "2023", ha="center",
            fontsize=12, style="italic", color="0.35")
    ax.text((yr2024 + TEST_END) / 2, ypos, "2024", ha="center",
            fontsize=12, style="italic", color="0.35")


# ===========================================================================
# LINE-PLOT FIGURES
# ===========================================================================
def fig_test_mae_baselines():
    base = _load(f"baseline_results_seed{DEFAULT_SEED}.pkl")
    v3 = _load(f"test_results_v3_seed{DEFAULT_SEED}.pkl")
    t = np.asarray(base["test_time_indices"])
    interp = np.asarray(base["interp_maes"])
    persist = np.asarray(base["persist_maes"])
    v3_map = {r["t_index"]: r["mae"] for r in v3["results"]}
    sinn = np.array([v3_map[int(ti)] for ti in t])

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.plot(t, persist, color=S.GREY, lw=1.4, alpha=0.9,
            label=f"Persistence  ({np.mean(persist):.3f} °C)")
    ax.plot(t, interp, color=S.ORANGE, lw=1.4, alpha=0.95,
            label=f"Boundary interpolation  ({np.mean(interp):.3f} °C)")
    ax.plot(t, sinn, color=S.NAVY, lw=2.0,
            label=f"Time-conditioned SINN  ({v3['mean_mae']:.3f} °C)")
    ax.set_xlabel("Test-period day index $t$")
    ax.set_ylabel("Per-timestep MAE (°C)")
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0, None)
    S.grid(ax)
    _year_dividers(ax, t)
    ax.legend(loc="upper right")
    S.save(fig, "test_mae_v3_with_baselines.png")


def fig_twostage_mae():
    ts = _load(f"test_results_twostage_seed{DEFAULT_SEED}.pkl")
    t, s1 = _sorted_series(ts["results"], "mae_stage1")
    _, comb = _sorted_series(ts["results"], "mae")
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.plot(t, s1, color=S.TEAL, lw=1.6, ls="--",
            label=f"Bypass SINN (Stage 1)  ({ts['stage1_test_mae']:.3f} °C)")
    ax.plot(t, comb, color=S.NAVY, lw=2.0,
            label=f"Two-stage (Stage 1 + CNN)  ({ts['combined_test_mae']:.3f} °C)")
    ax.set_xlabel("Test-period day index $t$")
    ax.set_ylabel("Per-timestep MAE (°C)")
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0, None)
    S.grid(ax)
    _year_dividers(ax, t)
    ax.legend(loc="upper right")
    S.save(fig, "test_mae_comparison_twostage_seed42.png")


def fig_small_domain_comparison():
    d = _load("small_domain_results_125.pkl")
    crop = np.asarray(d["crop_per_step_mae"])
    full = np.asarray(d["full_region_maes"])
    t = np.arange(TEST_START, TEST_END)[:len(crop)]
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.plot(t, full, color=S.ORANGE, lw=1.5, ls="--",
            label=f"Full domain, same region  ({d['full_region_mean_mae']:.3f} °C)")
    ax.plot(t, crop, color=S.NAVY, lw=2.0,
            label=f"Cropped thin strip 24×124  ({d['crop_mae']:.3f} °C)")
    ax.set_xlabel("Test-period day index $t$")
    ax.set_ylabel("Interior MAE (°C)")
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0, None)
    S.grid(ax)
    _year_dividers(ax, t)
    ax.legend(loc="upper right")
    S.save(fig, "small_domain_comparison_125.png")


def fig_training_loss():
    h = _load(f"loss_history_v3_seed{DEFAULT_SEED}.pkl")
    ep = np.arange(1, len(h["total"]) + 1)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.semilogy(ep, h["total"], color=S.BLACK, lw=2.0, label=r"Total $\Psi$")
    ax.semilogy(ep, h["recon"], color=S.VERMILION, lw=1.5, ls="--",
                label=r"Reconstruction $\Psi_u$")
    ax.semilogy(ep, h["latent"], color=S.SKY, lw=1.5, ls="--",
                label=r"Latent $\Psi_\ell$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_xlim(ep.min(), ep.max())
    S.grid(ax)
    ax.legend(loc="upper right")
    S.save(fig, "v3_training_loss.png")


def fig_patch_size():
    d = _load("ablation_patch_size_results.pkl")
    ps = sorted(d.keys())
    t = np.arange(TEST_START, TEST_END)[:len(d[ps[0]]["per_step_mae"])]
    cols = S.ordered_colors(len(ps))
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for p, c in zip(ps, cols):
        ax.plot(t, d[p]["per_step_mae"], lw=1.2, alpha=0.9, color=c,
                label=f"$P={p}$  (mean {d[p]['mean_mae']:.3f} °C)")
    ax.set_xlabel("Test-period day index $t$")
    ax.set_ylabel("Per-step MAE (°C)")
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0, None)
    S.grid(ax)
    ax.legend(loc="upper left", ncol=2, title="Patch size", fontsize=10,
              title_fontsize=10)
    S.save(fig, "ablation_patch_size.png")


def fig_latent_dim():
    d = _load("ablation_latent_dim_results.pkl")
    rs = sorted(d.keys())
    M = np.vstack([np.asarray(d[r]["per_step_mae"]) for r in rs])   # (n_r, n_t)
    t = np.arange(TEST_START, TEST_END)[:M.shape[1]]
    cols = S.ordered_colors(len(rs))

    # two-way additive decomposition  MAE(r,t) = a(r) + s(t) + eps
    a = M.mean(axis=1, keepdims=True)              # per-r mean
    centred = M - a                                # remove per-r level
    s = centred.mean(axis=0)                       # shared shape
    eps = centred - s
    frac = np.var(s) / (np.var(s) + np.var(eps))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.0))
    for i, (r, c) in enumerate(zip(rs, cols)):
        axL.plot(t, M[i], lw=1.2, alpha=0.9, color=c,
                 label=f"$r={r}$  (mean {a[i,0]:.3f} °C)")
    axL.set_xlabel("Test-period day index $t$")
    axL.set_ylabel("Per-step MAE (°C)")
    axL.set_xlim(t.min(), t.max())
    axL.set_ylim(0, None)
    S.grid(axL)
    axL.legend(loc="upper left", title="Latent dim", fontsize=10, title_fontsize=10)

    for i, c in enumerate(cols):
        axR.plot(t, centred[i], lw=1.0, alpha=0.55, color=c)
    axR.plot(t, s, color=S.BLACK, lw=2.4,
             label=f"shared shape $s(t)$  ({frac*100:.1f}% of variance)")
    axR.axhline(0, color="0.6", lw=0.8, ls="--")
    axR.set_xlabel("Test-period day index $t$")
    axR.set_ylabel("MAE − per-$r$ mean (°C)")
    axR.set_xlim(t.min(), t.max())
    S.grid(axR)
    axR.legend(loc="upper left")
    fig.tight_layout()
    S.save(fig, "ablation_latent_dim.png")
    print(f"    [latent dim] shared-shape variance fraction = {frac*100:.2f}%")


def fig_boundary_noise():
    d = _load("boundary_noise_results.pkl")
    sig = sorted(d["v3"].keys())
    v3 = [d["v3"][s] for s in sig]
    v8 = [d["v8_bypass"][s] for s in sig]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.0))

    a0.plot(sig, v3, "o-", color=S.ORANGE, lw=2, ms=8,
            label="Time-conditioned SINN")
    a0.plot(sig, v8, "s-", color=S.NAVY, lw=2, ms=8,
            label="Bypass SINN (Stage 1)")
    a0.set_xlabel("Boundary noise $\\sigma$ (°C)")
    a0.set_ylabel("Mean MAE (°C)")
    S.grid(a0); a0.legend(loc="upper left")

    a1.plot(sig, [m / v3[0] for m in v3], "o-", color=S.ORANGE, lw=2, ms=8,
            label="Time-conditioned SINN")
    a1.plot(sig, [m / v8[0] for m in v8], "s-", color=S.NAVY, lw=2, ms=8,
            label="Bypass SINN (Stage 1)")
    a1.axhline(1.0, color="0.6", ls="--", lw=0.8)
    a1.set_xlabel("Boundary noise $\\sigma$ (°C)")
    a1.set_ylabel("Relative MAE (normalised to $\\sigma=0$)")
    S.grid(a1); a1.legend(loc="upper left")
    fig.tight_layout()
    S.save(fig, "boundary_noise_robustness.png")


def fig_boundary_psd():
    from scipy.signal import welch
    U, _, _ = _data()
    nt, ny, nx = U.shape
    bx = nx - 1
    ys = np.linspace(ny // 2, ny - B_THICK - 1, 4, dtype=int)
    cols = [S.NAVY, S.TEAL, S.ORANGE, S.VERMILION]
    days = np.arange(nt)

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(13, 7.5),
                                 gridspec_kw={"height_ratios": [2, 1.6]})
    for y, c in zip(ys, cols):
        a0.plot(days, U[:, y, bx], color=c, lw=1.0, alpha=0.9,
                label=f"({y}, {bx})")
    a0.axvline(N_DAYS_2022, color="0.5", ls="--", lw=1, alpha=0.7)
    a0.axvline(N_DAYS_2022 + N_DAYS_2023, color="0.5", ls="--", lw=1, alpha=0.7)
    a0.set_ylabel("Boundary SST (°C)")
    a0.set_xlim(0, nt)
    S.grid(a0)
    ymin = a0.get_ylim()[0]
    for xc, lab in [(N_DAYS_2022/2, "2022 (train)"),
                    (N_DAYS_2022 + N_DAYS_2023/2, "2023 (test)"),
                    (N_DAYS_2022 + N_DAYS_2023 + N_DAYS_2024/2, "2024 (test)")]:
        a0.text(xc, ymin + 0.3, lab, ha="center", fontsize=11, style="italic",
                color="0.35")
    a0.legend(loc="upper left", ncol=2, title="Boundary point (row, col)",
              fontsize=10, title_fontsize=10)

    for y, c in zip(ys, cols):
        sst = U[:, y, bx].astype(np.float64)
        freqs, psd = welch(sst - sst.mean(), fs=1.0, nperseg=min(512, nt))
        a1.semilogy(1.0 / freqs[1:], psd[1:], color=c, lw=1.6, alpha=0.9,
                    label=f"({y}, {bx})")
    a1.axvline(365, color="0.35", ls=":", lw=1.6, label="Annual (365 d)")
    a1.axvline(182, color="0.6", ls=":", lw=1.3, label="Semi-annual (182 d)")
    a1.set_xlabel("Period (days)")
    a1.set_ylabel("PSD")
    a1.set_xlim(5, 600)
    S.grid(a1)
    a1.legend(loc="lower right", ncol=2, fontsize=10)
    fig.tight_layout()
    S.save(fig, "boundary_oscillation_psd.png")


def fig_enso_comparison():
    U, _, _ = _data()
    nd = {2022: N_DAYS_2022, 2023: N_DAYS_2023, 2024: N_DAYS_2024}
    years = [2022, 2023, 2024]
    idx = 0
    Uy, mean, std, q25, q75, doy = {}, {}, {}, {}, {}, {}
    for y in years:
        Uy[y] = U[idx:idx + nd[y]]; idx += nd[y]
        mean[y] = np.array([np.nanmean(Uy[y][t]) for t in range(nd[y])])
        std[y] = np.array([np.nanstd(Uy[y][t]) for t in range(nd[y])])
        q25[y] = np.array([np.nanpercentile(Uy[y][t], 25) for t in range(nd[y])])
        q75[y] = np.array([np.nanpercentile(Uy[y][t], 75) for t in range(nd[y])])
        doy[y] = np.arange(1, nd[y] + 1)
    nc = 365

    def stat(y1, y2):
        m1, m2 = mean[y1][:nc], mean[y2][:nc]
        return np.corrcoef(m1, m2)[0, 1], np.sqrt(np.mean((m2 - m1) ** 2))
    r23, rmsd23 = stat(2022, 2023)
    r24, rmsd24 = stat(2022, 2024)

    labels = {2022: "2022 — La Niña (training)",
              2023: "2023 — El Niño (test)",
              2024: "2024 — neutral (test)"}
    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2, 1.5]})
    for y in years:
        ax[0].plot(doy[y], mean[y], color=S.ENSO_COLOURS[y], lw=1.9,
                   label=labels[y])
        ax[0].fill_between(doy[y], q25[y], q75[y], color=S.ENSO_COLOURS[y],
                           alpha=0.12)
    ax[0].set_ylabel("SST (°C)")
    box = (f"2022 vs 2023:  r = {r23:.3f},  RMSD = {rmsd23:.3f} °C\n"
           f"2022 vs 2024:  r = {r24:.3f},  RMSD = {rmsd24:.3f} °C")
    ax[0].text(0.015, 0.97, box, transform=ax[0].transAxes, fontsize=11,
               va="top", family="monospace",
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))
    ax[0].legend(loc="upper right")
    S.grid(ax[0])

    dc = np.arange(1, nc + 1)
    d23 = mean[2023][:nc] - mean[2022][:nc]
    d24 = mean[2024][:nc] - mean[2022][:nc]
    ax[1].plot(dc, d23, color=S.ENSO_COLOURS[2023], lw=1.6, label="2023 − 2022")
    ax[1].plot(dc, d24, color=S.ENSO_COLOURS[2024], lw=1.6, label="2024 − 2022")
    ax[1].fill_between(dc, d23, 0, color=S.ENSO_COLOURS[2023], alpha=0.12)
    ax[1].fill_between(dc, d24, 0, color=S.ENSO_COLOURS[2024], alpha=0.12)
    ax[1].axhline(0, color="0.4", lw=0.8, ls="--")
    ax[1].set_ylabel("ΔSST from 2022 (°C)")
    ax[1].legend(loc="upper left")
    S.grid(ax[1])

    for y in years:
        ax[2].plot(doy[y], std[y], color=S.ENSO_COLOURS[y], lw=1.3, label=str(y))
    ax[2].set_ylabel("Spatial σ (°C)")
    ax[2].set_xlabel("Day of year")
    S.grid(ax[2])
    months = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax[2].set_xticks(months)
    ax[2].set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax[2].set_xlim(1, 366)
    ax[2].legend(loc="upper right", ncol=3)
    fig.tight_layout()
    S.save(fig, "sst_enso_phase_comparison.png")


# ===========================================================================
# FIELD / MAP FIGURES
# ===========================================================================
def fig_seasonal():
    U, X, Y = _data()
    U22 = U[:N_DAYS_2022]
    seasons = {
        "DJF — austral summer": list(range(0, 59)) + list(range(334, 365)),
        "MAM — austral autumn": list(range(59, 151)),
        "JJA — austral winter": list(range(151, 243)),
        "SON — austral spring": list(range(243, 334)),
    }
    fields = {k: np.nanmean(U22[v], axis=0) for k, v in seasons.items()}
    allv = np.concatenate([f.ravel() for f in fields.values()])
    vmin, vmax = np.nanpercentile(allv, 2), np.nanpercentile(allv, 98)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    im = None
    for ax, (name, fld) in zip(axes.ravel(), fields.items()):
        im = ax.pcolormesh(X, Y, fld, cmap=S.CMAP_SST, vmin=vmin, vmax=vmax,
                           shading="auto", rasterized=True)
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
        ax.set_title(name)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cb.set_label("SST (°C)")
    S.save(fig, "sst_seasonal_2022.png")


def _field_triple(results_pkl, t_index, out_name, geo=False):
    d = _load(results_pkl)
    rec = [r for r in d["results"] if r["t_index"] == t_index][0]
    U, X, Y = _data() if geo else (None, None, None)
    true, pred, err = rec["u_true"], rec["u_pred"], np.abs(rec["u_error"])
    vmin = np.nanpercentile(np.concatenate([true.ravel(), pred.ravel()]), 2)
    vmax = np.nanpercentile(np.concatenate([true.ravel(), pred.ravel()]), 98)
    evmax = np.nanpercentile(err.ravel(), 98)

    ny, nx = true.shape
    wide = nx / ny > 3
    if wide:   # thin strip -> stack rows
        fig, axes = plt.subplots(3, 1, figsize=(14, 6), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)

    im0 = S.plot_field(axes[0], true, S.CMAP_SST, vmin, vmax)
    axes[0].set_title("True SST")
    im1 = S.plot_field(axes[1], pred, S.CMAP_SST, vmin, vmax)
    axes[1].set_title("SINN prediction")
    im2 = S.plot_field(axes[2], err, S.CMAP_ERROR, 0, evmax)
    axes[2].set_title("Absolute error")

    if wide:
        S.cbar(fig, im0, axes[0], "SST (°C)", fraction=0.025, pad=0.01)
        S.cbar(fig, im1, axes[1], "SST (°C)", fraction=0.025, pad=0.01)
        S.cbar(fig, im2, axes[2], "|error| (°C)", fraction=0.025, pad=0.01)
    else:
        S.cbar(fig, im0, axes[0], "SST (°C)", fraction=0.046, pad=0.02)
        S.cbar(fig, im1, axes[1], "SST (°C)", fraction=0.046, pad=0.02)
        S.cbar(fig, im2, axes[2], "|error| (°C)", fraction=0.046, pad=0.02)
    S.save(fig, out_name)


def fig_field_v3():
    _field_triple(f"test_results_v3_seed{DEFAULT_SEED}.pkl", 1095,
                  "field_v3_seed42_late_t1095.png")


def fig_field_small_domain():
    _field_triple(f"test_results_small_domain_125_seed{DEFAULT_SEED}.pkl", 1095,
                  "field_small_domain_125_seed42_late_t1095.png")


def fig_spatial_error():
    z = np.load(os.path.join(RESULTS_DIR, "spatial_error_fields.npz"))
    s1, comb, grad = z["s1_error_avg"], z["combined_error_avg"], z["grad_mag"]
    evmax = np.nanpercentile(s1, 95)
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 11), constrained_layout=True)

    im0 = ax[0, 0].imshow(s1, cmap=S.CMAP_ERROR, origin="lower", vmin=0,
                          vmax=evmax, aspect="equal")
    ax[0, 0].set_title("Stage 1 — time-avg |error|")
    S.cbar(fig, im0, ax[0, 0], "MAE (°C)", fraction=0.046, pad=0.02)

    im1 = ax[0, 1].imshow(comb, cmap=S.CMAP_ERROR, origin="lower", vmin=0,
                          vmax=evmax, aspect="equal")
    ax[0, 1].set_title("Two-stage — time-avg |error|")
    S.cbar(fig, im1, ax[0, 1], "MAE (°C)", fraction=0.046, pad=0.02)

    im2 = ax[1, 0].imshow(grad, cmap=S.CMAP_SST, origin="lower", aspect="equal")
    ax[1, 0].set_title("Time-avg |∇SST|")
    S.cbar(fig, im2, ax[1, 0], "|∇SST| (°C / cell)", fraction=0.046, pad=0.02)

    gi = grad.copy().ravel(); ei = s1.copy().ravel()
    m = ~(np.isnan(gi) | np.isnan(ei))
    r = np.corrcoef(gi[m], ei[m])[0, 1]
    ax[1, 1].scatter(gi[m], ei[m], s=2, alpha=0.3, color=S.NAVY, rasterized=True)
    ax[1, 1].set_xlabel("|∇SST| (°C / cell)")
    ax[1, 1].set_ylabel("Stage 1 MAE (°C)")
    ax[1, 1].set_title(f"r = {r:.3f}")
    S.grid(ax[1, 1])
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xlabel("x"); a.set_ylabel("y")
    S.save(fig, "spatial_error_decomposition.png")
    print(f"    [spatial error] error-gradient r = {r:.3f}")


def fig_small_domain_region():
    v3 = _load(f"test_results_v3_seed{DEFAULT_SEED}.pkl")
    ny, nx = v3["results"][0]["u_true"].shape
    err = np.zeros((ny, nx)); n = 0
    for rr in v3["results"]:
        err += np.abs(rr["u_pred"] - rr["u_true"]); n += 1
    err /= n
    obs = np.zeros((ny, nx), bool)
    obs[:B_THICK, :] = obs[-B_THICK:, :] = obs[:, :B_THICK] = obs[:, -B_THICK:] = True
    err[obs] = np.nan

    d = _load("small_domain_results_125.pkl")
    cr = d["crop_region"]; cy, cx = d["error_centre"]
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    im = ax.imshow(err, cmap=S.CMAP_ERROR, origin="lower", aspect="auto",
                   vmin=0, vmax=np.nanpercentile(err, 98))
    S.cbar(fig, im, ax, "Time-averaged |error| (°C)", fraction=0.046, pad=0.02)
    rect = mpatches.Rectangle((cr["x_start"], cr["y_start"]),
                              cr["x_end"] - cr["x_start"],
                              cr["y_end"] - cr["y_start"],
                              fill=False, edgecolor=S.HIGHLIGHT, lw=2.5)
    ax.add_patch(rect)
    ax.plot(cx, cy, "x", color=S.HIGHLIGHT, ms=14, mew=3)
    ax.set_xlabel("grid column"); ax.set_ylabel("grid row")
    fig.tight_layout()
    S.save(fig, "small_domain_region_125.png")


def fig_latent_dims():
    """Needs tensorflow + trained v3 weights to recompute the latent field."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # The project pins scipy 1.11 (cg uses `tol`); this sandbox has scipy >=1.14
    # where that kwarg was renamed to `rtol`. Patch cg so the unmodified solver
    # source keeps working instead of silently returning a zero latent field.
    import scipy.sparse.linalg as _ssl
    if "rtol" in __import__("inspect").signature(_ssl.cg).parameters:
        _orig_cg = _ssl.cg
        def _cg_compat(A, b, **kw):
            if "tol" in kw:
                kw["rtol"] = kw.pop("tol")
            return _orig_cg(A, b, **kw)
        _ssl.cg = _cg_compat
    from config import SINN_V3_MODULE
    from utils import load_sinn_class, load_data, setup_sinn, load_sinn_weights

    data = load_data()
    solver = setup_sinn(load_sinn_class(SINN_V3_MODULE), data, seed=DEFAULT_SEED)
    load_sinn_weights(solver, f"v3_seed{DEFAULT_SEED}")
    rec = solver.reconstruct_field_at_timestep(730, use_fast_solver=True,
                                               return_latent=True)
    lat = np.asarray(rec["latent_field"])          # (ny, nx, r)
    r = lat.shape[2]

    ncol = 5
    nrow = int(np.ceil(r / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.4 * nrow))
    axes = np.atleast_2d(axes)
    for k in range(nrow * ncol):
        ax = axes.ravel()[k]
        if k < r:
            # per-panel symmetric scaling: latent coords are an arbitrary basis,
            # so each component is shown on its own scale to reveal structure.
            vmax = np.nanpercentile(np.abs(lat[:, :, k]), 99)
            im = ax.imshow(lat[:, :, k], cmap=S.CMAP_DIV, origin="lower",
                           vmin=-vmax, vmax=vmax, aspect="equal")
            ax.set_title(f"$\\ell_{{{k+1}}}$")
            ax.set_xticks([]); ax.set_yticks([])
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=9)
        else:
            ax.axis("off")
    fig.tight_layout()
    S.save(fig, "latent_dims_v3_seed42_mid_t730_crop.png")


def fig_combined_small_domain():
    """Full-domain true SST at t=1095 with crop box (left) + small-domain
    triple (right: true SST, prediction, absolute error at t=1095)."""
    v3 = _load(f"test_results_v3_seed{DEFAULT_SEED}.pkl")
    rec_full = next(r for r in v3["results"] if r["t_index"] == 1095)
    true_full = rec_full["u_true"]

    d = _load("small_domain_results_125.pkl")
    cr = d["crop_region"]
    cy, cx = d["error_centre"]

    sd = _load(f"test_results_small_domain_125_seed{DEFAULT_SEED}.pkl")
    rec_sd = next(r for r in sd["results"] if r["t_index"] == 1095)
    true_sd = rec_sd["u_true"]
    pred_sd = rec_sd["u_pred"]
    err_sd  = np.abs(rec_sd["u_error"])

    vmin_f  = np.nanpercentile(true_full, 2)
    vmax_f  = np.nanpercentile(true_full, 98)
    all_sd  = np.concatenate([true_sd.ravel(), pred_sd.ravel()])
    vmin_s  = np.nanpercentile(all_sd, 2)
    vmax_s  = np.nanpercentile(all_sd, 98)
    evmax_s = np.nanpercentile(err_sd.ravel(), 98)

    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    gs  = fig.add_gridspec(3, 2, width_ratios=[1.4, 1])

    ax_l  = fig.add_subplot(gs[:, 0])
    ax_r0 = fig.add_subplot(gs[0, 1])
    ax_r1 = fig.add_subplot(gs[1, 1])
    ax_r2 = fig.add_subplot(gs[2, 1])

    im_f = ax_l.imshow(true_full, cmap=S.CMAP_SST, vmin=vmin_f, vmax=vmax_f,
                        origin="lower", aspect="equal", interpolation="nearest")
    ax_l.set_title("True SST at $t=1095$ (full domain)")
    rect = mpatches.Rectangle(
        (cr["x_start"], cr["y_start"]),
        cr["x_end"] - cr["x_start"], cr["y_end"] - cr["y_start"],
        fill=False, edgecolor=S.HIGHLIGHT, lw=2.5)
    ax_l.add_patch(rect)
    ax_l.plot(cx, cy, "x", color=S.HIGHLIGHT, ms=14, mew=3)
    ax_l.set_xlabel("x"); ax_l.set_ylabel("y")
    S.cbar(fig, im_f, ax_l, "SST (°C)", fraction=0.046, pad=0.02)

    for ax, fld, cmap, vmin, vmax, title, clbl in [
            (ax_r0, true_sd, S.CMAP_SST,   vmin_s,  vmax_s,  "True SST",        "SST (°C)"),
            (ax_r1, pred_sd, S.CMAP_SST,   vmin_s,  vmax_s,  "SINN prediction", "SST (°C)"),
            (ax_r2, err_sd,  S.CMAP_ERROR,  0,       evmax_s, "Absolute error",  "|error| (°C)"),
    ]:
        im = ax.imshow(fld, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", aspect="equal", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        S.cbar(fig, im, ax, clbl, fraction=0.025, pad=0.01)

    S.save(fig, "combined_small_domain_125.png")


def fig_world_overview():
    """Global SST map with study-domain box. No cartopy: land shows via the
    NetCDF land mask (NaN) painted grey."""
    import netCDF4 as nc
    gpath = os.path.join(_MYFYP, "legacy", "sea_surf_temp", "sst.day.mean.2022.nc")
    ds = nc.Dataset(gpath)
    lat = ds.variables["lat"][:]
    lon = ds.variables["lon"][:]
    sst = np.ma.filled(np.ma.masked_invalid(ds.variables["sst"][0]), np.nan)
    ds.close()

    _, X, Y = _data()
    lon0, lon1 = float(X.min()), float(X.max())
    lat0, lat1 = float(Y.min()), float(Y.max())

    cmap = S.CMAP_SST.copy()
    cmap.set_bad(S.LAND_COLOUR)
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_facecolor(S.LAND_COLOUR)
    im = ax.pcolormesh(lon, lat, np.ma.masked_invalid(sst), cmap=cmap,
                       vmin=-2, vmax=32, shading="auto", rasterized=True)
    cb = S.cbar(fig, im, ax, "SST (°C)", fraction=0.024, pad=0.02, shrink=0.85)
    rect = mpatches.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                              fill=False, edgecolor=S.HIGHLIGHT, lw=2.5,
                              label="Study domain")
    ax.add_patch(rect)
    ax.set_xlim(0, 360); ax.set_ylim(-80, 80)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_yticks(np.arange(-60, 61, 30))
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    S.save(fig, "world_domain_overview.png")


# ===========================================================================
# driver
# ===========================================================================
ALL = {
    "test_mae_baselines": fig_test_mae_baselines,
    "twostage_mae": fig_twostage_mae,
    "small_domain_comparison": fig_small_domain_comparison,
    "training_loss": fig_training_loss,
    "patch_size": fig_patch_size,
    "latent_dim": fig_latent_dim,
    "boundary_noise": fig_boundary_noise,
    "psd": fig_boundary_psd,
    "enso": fig_enso_comparison,
    "seasonal": fig_seasonal,
    "field_v3": fig_field_v3,
    "combined_small": fig_combined_small_domain,
    "spatial_error": fig_spatial_error,
    "latent_dims": fig_latent_dims,
    "world": fig_world_overview,
}


def main(argv):
    keys = list(ALL)
    if argv:
        keys = [k for k in ALL if k in argv]
    print(f"Generating {len(keys)} figure(s) -> {S.OUT_DIR}")
    for k in keys:
        print(f"[{k}]")
        try:
            ALL[k]()
        except Exception as e:
            import traceback
            print(f"  !! FAILED {k}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main(sys.argv[1:])
