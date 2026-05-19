#!/usr/bin/env python3
"""
animate_results.py

Create a GIF animation from a test_results_*.pkl file in results/.
Shows ground truth, prediction, and absolute error side by side.
The figure aspect ratio is set to match the spatial domain (ny × nx)
so there is no stretching.

Usage:
    python animate_results.py results/test_results_v3_seed42.pkl
    python animate_results.py results/test_results_v8_seed42.pkl -o gifs/v8.gif
    python animate_results.py results/test_results_v3_seed42.pkl --stride 5 --fps 15
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_results(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if "results" not in data:
        raise ValueError(
            f"'{pkl_path}' has no 'results' key — expected a test_results_*.pkl file.\n"
            f"Found keys: {list(data.keys())}"
        )
    results = data["results"]
    if not results or "u_true" not in results[0]:
        raise ValueError(
            f"'results' list in '{pkl_path}' does not contain per-timestep fields "
            f"(u_true / u_pred). Cannot animate."
        )
    return data


# If a single panel's aspect ratio exceeds this, switch to vertical stacking.
_VERTICAL_THRESHOLD = 2.5
# Maximum figure width used when stacking vertically.
_MAX_VERTICAL_W = 10.0


def build_figure(ny: int, nx: int, panel_h_in: float = 4.0):
    """
    Build a figure with 3 data panels + 3 colorbars whose data axes match
    the domain aspect ratio (nx/ny) so there is no stretching.

    Layout is chosen automatically:
    - Horizontal (side-by-side) when nx/ny <= 2.5
    - Vertical   (stacked)      when nx/ny >  2.5

    Returns fig, (ax_true, ax_pred, ax_err), (cax_true, cax_pred, cax_err).
    """
    if nx / ny > _VERTICAL_THRESHOLD:
        return _build_vertical(ny, nx)
    return _build_horizontal(ny, nx, panel_h_in)


def _build_horizontal(ny: int, nx: int, panel_h_in: float):
    """3 panels side-by-side, colorbars to the right of each."""
    aspect     = nx / ny
    panel_w_in = aspect * panel_h_in

    cbar_w_in = max(0.15, panel_w_in * 0.05)
    pad_in    = max(0.06, panel_w_in * 0.02)
    gap_in    = max(0.25, panel_w_in * 0.07)
    margin_l  = 0.45
    margin_r  = 0.15
    margin_b  = 0.48
    margin_t  = 0.60

    n = 3
    total_w = margin_l + n * panel_w_in + n * (pad_in + cbar_w_in) + (n - 1) * gap_in + margin_r
    total_h = margin_b + panel_h_in + margin_t

    fig = plt.figure(figsize=(total_w, total_h))

    def x2f(x): return x / total_w
    def y2f(y): return y / total_h

    bottom = y2f(margin_b)
    height = y2f(panel_h_in)
    cursor = margin_l
    data_axes, cbar_axes = [], []

    for _ in range(n):
        ax  = fig.add_axes([x2f(cursor), bottom, x2f(panel_w_in), height])
        data_axes.append(ax)
        cursor += panel_w_in + pad_in

        cax = fig.add_axes([x2f(cursor), bottom, x2f(cbar_w_in), height])
        cbar_axes.append(cax)
        cursor += cbar_w_in + gap_in

    return fig, tuple(data_axes), tuple(cbar_axes)


def _build_vertical(ny: int, nx: int, max_w: float = _MAX_VERTICAL_W):
    """3 panels stacked top-to-bottom, colorbars to the right of each."""
    aspect    = nx / ny
    cbar_w_in = 0.20
    pad_in    = 0.08
    gap_h_in  = 0.35
    margin_l  = 0.45
    margin_r  = 0.15
    margin_b  = 0.48
    margin_t  = 0.60

    panel_w = max_w - margin_l - margin_r - pad_in - cbar_w_in
    panel_h = panel_w / aspect

    n       = 3
    total_w = max_w
    total_h = margin_t + n * panel_h + (n - 1) * gap_h_in + margin_b

    fig = plt.figure(figsize=(total_w, total_h))

    def x2f(x): return x / total_w
    def y2f(y): return y / total_h

    data_axes, cbar_axes = [], []

    for i in range(n):
        # i=0 → top panel, i=2 → bottom panel
        bottom = y2f(margin_b + (n - 1 - i) * (panel_h + gap_h_in))
        height = y2f(panel_h)

        ax  = fig.add_axes([x2f(margin_l), bottom, x2f(panel_w), height])
        data_axes.append(ax)

        cax = fig.add_axes([x2f(margin_l + panel_w + pad_in), bottom, x2f(cbar_w_in), height])
        cbar_axes.append(cax)

    return fig, tuple(data_axes), tuple(cbar_axes)


def make_gif(
    pkl_path: str,
    output_path: str,
    fps: int = 10,
    stride: int = 1,
    panel_h: float = 4.0,
):
    data = load_results(pkl_path)
    all_results = data["results"]
    model_name  = data.get("model", os.path.splitext(os.path.basename(pkl_path))[0])
    seed        = data.get("seed", "?")

    frames = all_results[::stride]
    print(f"Animating {len(frames)} frames (stride={stride}) from {pkl_path}")

    ny, nx = frames[0]["u_true"].shape
    layout = "vertical" if nx / ny > _VERTICAL_THRESHOLD else "horizontal"
    print(f"Domain: {ny} rows x {nx} cols  (aspect {nx/ny:.3f})  layout={layout}")

    # ---- Global colour scales (computed over sampled frames only for speed) ----
    all_true = np.stack([r["u_true"] for r in frames])
    vmin = float(np.nanpercentile(all_true, 1))
    vmax = float(np.nanpercentile(all_true, 99))

    all_err = np.stack([np.abs(r["u_error"]) for r in frames])
    vmax_err = float(np.nanpercentile(all_err, 98))

    # ---- Build figure ----
    fig, (ax_t, ax_p, ax_e), (cax_t, cax_p, cax_e) = build_figure(ny, nx, panel_h)

    im_kw = dict(origin="lower", aspect="auto", interpolation="nearest")
    field_kw = dict(cmap="RdYlBu_r", vmin=vmin, vmax=vmax, **im_kw)
    err_kw   = dict(cmap="hot_r",    vmin=0,    vmax=vmax_err, **im_kw)

    def frame_arrays(r):
        u_t = r["u_true"].copy()
        u_p = r["u_pred"].copy()
        u_e = np.abs(r["u_error"].copy())
        bm  = r.get("boundary_mask")
        if bm is not None:
            u_e[bm] = np.nan   # boundary is observed — no prediction error
        return u_t, u_p, u_e

    u_t0, u_p0, u_e0 = frame_arrays(frames[0])

    im_t = ax_t.imshow(u_t0, **field_kw)
    im_p = ax_p.imshow(u_p0, **field_kw)
    im_e = ax_e.imshow(u_e0, **err_kw)

    cb_t = fig.colorbar(im_t, cax=cax_t, label="SST (°C)")
    cb_p = fig.colorbar(im_p, cax=cax_p, label="SST (°C)")
    cb_e = fig.colorbar(im_e, cax=cax_e, label="|Error| (°C)")
    for cb in (cb_t, cb_p, cb_e):
        cb.ax.tick_params(labelsize=7)
        cb.set_label(cb.ax.get_ylabel(), fontsize=8)

    ax_t.set_title("Ground Truth",  fontsize=9, pad=4)
    ax_p.set_title("Prediction",    fontsize=9, pad=4)
    ax_e.set_title("|Error|",       fontsize=9, pad=4)

    for ax in (ax_t, ax_p, ax_e):
        ax.set_xlabel("x (col)", fontsize=7)
        ax.set_ylabel("y (row)", fontsize=7)
        ax.tick_params(labelsize=6)

    title_obj = fig.text(
        0.5, 1 - 0.10 / (fig.get_size_inches()[1]),
        "", ha="center", va="top", fontsize=10, fontweight="bold"
    )

    def update(fi):
        r = frames[fi]
        u_t, u_p, u_e = frame_arrays(r)
        im_t.set_data(u_t)
        im_p.set_data(u_p)
        im_e.set_data(u_e)
        mae = r.get("mae", float(np.nanmean(np.abs(r["u_error"]))))
        t   = r.get("t_index", fi * stride)
        title_obj.set_text(
            f"{model_name}  |  seed={seed}  |  t={t}  |  MAE={mae:.4f} °C"
        )
        return im_t, im_p, im_e

    update(0)

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=1000 // fps,
        blit=False,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved: {output_path}  ({len(frames)} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(
        description="Animate a test_results_*.pkl file as a GIF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pkl", help="Path to a test_results_*.pkl file in results/")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output GIF path (default: gifs/<pkl_stem>.gif)",
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second (default: 10)",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Use every Nth frame to reduce GIF size (default: 1 = all frames)",
    )
    parser.add_argument(
        "--panel-height", type=float, default=4.0,
        help="Height of each data panel in inches (default: 4.0)",
    )
    args = parser.parse_args()

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.pkl))[0]
        args.output = os.path.join("gifs", f"{stem}.gif")

    make_gif(
        args.pkl,
        args.output,
        fps=args.fps,
        stride=args.stride,
        panel_h=args.panel_height,
    )


if __name__ == "__main__":
    main()
