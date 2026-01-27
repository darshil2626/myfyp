import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# ---------------------------------------------------------
# CONFIG – EDIT THESE
# ---------------------------------------------------------
data_dir = Path("tempo_no2_l3_data")   # folder with your TEMPO L3 granules
output_gif = "tempo_animation.gif"

# Choose your region (example: around Eastern US)
# Bigger rectangular region
POI_lat = 39.5
POI_lon = -95.5

dlat = 5.5
dlon = 7.5

lat_min = POI_lat - dlat   # 27°
lat_max = POI_lat + dlat   # 51°
lon_min = POI_lon - dlon   # -95°
lon_max = POI_lon + dlon   # -59°


# ---------------------------------------------------------
# LOAD ALL FILES & BUILD FRAMES
# ---------------------------------------------------------
files = sorted(data_dir.glob("*.nc"))
if not files:
    raise RuntimeError(f"No .nc files found in {data_dir}")

frames = []       # list of 2D arrays (lat x lon)
frame_times = []  # list of strings
lat_reg = None
lon_reg = None

for fn in files:
    print("Reading:", fn.name)

    ds_root = xr.open_dataset(fn)
    ds_prod = xr.open_dataset(fn, group="product")

    # 1D lat/lon from root group
    lat_1d = ds_root["latitude"].values
    lon_1d = ds_root["longitude"].values

    # indices of our region in the 1D coords
    lat_idx = np.where((lat_1d >= lat_min) & (lat_1d <= lat_max))[0]
    lon_idx = np.where((lon_1d >= lon_min) & (lon_1d <= lon_max))[0]

    if lat_idx.size == 0 or lon_idx.size == 0:
        print("  -> no grid points in this file within bbox, skipping")
        continue

    # Subset 1D coords for region (we reuse these for all frames)
    lat_reg = lat_1d[lat_idx]
    lon_reg = lon_1d[lon_idx]

    # Data & flags (T, Y, X)
    trop_all  = ds_prod["vertical_column_troposphere"].values
    strat_all = ds_prod["vertical_column_stratosphere"].values
    qf_all    = ds_prod["main_data_quality_flag"].values
    times     = ds_prod["time"].values

    fv_trop  = ds_prod["vertical_column_troposphere"].attrs.get("_FillValue", -1e30)
    fv_strat = ds_prod["vertical_column_stratosphere"].attrs.get("_FillValue", -1e30)

    # Loop over all timesteps in this file
    for ti in range(trop_all.shape[0]):
        trop  = trop_all[ti]
        strat = strat_all[ti]
        qf    = qf_all[ti]

        total = trop + strat

        # Crop to region indices (Y, X)
        total_reg = total[np.ix_(lat_idx, lon_idx)]
        qf_reg    = qf[np.ix_(lat_idx, lon_idx)]
        trop_reg  = trop[np.ix_(lat_idx, lon_idx)]
        strat_reg = strat[np.ix_(lat_idx, lon_idx)]

        # Relaxed QA: qf <= 1, drop only fill values
        good_mask = (
            (qf_reg <= 1) &
            (trop_reg != fv_trop) &
            (strat_reg != fv_strat)
        )

        total_clean = np.where(good_mask, total_reg, np.nan)
        frames.append(total_clean)

        # Nice time label
        # Robust timestamp conversion for TEMPO L3
        t_val = times[ti]

        # Case 1: numpy datetime64
        if isinstance(t_val, np.datetime64):
            t_str = np.datetime_as_string(t_val, unit="m")

        # Case 2: Python datetime object
        elif hasattr(t_val, "isoformat"):
            t_str = t_val.isoformat(timespec="minutes")

        # Case 3: Already a string
        else:
            t_str = str(t_val)

        frame_times.append(t_str)

# If absolutely no frames, bail
if not frames:
    raise RuntimeError("No frames built – check region and files.")

frames = np.array(frames)  # shape: (N_frames, nlat, nlon)
frame_times = np.array(frame_times)

# Sort frames by time
order = np.argsort(frame_times)
frames = frames[order]
frame_times = frame_times[order]

# Build 2D lat/lon for plotting
lon2d, lat2d = np.meshgrid(lon_reg, lat_reg)

print("Total frames:", len(frames))
print("Frame shape:", frames[0].shape)

# ---------------------------------------------------------
# ANIMATION
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

# Use percentiles for robust colour limits
vmin = np.nanpercentile(frames, 5)
vmax = np.nanpercentile(frames, 95)

# Initial frame
img = ax.pcolormesh(
    lon2d, lat2d, frames[0],
    shading="auto", cmap="viridis", vmin=vmin, vmax=vmax
)
cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Total NO₂ column")

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
title = ax.set_title('NO2 Pollution Spread on 1/9/24')

def update(i):
    img.set_array(frames[i].ravel())
    return [img]

ani = animation.FuncAnimation(
    fig, update, frames=len(frames), interval=500, blit=False
)

print("Saving GIF...")
ani.save(output_gif, writer="pillow", dpi=150)
print("Saved:", output_gif)
