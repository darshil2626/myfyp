import earthaccess
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# 1) Login to Earthdata
# -----------------------------
# This will pop up a browser or prompt once, and then cache a token.
auth = earthaccess.login(persist=True)

# -----------------------------
# 2) Search for TEMPO NO2 L3
# -----------------------------
# This matches the NASA example:
#   short_name = "TEMPO_NO2_L3"
#   version    = "V03"
short_name = "TEMPO_NO2_L3"
version = "V03"

# Time range (UTC) – you can change this
date_start = "2024-09-01 00:00:00"
date_end   = "2024-09-01 23:59:59"

# More inward, ocean-free Central US region

POI_lat = 39.5
POI_lon = -95.5

dlat = 5.5
dlon = 7.5

bbox = (
    POI_lon - dlon,
    POI_lat - dlat,
    POI_lon + dlon,
    POI_lat + dlat,
)  # (min_lon, min_lat, max_lon, max_lat)

print("Searching TEMPO NO2 L3 granules...")
results = earthaccess.search_data(
    short_name=short_name,
    version=version,
    temporal=(date_start, date_end),
    bounding_box=bbox,
)

print(f"Found {len(results)} granules")

if not results:
    raise SystemExit("No TEMPO granules found for this time/region. Try another date.")

# -----------------------------
# 3) Download a couple of files
# -----------------------------
out_dir = Path("tempo_no2_l3_data")
out_dir.mkdir(exist_ok=True)

# For now, just download 2 granules for testing
to_download = results

print("Downloading granules...")
files = earthaccess.download(to_download, local_path=str(out_dir))
print("Downloaded files:")
for f in files:
    print("  ", f)

# -----------------------------
# 4) Open one file and inspect
# -----------------------------
fn = files[0]
print("\nOpening file:", fn)

# TEMPO L3 layout (from NASA example):
# - root group: latitude, longitude (1D)
# - 'product' group: NO2 columns & QA flags
ds_root = xr.open_dataset(fn)              # root group (lat, lon)
ds_prod = xr.open_dataset(fn, group="product")  # 'product' group

lat = ds_root["latitude"].values           # shape (nlat,)
lon = ds_root["longitude"].values          # shape (nlon,)

# NO2 columns (3D: time, lat, lon) – we take time index 0
trop = ds_prod["vertical_column_troposphere"].isel(time=0)
strat = ds_prod["vertical_column_stratosphere"].isel(time=0)
qf   = ds_prod["main_data_quality_flag"].isel(time=0)

# Fill values
fv_trop = trop.attrs.get("_FillValue", -1e30)
fv_strat = strat.attrs.get("_FillValue", -1e30)

# -----------------------------
# 5) Mask “good” pixels and subset region
# -----------------------------
# Good data mask as in NASA example
good_mask = (
    (qf == 0)
    & (trop != fv_trop)
    & (strat != fv_strat)
    & (trop > 0.0)
    & (strat > 0.0)
)

total_no2 = trop + strat  # total column

# -----------------------------
# 5) Build masks in latitude/longitude space
# -----------------------------
lat_min = POI_lat - dlat
lat_max = POI_lat + dlat
lon_min = POI_lon - dlon
lon_max = POI_lon + dlon

# Ensure lat_root / lon_root are 2D grids
lat2d = np.array(lat)
lon2d = np.array(lon)

# If they are 1D, turn into 2D with meshgrid
if lat2d.ndim == 1 and lon2d.ndim == 1:
    lon2d, lat2d = np.meshgrid(lon2d, lat2d)

# Basic sanity check
print("trop shape:", trop.shape)
print("lat2d shape:", lat2d.shape)
print("lon2d shape:", lon2d.shape)

# -----------------------------
# 5) Region + relaxed QA mask
# -----------------------------
lat_min = POI_lat - dlat
lat_max = POI_lat + dlat
lon_min = POI_lon - dlon
lon_max = POI_lon + dlon

region_mask = (
    (lat2d >= lat_min) & (lat2d <= lat_max) &
    (lon2d >= lon_min) & (lon2d <= lon_max)
)

# Relaxed "good" mask:
# - use qf <= 1 (good or suspect)
# - drop only fill values, keep negatives
good_mask = (
    (qf <= 1) &
    (trop != fv_trop) & (strat != fv_strat)
)

combined = region_mask & good_mask.values

print("Total pixels:", combined.size)
print("Pixels in region:", region_mask.sum())
print("Good pixels overall (qf<=1):", good_mask.sum().item())
print("Good pixels in region (qf<=1):", combined.sum())

if combined.sum() == 0:
    print("Still no good pixels in this bbox – try a slightly bigger dlat/dlon or a different POI.")
else:
    lat_pts = lat2d[combined]
    lon_pts = lon2d[combined]
    no2_pts = total_no2.values[combined]

    # -----------------------------
    # 6) Plot
    # -----------------------------
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        lon_pts,
        lat_pts,
        c=no2_pts,
        s=5,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("TEMPO total NO₂ column (subset, qf ≤ 1)")
    cbar = plt.colorbar(sc)
    cbar.set_label("NO₂ column (molecules/cm²)")
    plt.tight_layout()
    plt.show()

