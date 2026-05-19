"""
World map of global SST at t=0 with the study domain highlighted.
Global SST source: legacy/sea_surf_temp/sst.day.mean.2022.nc
"""

import os
import pickle
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config import DATA_PATH, FIGURES_DIR, ensure_dirs

ensure_dirs()

_HERE = os.path.dirname(os.path.abspath(__file__))
GLOBAL_SST_PATH = os.path.join(_HERE, "legacy", "sea_surf_temp", "sst.day.mean.2022.nc")

# ---- Global SST at t=0 ----
ds = nc.Dataset(GLOBAL_SST_PATH)
lat_g = ds.variables["lat"][:]          # (720,)
lon_g = ds.variables["lon"][:]          # (1440,)  0–360
sst_g = np.ma.filled(
    np.ma.masked_invalid(ds.variables["sst"][0]),   # (720, 1440)
    fill_value=np.nan,
)
ds.close()

# ---- Study domain bounds (from project data) ----
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)
X = np.asarray(data["X"], dtype=np.float32)
Y = np.asarray(data["Y"], dtype=np.float32)
x_vec = X[0]
y_vec = Y[:, 0]
lon_min, lon_max = float(x_vec[0]),  float(x_vec[-1])
lat_min, lat_max = float(y_vec[0]),  float(y_vec[-1])

# ---- Figure ----
proj     = ccrs.PlateCarree(central_longitude=180)
data_crs = ccrs.PlateCarree()

fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={"projection": proj})

ax.set_global()
ax.add_feature(cfeature.LAND,      facecolor="#d0d0d0", zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5,       zorder=3)
ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
             alpha=0.5, linestyle="--")

# Global SST
im = ax.pcolormesh(
    lon_g, lat_g, sst_g,
    cmap="RdYlBu_r",
    transform=data_crs,
    zorder=1,
    shading="auto",
    vmin=-2, vmax=32,
)
cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                    fraction=0.02, pad=0.04, shrink=0.75)
cbar.set_label("SST (°C)", fontsize=11)

# Study domain box
domain_rect = mpatches.Rectangle(
    (lon_min, lat_min),
    lon_max - lon_min,
    lat_max - lat_min,
    linewidth=2.5,
    edgecolor="cyan",
    facecolor="none",
    label="Study domain",
    transform=data_crs,
    zorder=4,
)
ax.add_patch(domain_rect)

ax.legend(loc="lower left", fontsize=12, framealpha=0.85)
ax.set_title(
    "Global SST — 1 Jan 2022  |  Red box: study domain (Eastern Tropical Pacific)",
    fontsize=13, fontweight="bold",
)

out_path = f"{FIGURES_DIR}/world_domain_overview.png"
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")
