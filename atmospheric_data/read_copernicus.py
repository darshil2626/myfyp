import xarray as xr
import numpy as np

# Open PRODUCT group
ds = xr.open_dataset("data_s5p_no2\S5P_OFFL_L2__NO2____20240101T110759_20240101T124929_32221_03_020600_20240103T033227.nc.zip", group="PRODUCT")

no2 = ds["nitrogendioxide_tropospheric_column"]
qa  = ds["qa_value"]
lat = ds["latitude"]
lon = ds["longitude"]

qa_min = 0.75
good = qa >= qa_min

no2_good = no2.where(good)
lat_good = lat.where(good)
lon_good = lon.where(good)

# Approx London bounding box
lat_min, lat_max = 51.2, 51.8
lon_min, lon_max = -0.6, 0.4

region = (
    (lat_good >= lat_min) & (lat_good <= lat_max) &
    (lon_good >= lon_min) & (lon_good <= lon_max)
)

no2_reg = no2_good.where(region)
lat_reg = lat_good.where(region)
lon_reg = lon_good.where(region)

import matplotlib.pyplot as plt
import numpy as np

# Flatten and drop NaNs for clean plotting
lon_flat = lon_reg.values.flatten()
lat_flat = lat_reg.values.flatten()
no2_flat = no2_reg.values.flatten()

mask = np.isfinite(lon_flat) & np.isfinite(lat_flat) & np.isfinite(no2_flat)

lon_flat = lon_flat[mask]
lat_flat = lat_flat[mask]
no2_flat = no2_flat[mask]

plt.figure(figsize=(6,6))
sc = plt.scatter(
    lon_flat,
    lat_flat,
    c=no2_flat,
    s=25,                # ← bigger markers
    cmap="viridis"       # ← optional, nicer colormap
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Spread of NO2 in London on 3/1/2024")

cbar = plt.colorbar(sc)              # ← add colorbar
cbar.set_label("NO₂ tropospheric column")
plt.tight_layout()
plt.show()
