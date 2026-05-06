import xarray as xr

# Load dataset
ds = xr.open_dataset("sst.day.mean.2024.nc")

sst = ds['sst']
lat = ds['lat']
lon = ds['lon']

sst0 = sst.isel(time=0)
