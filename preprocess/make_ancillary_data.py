import xarray as xr
import netCDF4 as nc
import numpy as np
import xesmf as xe


path = '/tera08/lilu/HybridHydro/input/'

# --------------------------------------------------------
# load DEM in 0.1 degree grids
with xr.open_dataset(path+'DEM_0p1.nc') as f:
    elv = np.array(f.elv)
    lat_in, lon_in = np.array(f.lat), np.array(f.lon)

# load SMAP in EASE 9km grids
f = nc.Dataset(path+'EASE_9km_grids.h5', 'r')
lat_out, lon_out = f['cell_lat'][:,0], f['cell_lon'][0,:]

# construct in/out grids
grid_in = {"lon": lon_in, "lat": lat_in}
grid_out = {"lon": lon_out, "lat": lat_out}

print(lat_out.shape)
print(lon_out.shape)

# regrid from (1800,3600)-(1624,3856)
regridder = xe.Regridder(grid_in, grid_out, "bilinear", reuse_weights=True)
tp_resampled = regridder(elv)

# crop China
idx_lat = np.where((lat_out>14.7) & (lat_out<53.5))[0]
idx_lon = np.where((lon_out>72.3) & (lon_out<135))[0]
tp_resampled = tp_resampled[idx_lat][:, idx_lon]
lon_out = lon_out[idx_lon]
lat_out = lat_out[idx_lat]
print(tp_resampled.shape)

# save EASE CN
f = nc.Dataset('CN_EASE_9km.nc', 'w', format='NETCDF4')
f.createDimension('longitude', size=tp_resampled.shape[-1])
f.createDimension('latitude', size=tp_resampled.shape[-2])
lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
lon0[:], lat0[:] = lon_out, lat_out
f.close()

# save China
f = nc.Dataset('DEM_CN_EASE_9km.nc', 'w', format='NETCDF4')
f.createDimension('longitude', size=tp_resampled.shape[-1])
f.createDimension('latitude', size=tp_resampled.shape[-2])
lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
data = f.createVariable('dem', 'f4', dimensions=('latitude','longitude'))
lon0[:], lat0[:], data[:] = lon_out, lat_out, tp_resampled
f.close()


# --------------------------------------------------------
# load LC in 0.1 degree grids
with xr.open_dataset(path+'LC_0p1.nc') as f:
    landcover = np.array(f['class'][:])

# regrid use nearest interp
regridder = xe.Regridder(grid_in, grid_out, "nearest_s2d", reuse_weights=True)
tp_resampled = regridder(landcover)

# crop China
tp_resampled = tp_resampled[idx_lat][:, idx_lon]
print(tp_resampled.shape)

# save
f = nc.Dataset('LC_CN_EASE_9km.nc', 'w', format='NETCDF4')
f.createDimension('longitude', size=tp_resampled.shape[-1])
f.createDimension('latitude', size=tp_resampled.shape[-2])
lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
data = f.createVariable('land_cover', 'f4', dimensions=('latitude','longitude'))
lon0[:], lat0[:], data[:] = lon_out, lat_out, tp_resampled
f.close()

