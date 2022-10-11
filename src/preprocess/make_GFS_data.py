from scipy import interpolate
import numpy as np
import netCDF4 as nc
import xarray as xr


path = '/tera03/lilu/work/CLFS/outputs/'

### downscaling of GFS to EASE-9km grids
# load gfs (365, 10, 721, 1440) & crop China regions
gfs = np.load(path + 'GFS_0p25_f16_swvl1_2015.npy')
print("gfs forecast shape is {}".format(gfs.shape))
lat_gfs = np.arange(90, -90.1, -0.25)
lon_gfs = np.arange(0, 359.8, 0.25)
idx_lat = np.where((lat_gfs>14.7) & (lat_gfs<53.5))[0]
idx_lon = np.where((lon_gfs>72.3) & (lon_gfs<135))[0]
gfs = gfs[:, :, idx_lat][:, :, :, idx_lon]
lat_gfs = lat_gfs[idx_lat]
lon_gfs = lon_gfs[idx_lon]

# load ease-9km grids
f = nc.Dataset(path + 'SMAP_L4_SSM_20150531.nc')
lat_smap = np.array(f['latitude'][:])
lon_smap = np.array(f['longitude'][:])
time = np.arange(1, gfs.shape[0]+1)
x, y = np.meshgrid(lon_gfs, lat_gfs)

# downscaling by mean & save
gfs_0p1 = np.full((gfs.shape[0], 16, len(lat_smap), len(lon_smap)), np.nan)
for i in range(16):
    da = xr.DataArray(gfs[:, i], [('time', time), ('lat', lat_gfs), ('lon', lon_gfs)])
    tmp = da.interp(time=time, lat=lat_smap, lon=lon_smap)
    for j in range(gfs.shape[0]):
        print(j)
        # interplot
        tmp = tmp[j]
        tmp[np.isnan(tmp)] = np.nanmean(tmp)
        gfs_0p1[j, i] = np.array(tmp)
np.save('GFS_0p1_f16_swvl1_2015.npy', gfs_0p1)


### interp of GFS
gfs = np.load(path + 'GFS_0p1_f16_swvl1_2015-2017.npy')
gfs_0p1 = np.full((gfs.shape[0], 16, gfs.shape[2], gfs.shape[3]), np.nan)
time = np.arange(gfs.shape[0])

for i in range(16):
    for lat in range(gfs.shape[2]):
        for lon in range(gfs.shape[3]):
            print(i, lat, lon)
            tmp = gfs[:, i, lat, lon]
            if np.isnan(tmp).all():
                pass
            else:
                tmp = xr.DataArray(tmp, [('time', time)])
                tmp = tmp.interpolate_na(dim='time')
                gfs[:, i, lat, lon] = tmp
                print(np.isnan(tmp).any())
for i in range(16):
    for j in range(gfs.shape[0]):
        print(i, j)
        # interplot
        tmp = gfs[j, i]
        tmp[np.isnan(tmp)] = np.nanmean(tmp)
        gfs_0p1[j, i] = np.array(tmp)
np.save('GFS_0p1_f16_swvl1_interp_2015-2017.npy', gfs_0p1)
