from pickletools import long1
from scipy import interpolate
import numpy as np
import netCDF4 as nc
import xarray as xr

# load gfs forecast
path = '/tera03/lilu/work/CLFS/outputs/'
# load gfs (365, 10, 721, 1440)
gfs = np.load(path + 'GFS_0p25_f16_swvl1_2015.npy')
print("gfs forecast shape is {}".format(gfs.shape))
lat_gfs = np.arange(90, -90.1, -0.25)
lon_gfs = np.arange(0, 359.8, 0.25)
print("lat shape is {} and lon shape is {}".format(lat_gfs.shape, lon_gfs.shape))

idx_lat = np.where((lat_gfs>14.7) & (lat_gfs<53.5))[0]
idx_lon = np.where((lon_gfs>72.3) & (lon_gfs<135))[0]

gfs = gfs[:, :, idx_lat][:, :, :, idx_lon]
lat_gfs = lat_gfs[idx_lat]
lon_gfs = lon_gfs[idx_lon]
print(lon_gfs.shape)
print(lat_gfs.shape)
print(gfs.shape)

f = nc.Dataset(path + 'SMAP_L4_SSM_20150531.nc')
lat_smap = np.array(f['latitude'][:])
lon_smap = np.array(f['longitude'][:])
print("lat shape is {} and lon shape is {}".format(lat_smap.shape, lon_smap.shape))



time = np.arange(1, gfs.shape[0]+1)
x, y = np.meshgrid(lon_gfs, lat_gfs)

gfs_0p1 = np.full((gfs.shape[0], 16, len(lat_smap), len(lon_smap)), np.nan)

for i in range(16):
    #for j in range(365):
    print(i)
    da = xr.DataArray(gfs[:, i], [('time', time), ('lat', lat_gfs), ('lon', lon_gfs)])
    #tmp = interpolate.interp2d(x,y, gfs[j, i], kind='linear')
    tmp = da.interp(time=time, lat=lat_smap, lon=lon_smap)

    for j in range(gfs.shape[0]):
        print(j)
        # interplot
        tmp = tmp[j]
        tmp[np.isnan(tmp)] = np.nanmean(tmp)
        gfs_0p1[j, i] = np.array(tmp)

np.save('GFS_0p1_f16_swvl1_2015.npy', gfs_0p1)
