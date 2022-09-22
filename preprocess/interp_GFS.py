from pickletools import long1
from scipy import interpolate
import numpy as np
import netCDF4 as nc
import xarray as xr

# load gfs forecast
path = '/tera03/lilu/work/CLFS/inputs/'
# load gfs (365, 10, 721, 1440)
gfs = np.load(path + 'GFS_0p1_f16_swvl1_2015-2017.npy')
print("gfs forecast shape is {}".format(gfs.shape))
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
