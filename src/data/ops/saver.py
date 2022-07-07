import os
import netCDF4 as nc
import numpy as np


def nc_saver(save_path, var_name, date, lon_2d, lat_2d, X):

    # save to nc files
    filename = 'SMAP_L4_{var_name}_{year}{month:02}{day:02}.nc'.format(
        var_name=var_name, year=date.year, month=date.month, day=date.day)

    # judge already exist file
    if os.path.exists(save_path + filename):
        print('[HRSEPP][IO]SMAP {} of {} already exist'.format(var_name, date))
    else:
        f = nc.Dataset(save_path + filename, 'w', format='NETCDF4')

        f.createDimension('longitude', size=X.shape[-1])
        f.createDimension('latitude', size=X.shape[-2])
        f.createDimension('feature', size=X.shape[-3])

        lon = f.createVariable('longitude', 'f4', dimensions='longitude')
        lat = f.createVariable('latitude', 'f4', dimensions='latitude')
        data = f.createVariable(var_name,
                                'f4',
                                dimensions=('feature', 'latitude',
                                            'longitude'))
        print(lon_2d.shape)
        print(lat_2d.shape)
        print(X.shape)
        #FIXME: Change stupid transform code
        lon[:], lat[:], data[:] = np.array(lon_2d)[0, :], np.array(
            lat_2d)[:, 0], X

        f.close()
