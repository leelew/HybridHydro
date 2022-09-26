import numpy as np
import matplotlib.pyplot as plt
import glob
import xarray as xr
import netCDF4 as nc
import os
from config import parse_args


def postprocess(configs):
    # init
    N = 365-configs.len_input-configs.len_out-configs.window_size
    pred = np.full((N,configs.len_out, 448, 672), np.nan)
    smap = np.full((365, 448, 672), np.nan)

    lon = [0, 112, 224, 336, 448, 560,
        0, 112, 224, 336, 448, 560,
        0, 112, 224, 336, 448, 560,
        0, 112, 224, 336, 448, 560]
    lat = [0, 0, 0, 0, 0, 0,
        112, 112, 112, 112, 112, 112,
        224, 224, 224, 224, 224, 224,
        336, 336, 336, 336, 336, 336]

    # concat
    for i in range(1, 25):
        print(i)
        path = configs.outputs_path+'forecast/'+configs.model_name+'/'+str(i)+'/'
        pred[:, :, lat[i-1]:lat[i-1]+112, lon[i-1]:lon[i-1]+112] = \
            np.load(path+configs.model_name+'_0p1_f16_{id:02}.npy'.format(id=i))[:,:,:,:,0]
        smap[:, lat[i-1]:lat[i-1]+112, lon[i-1]:lon[i-1]+112] = \
            np.load(path+'SMAPL4_0p1_f16_{id:02}.npy'.format(id=i))[:,0,:,:]
        

    # save to nc
    with xr.open_dataset(configs.inputs_path+"CN_EASE_9km.nc") as f:
        lat, lon = np.array(f.latitude), np.array(f.longitude)
    
    f = nc.Dataset(configs.model_name+'_0p1_f16.nc', 'w', format='NETCDF4')
    f.createDimension('longitude', size=pred.shape[-1])
    f.createDimension('latitude', size=pred.shape[-2])
    f.createDimension('forecast_scale', size=pred.shape[-3])
    f.createDimension('samples', size=pred.shape[-4])
    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable('swvl', 'f4', 
        dimensions=('samples','forecast_scale','latitude','longitude'))
    lon0[:], lat0[:], data[:] = lon, lat, pred
    f.close()

    f = nc.Dataset('SMAPL4_0p1.nc', 'w', format='NETCDF4')
    f.createDimension('longitude', size=smap.shape[-1])
    f.createDimension('latitude', size=smap.shape[-2])
    f.createDimension('samples', size=smap.shape[-3])
    lon0 = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat0 = f.createVariable('latitude', 'f4', dimensions='latitude')
    data = f.createVariable('swvl', 'f4', 
        dimensions=('samples','latitude','longitude'))
    lon0[:], lat0[:], data[:] = lon, lat, smap
    f.close()


    path = configs.outputs_path+'forecast/'+configs.model_name+'/'
    os.system('mv {} {}'.format(configs.model_name+'_0p1_f16.nc', path))
    os.system('mv {} {}'.format('SMAPL4_0p1.nc', path))

if __name__ == '__main__':
    configs = parse_args()
    postprocess(configs)


