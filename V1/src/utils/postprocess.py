
import glob
import numpy as np
import netCDF4 as nc
from data.preprocess import read_daily_CLDAS_forcing


def postprocess(predict_path, out_path, 
                lat_lower, lat_upper, 
                lon_left, lon_right, window_size, date):

    # 按照顺序读入输出的task的数据集（task, lat, lon）
    # ------------------------------------------
    l = glob.glob(predict_path + 'output*nc', recursive=True)

    # handle for nc file
    f = nc.Dataset(l[0], 'r')
    Nlat,Nlon = f['y_predict'][:].shape

    # 按照输入的比例，拆分成（Nlat*lat，Nlon*lon）
    # --------------------------------------
    Mlat = int((lat_upper-lat_lower)/window_size) # 11
    Mlon = int((lon_right-lon_left)/window_size) # 13

    data = np.full((Mlat*Nlat, Mlon*Nlon), np.nan)

    i = 0
    for task in np.arange(len(l)):
        
        # handle for nc file
        f = nc.Dataset(predict_path+'output_task_'+str(task)+'.nc', 'r')

        if (task % Mlon == 0) and (task >= Mlon):
            i+=1

        # get location
        j, k = task % Mlon, Mlat-i-1

        # concat
        data[k*Nlat:(k+1)*Nlat, j*Nlon:(j+1)*Nlon] = np.flip(f['y_predict'][:],0)


    # mask 
    # ----

    

    # boundary fuzzification
    # ----------------------
    for i in np.arange(Mlon):
        for j in np.arange(Mlat):

            boundary_left = i*Nlon-1
            boundary_right = i*Nlon

            data[:, boundary_left] = np.nanmean(data[:, boundary_left:boundary_right+1], axis=-1)
            data[:, boundary_right] = np.nanmean(data[:, boundary_left:boundary_right+1], axis=-1)

            print(np.nanmean(data[:, boundary_left:boundary_right+1], axis=-1).shape)

            boundary_upper = j*Nlat
            boundary_lower = j*Nlat-1

            data[:, boundary_upper] = np.nanmean(data[:, boundary_lower:boundary_upper+1], axis=-1)
            data[:, boundary_lower] = np.nanmean(data[:, boundary_lower:boundary_upper+1], axis=-1)


    # 存储成nc文件
    # ----------
    # save to nc files
    filename = 'FORE_SSM_{year}{month:02}{day:02}.nc'.\
        format(year=date[0].year,
                month=date[0].month,
                day=date[0].day)

    f = nc.Dataset(out_path + filename, 'w', format='NETCDF4')

    f.createDimension('longitude', size=data.shape[1])
    f.createDimension('latitude', size=data.shape[0])

    lon = f.createVariable('longitude', 'f4', dimensions='longitude')
    lat = f.createVariable('latitude', 'f4', dimensions='latitude')
    ssm = f.createVariable('ssm', 'f4', \
        dimensions=('latitude', 'longitude'))


    _, lat_, lon_ = read_daily_CLDAS_forcing('/hard/lilu/CLDAS_FORCING/CLDAS_FORCING_DD/', '2015-05-31', '2015-06-01')
    lat[:] = np.flip(lat_,0)
    lon[:] = lon_
    ssm[:] = data

    f.close()

if __name__ == '__main__':
    postprocess(
          predict_path='/hard/lilu/4_HRSMPP/predict/', 
          out_path='/hard/lilu/4_HRSMPP/', 
                lat_lower=22, lat_upper=33, 
                lon_left=110, lon_right=123, window_size=1)
