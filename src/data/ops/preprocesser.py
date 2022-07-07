import glob
import json
import os
import sys

from numpy.lib.polynomial import _raise_power

sys.path.append('../../../hrsepp/')
import netCDF4 as nc
import numpy as np
from data.ops.init import AuxManager
from data.ops.readers import RawSMAPReader
from data.ops.saver import nc_saver
from data.ops.time import TimeManager
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


class RawSMAPPreprocesser():
    def __init__(self,
                 raw_data_path,
                 aux,
                 save_path,
                 begin_date,
                 end_date,
                 var_name,
                 var_list,
                 lat_lower=-90,
                 lat_upper=90,
                 lon_left=-180,
                 lon_right=180,
                 save=True) -> None:

        self.raw_data_path = raw_data_path
        self.aux = aux
        self.begin_date = begin_date
        self.end_date = end_date
        self.save = save
        self.var_name = var_name
        self.var_list = var_list
        self.save_path = save_path

        self.raw_smap_reader = RawSMAPReader(lat_lower, lat_upper, lon_left,
                                             lon_right)

    def __call__(self):
        # get dates array according to begin/end dates
        dates = TimeManager().get_date_array(self.begin_date, self.end_date)

        # read and save file (after integrate spatial/temporal dimension)
        data = np.full((len(dates), len(
            self.var_list), self.aux['Nlat'], self.aux['Nlon']), np.nan)

        for t, date in enumerate(dates):

            # save to nc files
            filename = 'SMAP_L4_{var_name}_{year}{month:02}{day:02}.nc'.format(
                var_name=self.var_name,
                year=date.year,
                month=date.month,
                day=date.day)

            # judge already exist file
            if not os.path.exists(self.save_path + filename):

                # folder name
                foldername = '{year}.{month:02}.{day:02}/'.format(
                    year=date.year, month=date.month, day=date.day)
                # file list in each folder
                l = glob.glob(self.raw_data_path + foldername + 'SMAP*.h5',
                              recursive=True)

                assert len(l) == 8, '[HRSEPP][error]lack data of {}'.format(
                    date)


                # 

                # integrate from 3-hour to daily #NOTE:Only suite for SMAP
                tmp = np.full((len(l), len(
                    self.var_list), self.aux['Nlat'], self.aux['Nlon']),
                              np.nan)

                for i, one_file_path in enumerate(l):

                    tmp[i, :, :, :], _, _ = self.raw_smap_reader(
                        one_file_path, self.var_list)

                data[t] = np.nanmean(tmp, axis=0)

                nc_saver(save_path=self.save_path,
                         X=np.nanmean(tmp, axis=0),
                         var_name=self.var_name,
                         date=date,
                         lat_2d=self.aux['lat_2d'],
                         lon_2d=self.aux['lon_2d'])

        return data


class XPreprocesser():
    def __init__(self,
                 X,
                 ID,
                 save_path,
                 auxiliary_path,
                 begin_date,
                 end_date,
                 mode='train',
                 save=True,
                 var_name='SSM'):

        # get shape
        self.Nt, self.Nf, self.Nlat, self.Nlon = X.shape
        self.begin_date = begin_date
        self.end_date = end_date
        self.auxiliary_path = auxiliary_path
        with open(auxiliary_path + 'auxiliary.json', 'r') as f:
            self.aux = json.load(f)
        self.save_path = save_path
        self.mode = mode
        self.save = save
        self.X = X
        self.var_name = var_name
        self.ID = ID
        self.lat_id_low = self.aux['lat_low'][ID - 1]
        self.lon_id_left = self.aux['lon_left'][ID - 1]

        print(self.lat_id_low)
        print(self.lon_id_left)
        print(np.array(self.aux['lon_2d']).shape)

        self.lon_id = np.array(
            self.aux['lon_2d'])[self.lat_id_low:self.lat_id_low + 112,
            self.lon_id_left:self.lon_id_left + 112,
                                ]
        self.lat_id = np.array(
            self.aux['lat_2d'])[self.lat_id_low:self.lat_id_low + 112, 
            self.lon_id_left:self.lon_id_left + 112]

        print(self.lon_id.shape)
        print(self.lat_id.shape)

    def __call__(self):

        if self.mode == 'train':
            print('okkkkkkkk')
            #TODO:Add ID for min, max scale in aux dict.
            X, min_scale, max_scale = self._train_preprocesser(self.X)
            print(X.shape)
            AuxManager().update(self.auxiliary_path, 'min_scale',
                                min_scale.tolist())
            AuxManager().update(self.auxiliary_path, 'max_scale',
                                max_scale.tolist())

        else:
            X = self._test_preprocesser(self.X)

        if self.save:
            # get dates array according to begin/end dates
            dates = TimeManager().get_date_array(self.begin_date,
                                                 self.end_date)

            for i, date in enumerate(dates):
                print(i, date)
                print(self.save_path)
                print(self.lon_id.shape)
                print(self.lat_id.shape)
                print(X[i].shape)

                nc_saver(self.save_path,
                         'p_' + self.var_name + '_{}'.format(self.ID), date,
                         self.lon_id, self.lat_id, X[i])

        return X

    def _train_preprocesser(self, inputs):

        # interplot and scale for each feature on each grid
        #min_scale = np.full((self.Nf, self.Nlat, self.Nlon), np.nan)
        #max_scale = np.full((self.Nf, self.Nlat, self.Nlon), np.nan)
        """
        # interplot on time dimension.
        for i in range(self.Nlat):
            for j in range(self.Nlon):

                try:
                    # interplot along time dimension
                    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                    inputs[:, :, i, j] = imp.fit_transform(inputs[:, :, i, j])

                    # min max scaler
                    #FIXME:Change to MinMax for each channel.
                    
                    scaler = MinMaxScaler()
                    inputs[:, :, i, j] = scaler.fit_transform(inputs[:, :, i,
                                                                     j])
                    min_scale[:, i, j] = scaler.data_min_
                    max_scale[:, i, j] = scaler.data_max_
                    
                except:  # all missing data along time dimension
                    pass
        """
        min_scale = np.full((self.Nf, ), np.nan)
        max_scale = np.full((self.Nf, ), np.nan)

        for i in range(self.Nf):
            max_scale[i] = np.nanmax(inputs[:, i])
            min_scale[i] = np.nanmin(inputs[:, i])
            #print(i)
            #print(max_scale[i], min_scale[i])
            inputs[:, i] = (inputs[:, i] - min_scale[i]) / (max_scale[i] -
                                                            min_scale[i])

        # interplot on spatial dimension, in order to fill gaps of images.
        for m in range(self.Nt):
            for n in range(self.Nf):

                # interplot
                tmp = inputs[m, n, :, :]
                tmp[np.isnan(tmp)] = np.nanmean(tmp)
                inputs[m, n, :, :] = tmp

        return inputs, min_scale, max_scale

    def _test_preprocesser(self, inputs):

        try:
            min_scale = np.array(self.aux['min_scale'])
            max_scale = np.array(self.aux['max_scale'])
            print(inputs.shape)
            """
            # preprocess according normalized parameters
            for i in range(inputs.shape[-2]):
                for j in range(inputs.shape[-1]):
                    try:
                        # interplot
                        imp = SimpleImputer(missing_values=np.nan,
                                            strategy='mean')
                        print(inputs.shape)
                        inputs[:, :, i, j] = imp.fit_transform(inputs[:, :, i,
                                                                      j])
                    except:
                        pass
            """
            # min max scaler
            for m in np.arange(self.Nf):
                inputs[:, m] = (inputs[:, m] - min_scale[m]) / (max_scale[m] -
                                                                min_scale[m])

        except:
            raise IOError('preprocess train data before preprocess test data!')

        # interplot on spatial dimension, in order to fill gaps of images.
        for m in range(self.Nt):
            for n in range(self.Nf):

                # interplot
                tmp = inputs[m, n, :, :]
                tmp[np.isnan(tmp)] = np.nanmean(tmp)
                inputs[m, n, :, :] = tmp

        return inputs


class yPreprocesser():
    def __init__(self,
                 y,
                 ID,
                 save_path,
                 auxiliary_path,
                 begin_date,
                 end_date,
                 mode='train',
                 save=True,
                 var_name='SSM'):

        # get shape
        self.Nt, self.Nf, self.Nlat, self.Nlon = y.shape
        self.begin_date = begin_date
        self.end_date = end_date
        self.auxiliary_path = auxiliary_path
        with open(auxiliary_path + 'auxiliary.json', 'r') as f:
            self.aux = json.load(f)
        self.save_path = save_path
        self.mode = mode
        self.save = save
        self.y = y
        self.var_name = var_name
        self.ID = ID

        self.lat_id_low = self.aux['lat_low'][ID - 1]
        self.lon_id_left = self.aux['lon_left'][ID - 1]

        self.lon_id = np.array(
            self.aux['lon_2d'])[self.lat_id_low:self.lat_id_low + 112,
            self.lon_id_left:self.lon_id_left + 112,
                                ]
        self.lat_id = np.array(
            self.aux['lat_2d'])[self.lat_id_low:self.lat_id_low + 112, 
            self.lon_id_left:self.lon_id_left + 112]

    def __call__(self):
        """
        # interplot on time dimension.
        for i in range(self.y.shape[-2]):
            for j in range(self.y.shape[-1]):

                try:
                    # interplot
                    imp = SimpleImputer(missing_values=np.nan,
                                        strategy='constant',
                                        fill_value=0)
                    self.y[:, :, i, j] = imp.fit_transform(self.y[:, :, i, j])
                except:  # all missing data along time dimension
                    pass
        """

        # interplot on spatial dimension, in order to fill gaps of images.
        for m in range(self.Nt):
            for n in range(self.Nf):

                # interplot
                tmp = self.y[m, n, :, :]
                tmp[np.isnan(tmp)] = np.nanmean(tmp)
                self.y[m, n, :, :] = tmp

        # get dates array according to begin/end dates
        dates = TimeManager().get_date_array(self.begin_date, self.end_date)

        for i, date in enumerate(dates):
            nc_saver(self.save_path,
                     'p_' + self.var_name + '_{}'.format(self.ID), date,
                     self.lon_id, self.lat_id, self.y[i])

        return self.y


if __name__ == '__main__':

    AuxManager().init(raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                      auxiliary_data_path='/hard/lilu/SMAP_L4/test/',
                      lat_lower=10,
                      lat_upper=20,
                      lon_left=30,
                      lon_right=40)

    with open('/hard/lilu/SMAP_L4/test/auxiliary.json') as f:
        aux = json.load(f)

    print(aux.keys())

    data = RawSMAPPreprocesser(raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                               aux=aux,
                               save_path='/hard/lilu/SMAP_L4/test/SSM/',
                               begin_date='2015-05-31',
                               end_date='2015-06-31',
                               var_name='SSM',
                               var_list=['sm_surface'],
                               lat_lower=10,
                               lat_upper=20,
                               lon_left=30,
                               lon_right=40,
                               save=True)()

    print(data.shape)

    data = RawSMAPPreprocesser(raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                               aux=aux,
                               save_path='/hard/lilu/SMAP_L4/test/forcing/',
                               begin_date='2015-05-31',
                               end_date='2015-06-31',
                               var_name='forcing',
                               var_list=[
                                   'precipitation_total_surface_flux',
                                   'radiation_longwave_absorbed_flux',
                                   'radiation_shortwave_downward_flux',
                                   'specific_humidity_lowatmmodlay',
                                   'surface_pressure', 'surface_temp',
                                   'windspeed_lowatmmodlay'
                               ],
                               lat_lower=10,
                               lat_upper=20,
                               lon_left=30,
                               lon_right=40,
                               save=True)()

    print(data.shape)
