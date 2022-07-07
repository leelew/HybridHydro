import sys

sys.path.append('../../hrsepp/')

import json

import numpy as np

from data.ops.init import AuxManager
from data.ops.make_X_y import make_Xy
from data.ops.preprocesser import (RawSMAPPreprocesser, XPreprocesser,
                                   yPreprocesser)
from data.ops.readers import NCReader
from data.ops.time import TimeManager


# generate train/valid/test data for each subregion
class DataGenerator():
    """generate comtemporary data of X, y. 
       shape as [(t, f, lat, lon), (t, 1, lat, lon)]
    """
    def __init__(self,
                 ID,
                 raw_data_path,
                 auxiliary_data_path,
                 save_x_path,
                 save_y_path,
                 save_px_path,
                 save_py_path,
                 lat_lower,
                 lat_upper,
                 lon_left,
                 lon_right,
                 begin_date,
                 end_date,
                 x_var_name='forcing',
                 x_var_list=[
                     'precipitation_total_surface_flux',
                     'radiation_longwave_absorbed_flux',
                     'radiation_shortwave_downward_flux',
                     'specific_humidity_lowatmmodlay', 'surface_pressure',
                     'surface_temp', 'windspeed_lowatmmodlay'
                 ],
                 y_var_name='SSM',
                 y_var_list=['sm_surface'],
                 mode='train',
                 save=True,
                 use_lag_y=True):
        self.raw_data_path = raw_data_path
        self.auxiliary_data_path = auxiliary_data_path
        self.save_x_path = save_x_path + mode + '/'
        self.save_y_path = save_y_path + mode + '/'
        #FIXME:auto make dir if no corresponding folder
        self.save_px_path = save_px_path + mode + '/'
        self.save_py_path = save_py_path + mode + '/'

        self.lat_lower = lat_lower
        self.lat_upper = lat_upper
        self.lon_left = lon_left
        self.lon_right = lon_right
        self.ID = ID
        self.begin_date = begin_date
        self.end_date = end_date

        self.x_var_name = x_var_name
        self.x_var_list = x_var_list

        self.y_var_name = y_var_name
        self.y_var_list = y_var_list
        self.mode = mode
        self.save = save

        self.use_lag_y = use_lag_y

    def __call__(self):
        if self.mode == 'train':
            # init auxiliary data
            AuxManager().init(raw_data_path=self.raw_data_path,
                              auxiliary_data_path=self.auxiliary_data_path,
                              lat_lower=self.lat_lower,
                              lat_upper=self.lat_upper,
                              lon_left=self.lon_left,
                              lon_right=self.lon_right)

        # load auxiliary data
        with open(self.auxiliary_data_path + 'auxiliary.json') as f:
            aux = json.load(f)

        # read soil moisture from SMAP
        RawSMAPPreprocesser(raw_data_path=self.raw_data_path,
                            aux=aux,
                            save_path=self.save_y_path,
                            begin_date=self.begin_date,
                            end_date=self.end_date,
                            var_name=self.y_var_name,
                            var_list=self.y_var_list,
                            lat_lower=self.lat_lower,
                            lat_upper=self.lat_upper,
                            lon_left=self.lon_left,
                            lon_right=self.lon_right,
                            save=self.save)()

        # read forcing from SMAP
        RawSMAPPreprocesser(raw_data_path=self.raw_data_path,
                            aux=aux,
                            save_path=self.save_x_path,
                            begin_date=self.begin_date,
                            end_date=self.end_date,
                            var_name=self.x_var_name,
                            var_list=self.x_var_list,
                            lat_lower=self.lat_lower,
                            lat_upper=self.lat_upper,
                            lon_left=self.lon_left,
                            lon_right=self.lon_right,
                            save=self.save)()

        lat_id_low = aux['lat_low'][self.ID - 1]
        lon_id_left = aux['lon_left'][self.ID - 1]

        print('ok')
        print(np.array(aux['lat_2d']).shape)
        print(np.array(aux['lon_2d']).shape)

        X = NCReader(path=self.save_x_path,
                     aux=aux,
                     var_list=self.x_var_list,
                     var_name=self.x_var_name,
                     begin_date=self.begin_date,
                     end_date=self.end_date)()[:, :,
                                               lat_id_low:lat_id_low + 112,
                                               lon_id_left:lon_id_left + 112]

        print(X.shape)
        print('ok')
        print(np.array(aux['lat_2d']).shape)
        print(np.array(aux['lon_2d']).shape)
        # preprocess forcing
        X = XPreprocesser(X,
                          ID=self.ID,
                          save_path=self.save_px_path,
                          auxiliary_path=self.auxiliary_data_path,
                          begin_date=self.begin_date,
                          end_date=self.end_date,
                          mode=self.mode,
                          save=self.save,
                          var_name=self.x_var_name)()
        print('1ok')
        print(X.shape)

        np.save('X_{}_{}.npy'.format(self.mode, self.ID), X)

        y = NCReader(path=self.save_y_path,
                     aux=aux,
                     var_list=self.y_var_list,
                     var_name=self.y_var_name,
                     begin_date=self.begin_date,
                     end_date=self.end_date)()[:, :,
                                               lat_id_low:lat_id_low + 112,
                                               lon_id_left:lon_id_left + 112]

        y = yPreprocesser(y,
                          ID=self.ID,
                          save_path=self.save_py_path,
                          auxiliary_path=self.auxiliary_data_path,
                          begin_date=self.begin_date,
                          end_date=self.end_date,
                          mode=self.mode,
                          save=self.save,
                          var_name=self.y_var_name)()

        # get jd index
        jd_idx = TimeManager().jd(self.begin_date, self.end_date)

        #
        if self.mode == 'train':
            a = np.full((366, 1, 112, 112, 5), np.nan)
            for i in range(1, 367):  # for a year
                idx = np.where(jd_idx == i)[0]
                print(idx)
                tmp = y[idx]

                tmp_vec = np.stack([
                    np.nanmean(tmp, axis=0),
                    np.nanstd(tmp, axis=0),
                    np.nanmax(tmp, axis=0),
                    np.nanmin(tmp, axis=0),
                    np.nanmedian(tmp, axis=0)
                ],
                                         axis=-1)

                a[i-1] = tmp_vec
            #np.save('/hard/lilu/inputs/SMAP_L4/z_{}.npy'.format(self.ID), a)
        #a = np.load('/hard/lilu/inputs/SMAP_L4/z_{}.npy'.format(self.ID))
        #m = a[jd_idx-1]
        #print(m.shape)
        #np.save('/hard/lilu/inputs/SMAP_L4/{}/z_{}_{}.npy'.format(self.ID, self.mode, self.ID), m)

        #
        np.save('y_{}_{}.npy'.format(self.mode, self.ID), y)

        return X, y



class DataLoader():
    #[(sample, in_len, lat, lon, f), (sample, out_len, lat, lon, 1)]

    def __init__(self,
                 len_input,
                 len_output,
                 window_size,
                 use_lag_y=True,
                 mode='train'):
        self.len_input = len_input
        self.len_output = len_output
        self.window_size = window_size
        self.use_lag_y = use_lag_y
    
    def fit_teacher_forcing(self, y):
        """generate inputs for teacher forcing training.

        Args:
            y ([type]): shape as (S, 1, 112, 112)
        
        Outputs:
            X: shape as (S-14, 14, 112, 112, 1)
            y: shape as (S-14, 14, 112, 112, 1)
        """
        X, y = self.make_tf_Xy(y, len_input=7, len_output=7, window_size=6)
        return X, y


    def make_tf_Xy(
            self,
            inputs,
            len_input=7,
            len_output=7,
            window_size=0):

        Nt, _, Nlat, Nlon = inputs.shape
        """Generate inputs and outputs for LSTM."""
        # caculate the last time point to generate batch
        end_idx = inputs.shape[0] - len_input - len_output - window_size - 1
        print(end_idx)
        # generate index of batch start point in order
        batch_start_idx = range(end_idx)
        # get batch_size
        batch_size = len(batch_start_idx)
        print(batch_size)

        # generate inputs
        input_batch_idx = [(range(i, i + len_input)) for i in batch_start_idx]
        X = np.take(inputs, input_batch_idx, axis=0).reshape(batch_size, len_input, inputs.shape[-1], Nlat, Nlon)

        X = np.transpose(X, [0, 1, 3, 4, 2])
        print(X.shape)

        # generate outputs
        output_batch_idx = [(range(i + len_input + window_size,
                                i + len_input + window_size + len_output))
                            for i in batch_start_idx]
        y = np.take(inputs, output_batch_idx, axis=0). \
            reshape(batch_size,  len_output,  1, Nlat, Nlon)
        y = np.transpose(y, [0, 1, 3, 4, 2])
        print(y.shape)
        return X, y   


    def __call__(self, X, y):
        """[summary]

        Args:
            X ([type]): (samples, timestep, height, width, features)
            y ([type]): (samples, timestep, height, width, 1)
            z ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        # generate inputs
        for i in range(len(X)):
            _x, _y = make_Xy(X[i], y[i],
                             self.len_input, 
                             self.len_output, 
                             self.window_size,
                             self.use_lag_y)
            X[i], y[i] = _x, _y
            print('1')
            print(_x.shape)
            print(_y.shape)
            print('2')
        return X, y

if __name__ == '__main__':

    DataGenerator(raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                  auxiliary_data_path='/hard/lilu/SMAP_L4/test/',
                  save_x_path='/hard/lilu/SMAP_L4/test/forcing/',
                  save_y_path='/hard/lilu/SMAP_L4/test/SSM/',
                  save_px_path='/hard/lilu/SMAP_L4/test/px/',
                  save_py_path='/hard/lilu/SMAP_L4/test/py/',
                  lat_lower=14.7,
                  lat_upper=53.5,
                  lon_left=72.3,
                  lon_right=135,
                  begin_date='2015-05-31',
                  end_date='2020-05-31',
                  x_var_name='forcing',
                  x_var_list=[
                      'precipitation_total_surface_flux',
                      'radiation_longwave_absorbed_flux',
                      'radiation_shortwave_downward_flux',
                      'specific_humidity_lowatmmodlay', 'surface_pressure',
                      'surface_temp', 'windspeed_lowatmmodlay'
                  ],
                  y_var_name='SSM',
                  y_var_list=['sm_surface'],
                  mode='train',
                  save=True)(ID=2)

    DataGenerator(raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                  auxiliary_data_path='/hard/lilu/SMAP_L4/test/',
                  save_x_path='/hard/lilu/SMAP_L4/test/forcing/',
                  save_y_path='/hard/lilu/SMAP_L4/test/SSM/',
                  save_px_path='/hard/lilu/SMAP_L4/test/px/',
                  save_py_path='/hard/lilu/SMAP_L4/test/py/',
                  lat_lower=14.7,
                  lat_upper=53.5,
                  lon_left=72.3,
                  lon_right=135,
                  begin_date='2020-05-31',
                  end_date='2021-05-31',
                  x_var_name='forcing',
                  x_var_list=[
                      'precipitation_total_surface_flux',
                      'radiation_longwave_absorbed_flux',
                      'radiation_shortwave_downward_flux',
                      'specific_humidity_lowatmmodlay', 'surface_pressure',
                      'surface_temp', 'windspeed_lowatmmodlay'
                  ],
                  y_var_name='SSM',
                  y_var_list=['sm_surface'],
                  mode='test',
                  save=True)(ID=2)


# 获取3天前-13天前的起止日期。
#    begin_date = date - datetime.timedelta(days = 12)
#    end_date = date - datetime.timedelta(days = 3)  
