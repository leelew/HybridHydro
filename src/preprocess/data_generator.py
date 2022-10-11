import sys; sys.path.append('../../HybridHydro/')
import json
import numpy as np
import os

from init import AuxManager
from preprocesser import (RawSMAPPreprocesser, XPreprocesser,yPreprocesser)
from readers import NCReader
from time import TimeManager


# generate train/valid/test data for each subregion
class DataGenerator():
    """Get sub-regions train/test datasets."""
    def __init__(self, configs, id, mode='train',save=True):
        self.raw_data_path = configs.raw_data_path
        self.auxiliary_data_path = configs.inputs_path
        self.save_x_path = configs.raw_data_path + 'forcing/'+mode + '/'
        self.save_y_path = configs.raw_data_path +'SSM/' + mode + '/'
        self.save_px_path = configs.raw_data_path + 'px/' + mode + '/'
        self.save_py_path =configs.raw_data_path + 'py' + mode + '/'
        self.lat_lower = configs.lat_lower
        self.lat_upper = configs.lat_upper
        self.lon_left = configs.lon_left
        self.lon_right = configs.lon_right
        self.ID = id
        if mode == 'train':
            self.begin_date = configs.begin_train_date
            self.end_date = configs.end_train_date
        elif mode == 'test':
            self.begin_date = configs.begin_test_date
            self.end_date = configs.end_test_date
        self.x_var_name = 'forcing'
        self.x_var_list = ['precipitation_total_surface_flux',
                     'radiation_longwave_absorbed_flux',
                     'radiation_shortwave_downward_flux',
                     'specific_humidity_lowatmmodlay', 'surface_pressure',
                     'surface_temp', 'windspeed_lowatmmodlay']
        self.y_var_name = 'SSM'
        self.y_var_list = ['sm_surface']
        self.mode = mode
        self.save = save


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

        # read & save soil moisture from SMAP
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

        # read & save forcing from SMAP
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

        # read forcing data
        X = NCReader(path=self.save_x_path,
                     aux=aux,
                     var_list=self.x_var_list,
                     var_name=self.x_var_name,
                     begin_date=self.begin_date,
                     end_date=self.end_date)()[:, :,
                                               lat_id_low:lat_id_low + 112,
                                               lon_id_left:lon_id_left + 112]
        
        # preprocess & save forcing
        X = XPreprocesser(X,
                          ID=self.ID,
                          save_path=self.save_px_path,
                          auxiliary_path=self.auxiliary_data_path,
                          begin_date=self.begin_date,
                          end_date=self.end_date,
                          mode=self.mode,
                          save=self.save,
                          var_name=self.x_var_name)()
        np.save('X_{}_{}.npy'.format(self.mode, self.ID), X)

        # read soil moisture 
        y = NCReader(path=self.save_y_path,
                     aux=aux,
                     var_list=self.y_var_list,
                     var_name=self.y_var_name,
                     begin_date=self.begin_date,
                     end_date=self.end_date)()[:, :,
                                               lat_id_low:lat_id_low + 112,
                                               lon_id_left:lon_id_left + 112]
        
        # preprocess soil moisture
        y = yPreprocesser(y,
                          ID=self.ID,
                          save_path=self.save_py_path,
                          auxiliary_path=self.auxiliary_data_path,
                          begin_date=self.begin_date,
                          end_date=self.end_date,
                          mode=self.mode,
                          save=self.save,
                          var_name=self.y_var_name)()
        # save soil moisture
        np.save('y_{}_{}.npy'.format(self.mode, self.ID), y)

        return X, y


if __name__ == '__main__':

    from src.configs import parse_args
    configs = parse_args()

    # Data generator
    # NOTE: This module generate train/valid/test data
    #       for SMAP l3/l4 data with shape as
    #       (samples, height, width, features).
    #       We do not make inputs at once because the 
    #       process of reading inputs is complete.
    if not os.path.exists(configs.inputs_path+'{}/X_train_{}.npy'.format(id, id)):
        # make train, validate and test data
        DataGenerator(configs, id, mode='train')()        
        DataGenerator(configs, id, mode='test')()
        
