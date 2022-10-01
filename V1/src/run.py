import json
import logging
import os

import time
import numpy as np
import wandb
from config import parse_args
from data.data_generator import DataGenerator, DataLoader
from trainer import predict, train
#from ray import tune
#from ray.tune.schedulers import AsyncHyperBandScheduler


def main():
    """Main process for no backbone model."""

    #TODO:@lilu: Maybe need a params to control wandb when run on TH-2.
    # init parameters through wandb.
    configs = parse_args()
    id = configs.id
    default = dict(learning_rate=configs.learning_rate,
                   ROOT=configs.ROOT, 
                   n_filters_factor=configs.n_filters_factor,
                   kernel_size=configs.kernel_size,
                   batch_size=configs.batch_size,
                   epochs=configs.epochs,
                   shuffle_times=configs.shuffle_times,
                   padding=configs.padding,
                   kernel_initializer=configs.kernel_initializer,
                   len_out=configs.len_out,
                   convlstm_inputs_shape=configs.convlstm_inputs_shape,
                   cnn_inputs_shape=configs.cnn_inputs_shape,
                   model_name=configs.model_name,
                   validation_split=configs.validation_split,
                   saved_model_path=configs.saved_model_path,
                   dropout_rate=configs.dropout_rate,
                   stats_hs=configs.stats_hs,
                   id=configs.id)
    wandb.init(config=default, allow_val_change=True)

    
    # Data generator
    # NOTE: This module generate train/valid/test data
    #       for SMAP l3/l4 data with shape as
    #       (samples, height, width, features).
    #       We do not make inputs at once because the 
    #       process of reading inputs is complete.
    #if not os.path.exists('/work/lilu/inputs/SMAP_L4/{}'.format(id)):
    #    os.mkdir('/work/lilu/inputs/SMAP_L4/{}'.format(id))
    """
    if not os.path.exists('/work/lilu/inputs/SMAP_L4/{}/X_train_{}.npy'.format(id, id)):

        # make train, validate and test data
        DataGenerator(ID=id,
                      raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
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
                      end_date='2017-12-31',
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
                      save=True)()

        DataGenerator(ID=id,
                      raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                      auxiliary_data_path='/hard/lilu/SMAP_L4/test/',
                      save_x_path='/hard/lilu/SMAP_L4/test/forcing/',
                      save_y_path='/hard/lilu/SMAP_L4/test/SSM/',
                      save_px_path='/hard/lilu/SMAP_L4/test/px/',
                      save_py_path='/hard/lilu/SMAP_L4/test/py/',
                      lat_lower=14.7,
                      lat_upper=53.5,
                      lon_left=72.3,
                      lon_right=135,
                      begin_date='2018-01-01',
                      end_date='2018-12-31',
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
                      mode='valid',
                      save=True)()

        
        DataGenerator(ID=id,
                      raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
                      auxiliary_data_path='/hard/lilu/SMAP_L4/test/',
                      save_x_path='/hard/lilu/SMAP_L4/test/forcing/',
                      save_y_path='/hard/lilu/SMAP_L4/test/SSM/',
                      save_px_path='/hard/lilu/SMAP_L4/test/px/',
                      save_py_path='/hard/lilu/SMAP_L4/test/py/',
                      lat_lower=14.7,
                      lat_upper=53.5,
                      lon_left=72.3,
                      lon_right=135,
                      begin_date='2019-01-31',
                      end_date='2021-10-28',
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
                      save=True)()
        """

    # load land mask of sub-task
    with open(configs.ROOT + '/CLFS/auxiliary.json') as f:
        aux = json.load(f)
        mask = np.squeeze(np.array(aux['mask']))
        lat_id_low, lon_id_left = aux['lat_low'][id - 1], aux['lon_left'][id - 1]
        land_mask = mask[lat_id_low:lat_id_low + 112, lon_id_left:lon_id_left + 112]


    # load SMAP L4 and make inputs
    x_train = np.load(configs.ROOT + configs.inputs_path + '/{}/X_train_{}.npy'.format(id, id))
    x_test = np.load(configs.ROOT + configs.inputs_path + '/{}/X_valid_{}.npy'.format(id, id))
    y_train = np.load(configs.ROOT + configs.inputs_path + '/{}/y_train_{}.npy'.format(id, id))
    y_test = np.load(configs.ROOT + configs.inputs_path + '/{}/y_valid_{}.npy'.format(id, id))


    # Optional: load GFS input
    if configs.use_GFS:
        gfs_train = np.load(configs.ROOT + '/CLFS/inputs/GFS_0p1_f16_swvl1_interp_2015-2017.npy')
        gfs_id_train = gfs_train[:, :3, lat_id_low:lat_id_low + 112, lon_id_left:lon_id_left + 112]
        gfs_test = np.load(configs.ROOT + '/CLFS/inputs/GFS_0p1_f16_swvl1_interp_2018.npy')
        gfs_id_test = gfs_test[:, :3, lat_id_low:lat_id_low + 112, lon_id_left:lon_id_left + 112]

        #x_train = np.concatenate([x_train, gfs_id_train], axis=1)
        #x_test = np.concatenate([x_test, gfs_id_test], axis=1)
    

    # generate input-output for DL
    data_manager = DataLoader(len_input=configs.len_input, 
                              len_output=configs.len_out, 
                              window_size=configs.window_size, 
                              use_lag_y=configs.use_lag_y)
    #TODO: Add fit teaching as a selection params.
    #x_train, y_train = data_manager.fit_teacher_forcing(y_train)
    #x_test, y_test = data_manager.fit_teacher_forcing(y_test)
    #TODO: Add multimodal
    #x_train_l3 = np.load('/hard/lilu/inputs/SMAP_E_L3/X_train_l3.npy')[:, :, #lon_id_left:lon_id_left + 112, lat_id_low:lat_id_low + 112]
    #x_valid_l3 = np.load('/hard/lilu/inputs/SMAP_E_L3/X_valid_l3.npy')[:, :, #lon_id_left:lon_id_left + 112, lat_id_low:lat_id_low + 112]
    #x_test_l3 = np.load('/hard/lilu/inputs/SMAP_E_L3/X_test_l3.npy')[:, :, lon_id_left:lon_id_left + 112, lat_id_low:lat_id_low + 112]
    X, y = data_manager([x_train, x_test], [y_train, y_test])
    X_tf, _ = data_manager([gfs_id_train, gfs_id_test], [y_train, y_test])

    #TODO: Add this part into data generator.
    if wandb.config.model_name == 'unet':
        X = [X[0][:, -1], X[1][:, -1]]
    elif wandb.config.model_name in ['persistence', 'convlstm']:
        pass
    elif wandb.cofig.model_name == 'convlstm_tf':
        X = [[X[0], X_tf[0]], [X[1], X_tf[1]]]


    # train & inference DL
    train(X[0], y[0], wandb.config, land_mask)
    y_predict = predict(X[1], wandb.config)

    # save
    np.save(configs.model_name + '_0p1_f7_{id:02}.npy'.format(id=id), y_predict)
    np.save('smapl4_0p1_f7_{id:02}.npy'.format(id=id), y_test)
    

if __name__ == '__main__':
    a = time.time()
    main()
    b = time.time()

    logging.basicConfig(level=logging.INFO, filename='time.log', datefmt='%Y/%m/%d %H:%M:%S')
    logging.info(b-a)
    
