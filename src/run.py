from distutils.command.config import config
import json
import logging
import os
import time
import numpy as np
import wandb
import xarray as xr

from config import parse_args
from data.data_generator import DataGenerator
from data.data_loader import DataLoader
from trainer import predict, train


def main(configs):
    """Main process for no backbone model."""


    # load land mask of sub-task
    print('.............................................................')
    print("Please ensure that raw data is preprocessed"+
          "and saved in {}, Let's Go!".format(configs.inputs_path))
    print('.............................................................')
    print('[HybridHydro] Loading land mask')
    id = configs.id
    with open(configs.inputs_path + 'auxiliary.json') as f:
        aux = json.load(f)
        mask = np.squeeze(np.array(aux['mask']))
        lat_id_low, lon_id_left = aux['lat_low'][id - 1], aux['lon_left'][id - 1]
        land_mask = mask[lat_id_low:lat_id_low + 112, lon_id_left:lon_id_left + 112]
    print('.............................................................')


    # load SMAP L4 and make inputs
    print('[HybridHydro] Loading train/test raw data for sub-region {}'.format(id))
    x_train = np.load(configs.inputs_path + '{}/X_train_{}.npy'.format(id, id))
    x_test = np.load(configs.inputs_path + '{}/X_valid_{}.npy'.format(id, id))
    y_train = np.load(configs.inputs_path + '{}/y_train_{}.npy'.format(id, id))
    y_test = np.load(configs.inputs_path + '{}/y_valid_{}.npy'.format(id, id))
    num_train_sample = x_train.shape[0]
    num_test_sample = x_test.shape[0]
    print('.............................................................')


    # load static vars 
    print('[HybridHydro] Loading ancillary data')
    if configs.use_ancillary:
        with xr.open_dataset(configs.inputs_path+'LC_CN_EASE_9km.nc') as f:
            lc = np.array(f.land_cover)[np.newaxis,np.newaxis,
                lat_id_low:lat_id_low + 112, lon_id_left:lon_id_left + 112]
        with xr.open_dataset(configs.inputs_path+'DEM_CN_EASE_9km.nc') as f:
            dem = np.array(f.dem)[np.newaxis,np.newaxis,
                lat_id_low:lat_id_low + 112, lon_id_left:lon_id_left + 112]
        ancil = np.concatenate([lc,dem],axis=1)
        ancil_train = np.tile(ancil, (num_train_sample,1,1,1))
        ancil_test = np.tile(ancil,(num_test_sample,1,1,1))
        x_train = np.concatenate([x_train,ancil_train],axis=1)
        x_test = np.concatenate([x_test,ancil_test],axis=1)
    print('.............................................................')


    # load gfs
    print('[HybridHydro] Loading GFS data')
    gfs_train = np.load(configs.inputs_path+'GFS_0p1_f16_swvl1_interp_2015-2017.npy')
    gfs_id_train = gfs_train[:, :3, lat_id_low:lat_id_low + 112, lon_id_left:lon_id_left + 112]
    gfs_test = np.load(configs.inputs_path+'GFS_0p1_f16_swvl1_interp_2018.npy')
    gfs_id_test = gfs_test[:, :3, lat_id_low:lat_id_low + 112, lon_id_left:lon_id_left + 112]
    del gfs_train, gfs_test
    print('.............................................................')


    # generate input/output for DL models
    print('[HybridHydro] Making input data for {} model'.format(configs.model_name))
    data_manager = DataLoader(configs)

    if configs.model_name == 'convlstm':
        X, y = data_manager([x_train, x_test], [y_train, y_test])
    elif configs.model_name == 'convlstm_drive':
        x_train = np.concatenate([x_train, gfs_id_train], axis=1)
        x_test = np.concatenate([x_test, gfs_id_test], axis=1)
        X, y = data_manager([x_train, x_test], [y_train, y_test])
    elif configs.model_name in ['convlstm_condition', 'convlstm_att_condition']:
        # construct GFS inputs only for decoder 
        X, y = data_manager([x_train, x_test], [y_train, y_test])
        _, X_tf = data_manager([x_train, x_test], [gfs_id_train, gfs_id_test]) 
        X = [[X[0], X_tf[0]], [X[1], X_tf[1]]]

    print('[HybridHydro] Training samples shape: {}'.format(y[0].shape[0]))
    print('[HybridHydro] Testing samples shape: {}'.format(y[1].shape[0]))
    print('.............................................................')

    print(X[0].shape, X[1].shape)
    print(y[0].shape, y[1].shape)

    # train & inference DL
    print('[HybridHydro] Training {} model'.format(configs.model_name))
    loss = train(X[0], y[0], configs, land_mask)
    print('[HybridHydro] Inference {} model'.format(configs.model_name))
    y_predict = predict(X[1], configs)
    print('.............................................................')

    # save
    print('[HybridHydro] Saving')
    path = configs.outputs_path+'forecast/'+configs.model_name+'/'+str(configs.id)+'/'
    if not os.path.exists(configs.outputs_path+'forecast/'+configs.model_name):
        os.mkdir(configs.outputs_path+'forecast/'+configs.model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(configs.model_name+'_0p1_f16_{id:02}.npy'.format(id=id), y_predict)
    np.save('SMAPL4_0p1_f16_{id:02}.npy'.format(id=id), y_test)
    np.save('loss_{id:02}_{model}.npy'.format(id=id,model=configs.model_name), loss)
    os.system('mv {} {}'.format(configs.model_name+'_0p1_f16_{id:02}.npy'.format(id=id), path))
    os.system('mv {} {}'.format('SMAPL4_0p1_f16_{id:02}.npy'.format(id=id), path))
    os.system('mv {} {}'.format('loss_{id:02}_{model}.npy'.format(id=id,model=configs.model_name), path))
    print('[HybridHydro] Finished! You could check the saved'+
          'models and predictions in path: {}'.format(configs.outputs_path))


if __name__ == '__main__':
    a = time.time()
    configs = parse_args()
    main(configs)
    b = time.time()

    logging.basicConfig(level=logging.INFO, filename='time.log', datefmt='%Y/%m/%d %H:%M:%S')
    logging.info(b-a)
    
