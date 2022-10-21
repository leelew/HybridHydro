import imp
import json
import os
import time
import numpy as np
import wandb
import xarray as xr

from config import parse_args
from trainer import predict, train, train_generator
from data_loader import DataLoader
from utils import init_fold


def main(configs):
    """Main process for no backbone model."""
    # init fold for work path
    init_fold(configs.work_path)

    # wandb init important params
    default = dict(id=configs.id,
                   # model
                   model_name=configs.model_name,
                   n_filters_factor=configs.n_filters_factor,
                   stats_hs=configs.stats_hs,
                   # train
                   batch_size=configs.batch_size,
                   epochs=configs.epochs)
    wandb.init("HybridHydro", config=default, allow_val_change=True)

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
    data_manager = DataLoader(configs)
    tmp_x = np.load(configs.inputs_path + '{}/X_train_{}.npy'.format(id, id))
    tmp_y = np.load(configs.inputs_path + '{}/y_train_{}.npy'.format(id, id))
    x_train, y_train = [], []
    for i in range(4):
        for j in range(4):
            X, y = data_manager([tmp_x[:,:,i*28:(i+1)*28, j*28:(j+1)*28]], [tmp_y[:,:,i*28:(i+1)*28, j*28:(j+1)*28]])
            x_train.append(X[0])
            y_train.append(y[0])
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    num_train_sample = x_train.shape[0]
    print('.............................................................')


    # load gfs
    print('[HybridHydro] Loading GFS data')
    if configs.model_name != 'convlstm':
        gfs_train = np.load(configs.inputs_path+'GFS_0p1_f3_swvl1_interp_2015-2017.npy')
        gfs_id_train = gfs_train[:, :3, lat_id_low:lat_id_low + 112, lon_id_left:lon_id_left + 112]
        gfs_test = np.load(configs.inputs_path+'GFS_0p1_f3_swvl1_interp_2018.npy')
        gfs_id_test = gfs_test[:, :3, lat_id_low:lat_id_low + 112, lon_id_left:lon_id_left + 112]
        del gfs_train, gfs_test
    print('.............................................................')
    

    # train & inference DL
    print('[HybridHydro] Training {} model'.format(configs.model_name))
    loss = train_generator(x_train, y_train, configs)
    print('[HybridHydro] Inference {} model'.format(configs.model_name))
    data_manager = DataLoader(configs)
    
    tmp_x = np.load(configs.inputs_path + '{}/X_valid_{}.npy'.format(id, id))
    tmp_y = np.load(configs.inputs_path + '{}/y_valid_{}.npy'.format(id, id))
    X, y = data_manager([tmp_x], [tmp_y])

    y_predict = np.zeros((365-configs.len_out-configs.len_input, configs.len_out, configs.h_w*4, configs.h_w*4, 1))*np.nan
    for i in range(4):
        for j in range(4):
            y_predict[:,:,i*28:(i+1)*28, j*28:(j+1)*28] = predict(X[0][:,:,i*28:(i+1)*28, j*28:(j+1)*28], configs)

    # save
    print('[HybridHydro] Saving')
    path = configs.outputs_path+'forecast/'+configs.model_name+'/'+str(id)+'/'
    if not os.path.exists(configs.outputs_path+'forecast/'+configs.model_name+'/'):
        os.mkdir(configs.outputs_path+'forecast/'+configs.model_name+'/')
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(configs.model_name+'_0p1_f16_{id:02}.npy'.format(id=id), y_predict)
    np.save('SMAPL4_0p1_f16_{id:02}.npy'.format(id=id), y[0])
    os.system('mv {} {}'.format(configs.model_name+'_0p1_f16_{id:02}.npy'.format(id=id), path))
    os.system('mv {} {}'.format('SMAPL4_0p1_f16_{id:02}.npy'.format(id=id), path))
    print('[HybridHydro] Finished! You could check the saved'+
            'models and predictions in path: {}'.format(configs.outputs_path))

if __name__ == '__main__':
    a = time.time()
    configs = parse_args()
    main(configs)
    b = time.time()
    print('[HybridHydro] Cost {} days'.format((b-a)/(3600*24)))
    
