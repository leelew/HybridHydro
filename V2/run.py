import json
import logging
import os
import time
import numpy as np
import wandb
import xarray as xr

from config import parse_args
from data_loader import DataLoader
from trainer import predict, train_generator


def main(configs):
    """Main process for no backbone model."""
    # wandb init
    wandb.init("HybridHydro-V2")
    """
    # load SMAP L4 and make inputs
    x_train, y_train = [], []
    data_manager = DataLoader(configs)
    for id in range(1, 24):
        print(id)
        tmp_x = np.load(configs.inputs_path + '{}/X_train_{}.npy'.format(id, id))
        tmp_y = np.load(configs.inputs_path + '{}/y_train_{}.npy'.format(id, id))
        X, y = data_manager([tmp_x], [tmp_y])
        print(X[0].shape, y[0].shape)
        x_train.append(X[0])
        y_train.append(y[0])
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    num_train_sample = x_train.shape[0]
    print('[HybridHydro] Training samples shape: {}'.format(num_train_sample))
    print('.............................................................')


    # train & inference DL
    print('[HybridHydro] Training {} model'.format(configs.model_name))
    loss = train_generator(x_train, y_train, configs)
    """
    print('[HybridHydro] Inference {} model'.format(configs.model_name))
    data_manager = DataLoader(configs)
    for id in range(1, 25):
        tmp_x = np.load(configs.inputs_path + '{}/X_valid_{}.npy'.format(id, id))
        tmp_y = np.load(configs.inputs_path + '{}/y_valid_{}.npy'.format(id, id))
        X, y = data_manager([tmp_x], [tmp_y])
        y_predict = predict(X[0], configs)
        print(y_predict.shape, y[0].shape)
        # save
        print('[HybridHydro] Saving')
        path = configs.outputs_path+'forecast/'+configs.model_name+'/'+str(id)+'/'
        if not os.path.exists(configs.outputs_path+'forecast/'+configs.model_name):
            os.mkdir(configs.outputs_path+'forecast/'+configs.model_name)
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(configs.model_name+'_0p1_f16_{id:02}.npy'.format(id=id), y_predict)
        np.save('SMAPL4_0p1_f16_{id:02}.npy'.format(id=id), y[0])
        #np.save('loss_{id:02}_{model}.npy'.format(id=id,model=configs.model_name), loss)
        os.system('mv {} {}'.format(configs.model_name+'_0p1_f16_{id:02}.npy'.format(id=id), path))
        os.system('mv {} {}'.format('SMAPL4_0p1_f16_{id:02}.npy'.format(id=id), path))
        #os.system('mv {} {}'.format('loss_{id:02}_{model}.npy'.format(id=id,model=configs.model_name), path))
        print('[HybridHydro] Finished! You could check the saved'+
            'models and predictions in path: {}'.format(configs.outputs_path))


if __name__ == '__main__':
    a = time.time()
    configs = parse_args()
    main(configs)
    b = time.time()
    print('[HybridHydro] Cost {} days'.format((b-a)/(3600*24)))
    
