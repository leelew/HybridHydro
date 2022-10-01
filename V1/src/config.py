import argparse


def parse_args():
    """Hyperparameters
    Parameters
    ----------
    PATH:
    0. path_rawinputs: str
        path to load raw inputs
    1. path_inputs: str
        path to load inputs or save inputs
    2. path_outputs: str
        path to save outputs
    3. path_log: str
        path to save log, including best model etc.
    BLOCKS:
    0. downsampling: bool, optional, (default true)
        control downsampling module
    1. channel_attention: bool, optional, (default true)
        control channel attention module
    2. spatial_attention: bool, optional, (default true)
        control spatial attention module
    3. convlstm: bool, optional, (default False)
        control convlstm module:
        if true, exec covlstm module
        if false, exec encoder-decoder convlstm
    4. self_attention: bool, optional, (default true)
        control self attention module
    HYPERPARAMETERS:
    0. len_inputs: int, (default 10)
    1. height_inputs: int, (default 32)
    2. width_inputs: int, (default 32)
    3. channel_inputs: int, (default 3)
    4. len_outputs: int, (default 8)
    5. height_outputs: int, (default 8)
    6. width_outputs: int, (default 8)
    7. window_size: int, (default 1)
    8. fillvalue: float, (default -9999)
    9. train_test_ratio: float, (default 0.8)
    10. nums_input_attention: int, (default 1)
    11. nums_self_attention: int, (default 1)
    12. channel_dense_ratio: int, (default 1)
    13. spatial_kernel_size: int, (default 3)
    MODEL PARAMETERS:
    0. epoch, int, (default 1)
    1. batch_size, int, (default 100)
    2. loss, str, (default 'mse')
    3. learning_rate, float, (default 0.01)
    4. metrics, list, (default ['mae','mse'])
    5. split_ratio, float, (default 0.2)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int,default=1)

    # paths
    #parser.add_argument('--ROOT', type=str, default='/tera03/lilu/work')
    parser.add_argument('--ROOT', type=str, default='/home/dell/work/lilu')
    parser.add_argument('--x_path', type=str, default='SMAP_L4/')
    parser.add_argument('--y_path', type=str, default='SMAP_L4/')
    parser.add_argument('--raw_x_path', type=str, default='SMAP_L4/')
    parser.add_argument('--raw_y_path', type=str, default='SMAP_L4/')
    parser.add_argument('--daily_x_path', type=str, default='test/forcing/')
    parser.add_argument('--daily_y_path', type=str, default='test/SSM/')
    parser.add_argument('--saved_model_path', type=str, default='/CLFS/saved_model/')
    parser.add_argument('--saved_forecast_path', type=str, default='/CLFS/outputs/')
    parser.add_argument('--inputs_path', type=str, default='/CLFS/inputs/SMAP_L4')    

    # basic 
    parser.add_argument('--begin_train_date', type=str, default='2015-05-31')
    parser.add_argument('--end_train_date', type=str, default='2020-05-31')
    parser.add_argument('--begin_test_date', type=str, default='2020-05-31')
    parser.add_argument('--end_test_date', type=str, default='2021-05-31')
    parser.add_argument('--begin_inference_date',type=str, default='2017-12-02')
    parser.add_argument('--end_inference_date', type=str, default='2017-12-06')
    parser.add_argument('--lat_lower', type=int, default=14.7)
    parser.add_argument('--lat_upper', type=int, default=53.5)
    parser.add_argument('--lon_left', type=int, default=72.3)
    parser.add_argument('--lon_right', type=int, default=135)

    # data 
    parser.add_argument('--len_input', type=int, default=7)
    #NOTE: 0 for ideal, 2 for operational
    parser.add_argument('--window_size', type=int, default=0)
    parser.add_argument('--use_lag_y', type=bool, default=True)
    parser.add_argument('--fillvalue', type=float, default=-9999)
    parser.add_argument('--train_test_ratio', type=float, default=0.2)
    parser.add_argument('--use_GFS', type=bool, default=True)

    # model paramters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--metrics', type=list, default=['mae', 'mse'])
    parser.add_argument('--split_ratio', type=float, default=0.2)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--stats_hs', type=int, default=0)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--n_filters_factor', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--padding', type=str, default='same')
    parser.add_argument('--kernel_initializer', type=str, default='he_normal')
    parser.add_argument('--len_out',type=int, default=16)
    parser.add_argument('--cnn_inputs_shape', type=tuple, default=(112, 112, 8))
    parser.add_argument('--model_name', type=str, default='convlstm_tf')
    parser.add_argument('--validation_split', type=float,default=0.2)
    parser.add_argument('--shuffle_times', type=int,default=10000)
    #NOTE:24 for add GFS, 8 for original
    parser.add_argument('--convlstm_inputs_shape', type=tuple, default=(7, 112, 112, 8)) 
    parser.add_argument('--gfs_inputs_shape', type=tuple, default=(16, 112, 112, 3)) 

    return parser.parse_args()
