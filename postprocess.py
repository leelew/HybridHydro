import numpy as np
import matplotlib.pyplot as plt
import glob

"""concat the 24 subregions result."""

def postprocess(configs):
    pred = np.full((365-7-16-configs.window_size,16, 448, 672, 1), np.nan)
    smap = np.full((365, 1, 448, 672), np.nan)

    lon = [0, 112, 224, 336, 448, 560,
        0, 112, 224, 336, 448, 560,
        0, 112, 224, 336, 448, 560,
        0, 112, 224, 336, 448, 560]

    lat = [0, 0, 0, 0, 0, 0,
        112, 112, 112, 112, 112, 112,
        224, 224, 224, 224, 224, 224,
        336, 336, 336, 336, 336, 336]

    for i in range(1, 25):
        print(i)
        pred[:, :, lat[i-1]:lat[i-1]+112, lon[i-1]:lon[i-1]+112] = \
            np.load(configs.outputs_path+'convlstm_tf_0p1_f7_{idx:02}.npy'.format(idx=i))
        smap[:, :, lat[i-1]:lat[i-1]+112, lon[i-1]:lon[i-1]+112] = \
                np.load(configs.outputs_path+'smapl4_0p1_f7_{idx:02}.npy'.format(idx=i))

    np.save('pred_convlstm_lagy_gfs_0p1_f16.npy', pred) 
    np.save('smapl4_0p1_f16.npy', smap)
