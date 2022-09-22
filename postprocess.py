import numpy as np
import matplotlib.pyplot as plt
import glob

"""concat the 24 subregions result."""

ROOT = "/home/dell/work/lilu/CLFS/outputs/exp5-pb-tf/"
ROOT = '/tera03/lilu/work/CLFS/outputs/exp_se/'
window_size=0
pred = np.full((365-7-16-window_size,16, 336+112, 560+112, 1), np.nan)
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
        np.load(ROOT+'convlstm_tf_0p1_f7_{idx:02}.npy'.format(idx=i))
    smap[:, :, lat[i-1]:lat[i-1]+112, lon[i-1]:lon[i-1]+112] = \
            np.load(ROOT+'smapl4_0p1_f7_{idx:02}.npy'.format(idx=i))

np.save('pred_convlstm_lagy_gfs_0p1_f16.npy', pred) 
np.save('smapl4_0p1_f16.npy', smap)
