import numpy as np
import matplotlib.pyplot as plt
import glob

ROOT = "/home/dell/work/lilu/CLFS/"

pred = np.full((339,16, 336+112, 560+112, 1), np.nan)
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
        np.load('convlstm_0p1_f7_{idx:02}.npy'.format(idx=i))
    smap[:, :, lat[i-1]:lat[i-1]+112, lon[i-1]:lon[i-1]+112] = \
            np.load('smapl4_0p1_f7_{idx:02}.npy'.format(idx=i))

np.save('pred_convlstm_0p1_f16.npy', pred) 
np.save('smapl4_0p1_f16.npy', smap)
