# load library
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import seaborn as sns
import tca as tc
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.basemap import Basemap
from pytesmo.metrics import tcol_snr
from sklearn.metrics import mean_squared_error, r2_score

from utils import unbiased_RMSE
from config import parse_args
import os


configs = parse_args()

# load baseline convlstm (342, 16, 448, 672)
path = configs.outputs_path+'forecast/'+configs.model_name+'/'
with xr.open_dataset(path+configs.model_name+'_0p1_f16.nc') as f:
    lat, lon, swvl = np.array(f.latitude), np.array(f.longitude), np.array(f.swvl)
swvl = np.load('/tera03/lilu/work/CLFS/outputs/exp3-best/pred_convlstm_0p1_f16.npy')[:,:,:,:,0]
#swvl = np.load('/tera03/lilu/work/CLFS/outputs/exp5-pb-tf/pred_convlstm_tf_0p1_f16.npy')[:,:,:,:,0]
#swvl = np.load('/tera06/lilu/pred_convlstm_0p1_f16.npy')[:,:,:,:,0]
print("Swvl shape is {}".format(swvl.shape))

# load agme observation
AGME_obs = np.load(configs.inputs_path+'valid_data/AGME_swvl1_2018.npy')/100
AGME_auxiliary = np.load(configs.inputs_path+'valid_data/AGME_auxiliary.npy')
lat_agme,lon_agme = AGME_auxiliary[:,0], AGME_auxiliary[:,1]
print("AGME observation shape is {}".format(AGME_obs.shape))
n_sites = AGME_obs.shape[-1]

# agme > (342, 1674) > (342, 16, 1674)
agme = np.full((342, 16, n_sites), np.nan)
for i in range(7, 349):
    agme[i-7] = AGME_obs[i:i+16]

r = np.full((len(lat_agme), configs.len_out), np.nan)
urmse = np.full((len(lat_agme), configs.len_out), np.nan)

# calculate r, ubrmse
for i in range(len(lat_agme)):
    for t in range(configs.len_out):
        print(t, i) # for each points and each scale

        # found latest grids
        lat_obs = lat_agme[i]
        lon_obs = lon_agme[i]
        index_lat = np.argmin(np.abs(lat-lat_obs))
        index_lon = np.argmin(np.abs(lon-lon_obs))

        # cal perf
        if not (np.isnan(agme[:, t, i]).any() or \
            np.isnan(swvl[:, t, index_lat, index_lon]).any()):
            urmse[i, t] = \
                unbiased_RMSE(agme[:, t, i], swvl[:, t, index_lat, index_lon])
            r[i, t] = \
                np.corrcoef(agme[:, t, i], swvl[:, t, index_lat, index_lon])[0, 1]

print(np.nanmean(r,axis=0))
# save
np.save('R_'+configs.model_name+'_AGME.npy', r)
np.save('URMSE_'+configs.model_name+'_AGME.npy', urmse)
path = configs.outputs_path+'forecast/'+configs.model_name+'/'
os.system('mv {} {}'.format('R_'+configs.model_name+'_AGME.npy', path))
os.system('mv {} {}'.format('URMSE_'+configs.model_name+'_AGME.npy', path))



