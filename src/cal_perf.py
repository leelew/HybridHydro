# load library
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from utils import unbiased_RMSE
from config import parse_args
import os

# get config
configs = parse_args()

# load model forecast (342, 16, 448, 672)
path = configs.outputs_path+'forecast/'+configs.model_name+'/'
with xr.open_dataset(path+configs.model_name+'_0p1_f16.nc') as f:
    lat, lon, swvl = np.array(f.latitude), np.array(f.longitude), np.array(f.swvl)
with xr.open_dataset(path+'SMAPL4_0p1.nc') as f:
    smap = np.array(f.swvl)
print("Swvl shape is {}".format(swvl.shape))

# load agme observation and make it to (342, 16, nsites)
AGME_obs = np.load(configs.inputs_path+'valid_data/AGME_swvl1_2018.npy')/100
AGME_auxiliary = np.load(configs.inputs_path+'valid_data/AGME_auxiliary.npy')
lat_agme,lon_agme = AGME_auxiliary[:,0], AGME_auxiliary[:,1]
print("AGME observation shape is {}".format(AGME_obs.shape))
n_sites = AGME_obs.shape[-1]

agme = np.full((342, 16, n_sites), np.nan)
smap_all = np.full((342, 16, 448, 672), np.nan)
for i in range(7, 349):
    agme[i-7] = AGME_obs[i:i+16]
    smap_all[i-7] = smap[i:i+16]


# calculate r, ubrmse
r = np.full((len(lat_agme), configs.len_out), np.nan)
urmse = np.full((len(lat_agme), configs.len_out), np.nan)
r_smap = np.full((len(lat_agme), configs.len_out), np.nan)

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
            r_smap[i, t] = \
                np.corrcoef(agme[:, t, i], smap_all[:, t, index_lat, index_lon])[0, 1]
print(np.nanmean(r,axis=0))

# save
np.save('R_'+configs.model_name+'_AGME.npy', r)
np.save('URMSE_'+configs.model_name+'_AGME.npy', urmse)
path = configs.outputs_path+'forecast/'+configs.model_name+'/'
os.system('mv {} {}'.format('R_'+configs.model_name+'_AGME.npy', path))
os.system('mv {} {}'.format('URMSE_'+configs.model_name+'_AGME.npy', path))



