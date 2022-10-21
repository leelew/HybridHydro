# load library
from cProfile import label
from distutils.command.config import config
from os import R_OK
from turtle import color
from unicodedata import category
from unittest import result

import h5py
import matplotlib
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
import os

from utils import unbiased_RMSE, get_KG, gen_week_swdi
from config import parse_args

# get config
configs = parse_args()
#--------------------LOAD--------------------------------------------------------------------------
# get koppen index
path = configs.inputs_path
kg_idx = get_KG(path)
f = nc.Dataset(path+'CN_EASE_9km.nc')
lat_smap = np.array(f['latitude'][:])
lon_smap = np.array(f['longitude'][:])
mask_in_china = np.load(path+'mask_in_china.npy')

print('[HybridHydro] Loading GFS')
# load 0.25 gfs (365, 16, 721, 1440)
gfs = np.load(path+'valid_data/GFS_0p25_f16_swvl1_2018.npy')
lat_gfs = np.arange(90, -90.1, -0.25)
lon_gfs = np.arange(0, 359.8, 0.25)
idx_lat = np.where((lat_gfs>14.7) & (lat_gfs<53.5))[0]
idx_lon = np.where((lon_gfs>72.3) & (lon_gfs<135))[0]
gfs = gfs[:, :, idx_lat, :][:, :, :, idx_lon]
lat_gfs = lat_gfs[idx_lat]
lon_gfs = lon_gfs[idx_lon]
_, nt, nlat_gfs, nlon_gfs = gfs.shape
# load 0.1 GFS over China (365, 16, 448, 672)
gfs_d = np.load(path+'GFS_0p1_f16_swvl1_2018.npy')

print('[HybridHydro] Loading DL forecasts')
path = configs.saved_forecast_path
# load convlstm (342, 16, 448, 672)
convlstm = np.load(path+'convlstm/pred_convlstm_0p1_f16.npy')[:, :, :, :, 0]
# load drive convlstm (342, 16, 448, 672)
convlstm_pb = np.load(path+'convlstm_drive/pred_convlstm_0p1_f16.npy')[:, :, :, :, 0]
# load condition convlstm (342, 16, 448, 672)
convlstm_pb_tf = np.load(path+'convlstm_condition/pred_convlstm_tf_0p1_f16_1.npy')[:, :, :, :, 0]
# load att condition convlstm (342, 16, 448, 672)
convlstm_pb_att_tf = np.load(path+'convlstm_att_condition/pred_convlstm_tf_0p1_f16.npy')[:, :, :, :, 0]
_, nt, nlat_pred, nlon_pred = convlstm.shape
"""
# load gfs tf att
convlstm_noy = np.load(path + 'exp8-no-lagy/pred_convlstm_no_lagy_0p1_f16.npy')[:, :, :, :, 0]
# load gfs tf att
convlstm_noy_pb = np.load(path + 'exp9-no-lagy-gfs-drived/pred_convlstm_no_lagy_drived_0p1_f16.npy')[:, :, :, :, 0]
convlstm_lagy = np.load(path + 'exp_lagy/pred_convlstm_lagy_0p1_f16.npy')[:, :, :, :, 0]
convlstm_lagy_pb = np.load(path + 'exp_lagy_gfs/pred_convlstm_lagy_gfs_0p1_f16.npy')[:, :, :, :, 0]
"""

print('[HybridHydro] Loading validation data')
path = configs.inputs_path+'valid_data/'
# load agme observation
AGME_obs = np.load(path+'AGME_swvl1_2018.npy')/100
AGME_auxiliary = np.load(path + 'AGME_auxiliary.npy')
lat_agme = AGME_auxiliary[:,0]
lon_agme = AGME_auxiliary[:,1]
n_sites = AGME_obs.shape[-1]
# load somo (365, 721, 1440)
with xr.open_dataset(path+'SoMo.ml_v1_layer1_2018.nc') as ds:
    somo  = np.array(ds.layer1)
    lat_somo = np.array(ds.lat)
    lon_somo = np.array(ds.lon)
idx_lat = np.where((lat_somo>14.7) & (lat_somo<53.5))[0]
idx_lon = np.where((lon_somo>72.3) & (lon_somo<135))[0]
lat_somo = lat_somo[idx_lat]
lon_somo = lon_somo[idx_lon]
somo = somo[:, idx_lat, :][:, :, idx_lon]
# load era5-land
era5land = np.load(path+'ERA5-land_swvl1_2018.npy')
land_mask = np.nanmean(era5land, axis=0)
land_mask[~np.isnan(land_mask)] = 0
# load smap
smap = np.load(path+'smapl4_0p1_f16.npy')
# load SMOS L3
smos = np.load(path+"SMOS_L3_2018.npy")
lat_smos = np.load(path+"SMOS_L3_lat.npy")
lon_smos = np.load(path+"SMOS_L3_lon.npy")
# load LPRM-AMSR2
amsr2 = np.load(path+"LPRM-AMSR2_L3_DS_D_SOILM3_V001_2018.npy")
lat_amsr2 = np.load(path+"LPRM-AMSR2_lat.npy")
lon_amsr2 = np.load(path+"LPRM-AMSR2_lon.npy")
#--------------------------------------------------------------------------------------------------


#--------------------PREPROCESS--------------------------------------------------------------------
print('[HybridHydro] Preprocessing')
# gfs > (342, 16, 448, 672)
gfs = gfs[7:349]
# gfs_downscaling
gfs_d = gfs_d[7:349]
# baseline > (339, 16, 448, 672)
convlstm = convlstm + land_mask
# mean
gfs_convlstm_mean = (gfs_d+convlstm)/2 + land_mask
# pb tf att mean
gfs_convlstm_pb_att_tf_mean = (gfs_convlstm_mean+convlstm_pb_tf+convlstm_pb_att_tf)/3 + land_mask
# agme > (342, 1674) > (342, 16, 1674)
agme = np.full((342, 16, n_sites), np.nan)
for i in range(7, 349):
    agme[i-7] = AGME_obs[i:i+16]
# somo > (342, 16, 155, 250)
SoMo = np.full((342, 16, 155, 251), np.nan)
for i in range(7, 349):
    SoMo[i-11] = somo[i:i+16]
# era5 > (342, 16, 448, 672)
era5 = np.full((342, 16, 448, 672), np.nan)
for i in range(7, 349):
    era5[i-7] = era5land[i:i+16,0]
# smap > (342, 16, 448, 672)
smapl4 = np.full((342, 16, 448, 672), np.nan)
for i in range(7, 349):
    smapl4[i-7] = smap[i:i+16,0]
# smos > (342, 16, 161, 241)
f_smos = np.full((342, 16, 161, 241), np.nan)
for i in range(7, 349):
    f_smos[i-7] = smos[i:i+16]
# amsr2 > (342, 16, 388, 627)
f_amsr2 = np.full((342, 16, 388, 627), np.nan)
for i in range(7, 349):
    f_amsr2[i-7] = amsr2[i:i+16]
#-------------------------------------------------------------------------------------------------


"""
#-------------------------------------------------------------------------------------------------
# calculate conventional metrics
#-------------------------------------------------------------------------------------------------
print('[HybridHydro] Calculate R and URMSE of forecast')
urmse_gfs = np.full((len(lat_agme), nt), np.nan)
urmse_convlstm = np.full((len(lat_agme), nt), np.nan)
urmse_smap = np.full((len(lat_agme), nt), np.nan)
urmse_convlstm_pb = np.full((len(lat_agme), nt), np.nan)
urmse_convlstm_pb_tf = np.full((len(lat_agme), nt), np.nan)
urmse_convlstm_mean = np.full((len(lat_agme), nt), np.nan)
urmse_convlstm_pb_att_tf = np.full((len(lat_agme), nt), np.nan)
urmse_convlstm_pb_att_tf_mean = np.full((len(lat_agme), nt), np.nan)
r_gfs = np.full((len(lat_agme), nt), np.nan)
r_convlstm = np.full((len(lat_agme), nt), np.nan)
r_smap = np.full((len(lat_agme), nt), np.nan)
r_convlstm_pb = np.full((len(lat_agme), nt), np.nan)
r_convlstm_pb_tf = np.full((len(lat_agme), nt), np.nan)
r_convlstm_mean = np.full((len(lat_agme), nt), np.nan)
r_convlstm_pb_att_tf = np.full((len(lat_agme), nt), np.nan)
r_convlstm_pb_att_tf_mean = np.full((len(lat_agme), nt), np.nan)

# calculate r, ubrmse
for i in range(len(lat_agme)):
    for t in range(nt):
        print(t, i)
        lat_obs = lat_agme[i]
        lon_obs = lon_agme[i]
        index_lat = np.argmin(np.abs(lat_gfs-lat_obs))
        index_lon = np.argmin(np.abs(lon_gfs-lon_obs))
        
        if not (np.isnan(agme[:, t, i]).any() or np.isnan(gfs[:, t, index_lat, index_lon]).any()):
            urmse_gfs[i, t] = unbiased_RMSE(agme[:, t, i], gfs[:, t, index_lat, index_lon])
            r_gfs[i, t] = np.corrcoef(agme[:, t, i], gfs[:, t, index_lat, index_lon])[0, 1]

        index_lat = np.argmin(np.abs(lat_smap-lat_obs))
        index_lon = np.argmin(np.abs(lon_smap-lon_obs))
        
        if not (np.isnan(agme[:, t, i]).any() or \
            np.isnan(convlstm[:, t, index_lat, index_lon]).any()):
            urmse_convlstm[i, t] = \
                unbiased_RMSE(agme[:, t, i], convlstm[:, t, index_lat, index_lon])
            r_convlstm[i, t] = \
                np.corrcoef(agme[:, t, i], convlstm[:, t, index_lat, index_lon])[0, 1]
            urmse_convlstm_pb_att_tf[i, t] = \
                unbiased_RMSE(agme[:, t, i], convlstm_pb_att_tf[:, t, index_lat, index_lon])
            r_convlstm_pb_att_tf[i, t] = \
                np.corrcoef(agme[:, t, i], convlstm_pb_att_tf[:, t, index_lat, index_lon])[0, 1]
            urmse_convlstm_pb_att_tf_mean[i, t] = \
                unbiased_RMSE(agme[:, t, i], gfs_convlstm_pb_att_tf_mean[:, t, index_lat, index_lon])
            r_convlstm_pb_att_tf_mean[i, t] = \
                np.corrcoef(agme[:, t, i], gfs_convlstm_pb_att_tf_mean[:, t, index_lat, index_lon])[0, 1]
            urmse_convlstm_pb[i, t] = \
                unbiased_RMSE(agme[:, t, i], convlstm_pb[:, t, index_lat, index_lon])
            r_convlstm_pb[i, t] = \
                np.corrcoef(agme[:, t, i], convlstm_pb[:, t, index_lat, index_lon])[0, 1]
            urmse_convlstm_pb_tf[i, t] = \
                unbiased_RMSE(agme[:, t, i], convlstm_pb_tf[:, t, index_lat, index_lon])
            r_convlstm_pb_tf[i, t] = \
                np.corrcoef(agme[:, t, i], convlstm_pb_tf[:, t, index_lat, index_lon])[0, 1]
            urmse_convlstm_mean[i, t] = \
                unbiased_RMSE(agme[:, t, i], gfs_convlstm_mean[:, t, index_lat, index_lon])
            r_convlstm_mean[i, t] = \
                np.corrcoef(agme[:, t, i], gfs_convlstm_mean[:, t, index_lat, index_lon])[0, 1]

        if not (np.isnan(agme[:, t, i]).any() or \
            np.isnan(smapl4[:, t, index_lat, index_lon]).any()):
            urmse_smap[i, t] = \
                unbiased_RMSE(agme[:, t, i], smapl4[:, t, index_lat, index_lon])
            r_smap[i, t] = \
                np.corrcoef(agme[:, t, i], smapl4[:, t, index_lat, index_lon])[0, 1]

        if not (np.isnan(agme[:, t, i]).any() or \
            np.isnan(gfs_d[:, t, index_lat, index_lon]).any()):
            urmse_gfs[i, t] = \
                unbiased_RMSE(agme[:, t, i], gfs_d[:, t, index_lat, index_lon])
            r_gfs[i, t] = \
                np.corrcoef(agme[:, t, i], gfs_d[:, t, index_lat, index_lon])[0, 1]

r_all = np.stack([r_gfs, r_convlstm, r_convlstm_mean, \
    r_convlstm_pb, r_convlstm_pb_tf, \
    r_convlstm_pb_att_tf, r_convlstm_pb_att_tf_mean, r_smap], axis=0)
urmse_all = np.stack([urmse_gfs, urmse_convlstm, urmse_convlstm_mean, \
    urmse_convlstm_pb, urmse_convlstm_pb_tf, \
    urmse_convlstm_pb_att_tf, urmse_convlstm_pb_att_tf_mean, urmse_smap], axis=0)
np.save('urmse.npy', urmse_all)
np.save('r.npy', r_all)
"""
"""
r_noy_drived = np.full((len(lat_agme), nt), np.nan)
r_noy = np.full((len(lat_agme), nt), np.nan)
# calculate r, ubrmse
for i in range(len(lat_agme)):
    for t in range(nt):
        print(t, i)
        lat_obs = lat_agme[i]
        lon_obs = lon_agme[i]
        index_lat = np.argmin(np.abs(lat_smap-lat_obs))
        index_lon = np.argmin(np.abs(lon_smap-lon_obs))
        print('distance of baseline and in-situ data')
        print(np.nanmin(np.abs(lat_smap-lat_obs)), np.nanmin(np.abs(lon_smap-lon_obs)))
        
        if not (np.isnan(agme[:, t, i]).any() or \
            np.isnan(convlstm_noy[:, t, index_lat, index_lon]).any()):
            r_noy[i, t] = \
                unRMSE(agme[:, t, i], convlstm_noy[:, t, index_lat, index_lon])
            r_noy_drived[i, t] = \
                np.corrcoef(agme[:, t, i], convlstm_noy_pb[:, t, index_lat, index_lon])[0, 1]
np.save('r_noy.npy', r_noy)
np.save('r_noy_drived.npy', r_noy_drived)

r_lagy_drived = np.full((len(lat_agme), nt), np.nan)
r_lagy = np.full((len(lat_agme), nt), np.nan)
# calculate r, ubrmse
for i in range(len(lat_agme)):
    for t in range(nt):
        print(t, i)
        lat_obs = lat_agme[i]
        lon_obs = lon_agme[i]
        index_lat = np.argmin(np.abs(lat_smap-lat_obs))
        index_lon = np.argmin(np.abs(lon_smap-lon_obs))
        print('distance of baseline and in-situ data')
        print(np.nanmin(np.abs(lat_smap-lat_obs)), np.nanmin(np.abs(lon_smap-lon_obs)))
        
        if not (np.isnan(agme[:, t, i]).any() or \
            np.isnan(convlstm_lagy[:, t, index_lat, index_lon]).any()):
            r_lagy[i, t] = \
                unRMSE(agme[:, t, i], convlstm_lagy[:, t, index_lat, index_lon])
            r_lagy_drived[i, t] = \
                np.corrcoef(agme[:, t, i], convlstm_lagy_pb[:, t, index_lat, index_lon])[0, 1]
np.save('r_lagy.npy', r_lagy)
np.save('r_lagy_drived.npy', r_lagy_drived)
#--------------------------------------------------------------------------------------------------
"""


"""
#--------------------------------------------------------------------------------------------------
# calculate TCA of (ERA5-Land, SoMo)
#--------------------------------------------------------------------------------------------------
print('[HybridHydro] Calculate TCA(ERA5LAND, SOMO)')
tc_convlstm = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_gfs = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_convlstm = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_smap = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_convlstm_pb = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_convlstm_pb_tf = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_convlstm_mean = np.full((nt, nlat_pred, nlon_pred),np.nan)
tc_convlstm_pb_att_tf = np.full((nt, nlat_pred, nlon_pred),np.nan)
tc_convlstm_pb_att_tf_mean = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_smap = np.full((nlat_pred, nlon_pred), np.nan)

for t in [0, 6, -1]:
    for i in range(nlat_pred):
        for j in range(nlon_pred):
            print(t, i, j)
            # location
            lat_obs = lat_smap[i]
            lon_obs = lon_smap[j]
            index_lat_era5 = np.argmin(np.abs(lat_smap-lat_obs))
            index_lon_era5 = np.argmin(np.abs(lon_smap-lon_obs))
            index_lat_somo = np.argmin(np.abs(lat_somo-lat_obs))
            index_lon_somo = np.argmin(np.abs(lon_somo-lon_obs))

            # array
            x1 = convlstm[:, t, i, j]
            x2 = gfs_convlstm_mean[:, t, i, j]
            x3 = convlstm_pb_tf[:, t, i, j]
            x5 = convlstm_pb_att_tf[:, t, i, j]
            x6 = gfs_convlstm_pb_att_tf_mean[:, t, i, j]
            y = era5[:, t, index_lat_era5, index_lon_era5]
            z = SoMo[:, t, index_lat_somo, index_lon_somo]

            # cal SNR
            if not (np.isnan(x1).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x1, y, z)
                tc_convlstm[t, i, j] = snr[0]
            if not (np.isnan(x2).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x2, y, z)
                tc_convlstm_mean[t, i, j] = snr[0]
            if not (np.isnan(x3).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x3, y, z)
                tc_convlstm_pb_tf[t, i, j] = snr[0]
            if not (np.isnan(x5).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x5, y, z)
                tc_convlstm_pb_att_tf[t, i, j] = snr[0]   
            if not (np.isnan(x6).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x6, y, z)
                tc_convlstm_pb_att_tf_mean[t, i, j] = snr[0]

for t in [0, 6, -1]:
    for i in range(nlat_pred):
        for j in range(nlon_pred):
            print(t, i, j)
            lat_obs = lat_smap[i]
            lon_obs = lon_smap[j]
            index_lat_era5 = np.argmin(np.abs(lat_smap-lat_obs))
            index_lon_era5 = np.argmin(np.abs(lon_smap-lon_obs))
            index_lat_somo = np.argmin(np.abs(lat_somo-lat_obs))
            index_lon_somo = np.argmin(np.abs(lon_somo-lon_obs))
            x = gfs_d[:, t, i, j]
            y = era5[:, t, index_lat_era5, index_lon_era5]
            z = SoMo[:, t, index_lat_somo, index_lon_somo]
            if not (np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x, y, z)
                tc_gfs[t, i, j] = snr[0]

for i in range(nlat_pred):
    for j in range(nlon_pred):
        # location
        lat_obs = lat_smap[i]
        lon_obs = lon_smap[j]
        index_lat_somo = np.argmin(np.abs(lat_somo-lat_obs))
        index_lon_somo = np.argmin(np.abs(lon_somo-lon_obs))
        # array
        x1 = smap[:, 0, i, j]
        y = era5land[:, 0, i, j]
        z = somo[:, index_lat_somo, index_lon_somo]
        # cal SNR
        if not (np.isnan(x1).any() or np.isnan(y).any() or np.isnan(z).any()):
            snr, _, _ = tcol_snr(x1, y, z)
            tc_smap[i, j] = snr[0]

tc_all = np.stack([tc_convlstm, tc_convlstm_mean, \
    tc_convlstm_pb_tf, 
    tc_convlstm_pb_att_tf, tc_convlstm_pb_att_tf_mean, tc_gfs], axis=0)
np.save('tc.npy', tc_all)
np.save('tc_smap.npy', tc_smap)
# -------------------------------------------------------------------------------------------------
"""




#--------------------------------------------------------------------------------------------------
# calculate TCA of (SMOS, AMSR2)
#--------------------------------------------------------------------------------------------------
print('[HybridHydro] Calculate TCA(SMOS, AMSR2)')
tc_convlstm = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_gfs = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_convlstm = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_smap = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_convlstm_pb = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_convlstm_pb_tf = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_convlstm_mean = np.full((nt, nlat_pred, nlon_pred),np.nan)
tc_convlstm_pb_att_tf = np.full((nt, nlat_pred, nlon_pred),np.nan)
tc_convlstm_pb_att_tf_mean = np.full((nt, nlat_pred, nlon_pred), np.nan)
tc_smap = np.full((nlat_pred, nlon_pred), np.nan)

for t in [0, 6, -1]:
    for i in range(nlat_pred):
        for j in range(nlon_pred):
            # location
            lat_obs = lat_smap[i]
            lon_obs = lon_smap[j]
            index_lat_smos = np.argmin(np.abs(lat_smos-lat_obs))
            index_lon_smos = np.argmin(np.abs(lon_smos-lon_obs))
            index_lat_amsr2 = np.argmin(np.abs(lat_amsr2-lat_obs))
            index_lon_amsr2 = np.argmin(np.abs(lon_amsr2-lon_obs))

            # array
            x1 = convlstm[:, t, i, j]
            x2 = gfs_convlstm_mean[:, t, i, j]
            x3 = convlstm_pb_tf[:, t, i, j]
            x5 = convlstm_pb_att_tf[:, t, i, j]
            x6 = gfs_convlstm_pb_att_tf_mean[:, t, i, j]
            y = f_smos[:, t, index_lat_smos, index_lon_smos]
            z = f_amsr2[:, t, index_lat_amsr2, index_lon_amsr2]
            x = gfs_d[:, t, i, j]

            x1 = np.delete(x1, np.where(np.isnan(y)), axis=0)
            x2 = np.delete(x2, np.where(np.isnan(y)), axis=0)
            x3 = np.delete(x3, np.where(np.isnan(y)), axis=0)
            x5 = np.delete(x5, np.where(np.isnan(y)), axis=0)
            x6 = np.delete(x6, np.where(np.isnan(y)), axis=0)
            x = np.delete(x, np.where(np.isnan(y)), axis=0)
            z = np.delete(z, np.where(np.isnan(y)), axis=0)
            y = np.delete(y, np.where(np.isnan(y)), axis=0)

            x1 = np.delete(x1, np.where(np.isnan(z)), axis=0)
            x2 = np.delete(x2, np.where(np.isnan(z)), axis=0)
            x3 = np.delete(x3, np.where(np.isnan(z)), axis=0)
            x5 = np.delete(x5, np.where(np.isnan(z)), axis=0)
            x6 = np.delete(x6, np.where(np.isnan(z)), axis=0)
            x = np.delete(x, np.where(np.isnan(z)), axis=0)
            y = np.delete(y, np.where(np.isnan(z)), axis=0)
            z = np.delete(z, np.where(np.isnan(z)), axis=0)

            if len(y) > 20:
                # cal SNR
                if not (np.isnan(x1).any() or np.isnan(y).any() or np.isnan(z).any()):
                    snr, _, _ = tcol_snr(x1, y, z)
                    tc_convlstm[t, i, j] = snr[0]
                if not (np.isnan(x2).any() or np.isnan(y).any() or np.isnan(z).any()):
                    snr, _, _ = tcol_snr(x2, y, z)
                    tc_convlstm_mean[t, i, j] = snr[0]
                if not (np.isnan(x3).any() or np.isnan(y).any() or np.isnan(z).any()):
                    snr, _, _ = tcol_snr(x3, y, z)
                    tc_convlstm_pb_tf[t, i, j] = snr[0]
                if not (np.isnan(x5).any() or np.isnan(y).any() or np.isnan(z).any()):
                    snr, _, _ = tcol_snr(x5, y, z)
                    tc_convlstm_pb_att_tf[t, i, j] = snr[0]   
                if not (np.isnan(x6).any() or np.isnan(y).any() or np.isnan(z).any()):
                    snr, _, _ = tcol_snr(x6, y, z)
                    tc_convlstm_pb_att_tf_mean[t, i, j] = snr[0]
                if not (np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any()):
                    snr, _, _ = tcol_snr(x, y, z)
                    tc_gfs[t, i, j] = snr[0]

tc_all = np.stack([
    tc_convlstm, tc_convlstm_mean, \
    tc_convlstm_pb_tf, tc_convlstm_pb_att_tf, \
    tc_convlstm_pb_att_tf_mean, tc_gfs], axis=0)
np.save('tc_smos_amsr2.npy', tc_all)
# -------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------
# calculate TCA of (SMOS, AGME)
#--------------------------------------------------------------------------------------------------
print('[HybridHydro] Calculate TCA(SMOS, AGME)')
tc_convlstm = np.full((nt, len(lat_agme)), np.nan)
tc_gfs = np.full((nt, len(lat_agme)), np.nan)
tc_convlstm = np.full((nt, len(lat_agme)), np.nan)
tc_smap = np.full((nt, len(lat_agme)), np.nan)
tc_convlstm_pb = np.full((nt, len(lat_agme)), np.nan)
tc_convlstm_pb_tf = np.full((nt, len(lat_agme)), np.nan)
tc_convlstm_mean = np.full((nt, len(lat_agme)),np.nan)
tc_convlstm_pb_att_tf = np.full((nt, len(lat_agme)),np.nan)
tc_convlstm_pb_att_tf_mean = np.full((nt, len(lat_agme)), np.nan)
tc_smap = np.full((len(lat_agme)), np.nan)

for t in [0, 6, -1]:
    for i in range(len(lat_agme)):
        lat_obs = lat_agme[i]
        lon_obs = lon_agme[i]
        index_lat_smos = np.argmin(np.abs(lat_smos-lat_obs))
        index_lon_smos = np.argmin(np.abs(lon_smos-lon_obs))
        index_lat_smap = np.argmin(np.abs(lat_smap-lat_obs))
        index_lon_smap = np.argmin(np.abs(lon_smap-lon_obs))

        x1 = convlstm[:, t, index_lat_smap, index_lon_smap]
        x2 = gfs_convlstm_mean[:, t, index_lat_smap, index_lon_smap]
        x3 = convlstm_pb_tf[:, t, index_lat_smap, index_lon_smap]
        x5 = convlstm_pb_att_tf[:, t, index_lat_smap, index_lon_smap]
        x6 = gfs_convlstm_pb_att_tf_mean[:, t, index_lat_smap, index_lon_smap]
        y = f_smos[:, t, index_lat_smos, index_lon_smos]
        z = agme[:, t, i]
        x = gfs_d[:, t, index_lat_smap, index_lon_smap]

        x1 = np.delete(x1, np.where(np.isnan(y)), axis=0)
        x2 = np.delete(x2, np.where(np.isnan(y)), axis=0)
        x3 = np.delete(x3, np.where(np.isnan(y)), axis=0)
        x5 = np.delete(x5, np.where(np.isnan(y)), axis=0)
        x6 = np.delete(x6, np.where(np.isnan(y)), axis=0)
        x = np.delete(x, np.where(np.isnan(y)), axis=0)
        z = np.delete(z, np.where(np.isnan(y)), axis=0)
        y = np.delete(y, np.where(np.isnan(y)), axis=0)

        x1 = np.delete(x1, np.where(np.isnan(z)), axis=0)
        x2 = np.delete(x2, np.where(np.isnan(z)), axis=0)
        x3 = np.delete(x3, np.where(np.isnan(z)), axis=0)
        x5 = np.delete(x5, np.where(np.isnan(z)), axis=0)
        x6 = np.delete(x6, np.where(np.isnan(z)), axis=0)
        x = np.delete(x, np.where(np.isnan(z)), axis=0)
        y = np.delete(y, np.where(np.isnan(z)), axis=0)
        z = np.delete(z, np.where(np.isnan(z)), axis=0)

        if len(y) > 50:
            # cal SNR
            if not (np.isnan(x1).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x1, y, z)
                tc_convlstm[t, i] = snr[0]
            if not (np.isnan(x2).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x2, y, z)
                tc_convlstm_mean[t, i] = snr[0]
            if not (np.isnan(x3).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x3, y, z)
                tc_convlstm_pb_tf[t, i] = snr[0]
            if not (np.isnan(x5).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x5, y, z)
                tc_convlstm_pb_att_tf[t, i] = snr[0]   
            if not (np.isnan(x6).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x6, y, z)
                tc_convlstm_pb_att_tf_mean[t, i] = snr[0]
            if not (np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any()):
                snr, _, _ = tcol_snr(x, y, z)
                tc_gfs[t, i] = snr[0]

tc_all = np.stack([\
    tc_convlstm, tc_convlstm_mean, \
    tc_convlstm_pb_tf, tc_convlstm_pb_att_tf, \
    tc_convlstm_pb_att_tf_mean, tc_gfs], axis=0)
np.save('tc_smos_agme.npy', tc_all)
# -------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------
# calculate R of SWDI
# -------------------------------------------------------------------------------------------------
print('[HybridHydro] Calculate R and URMSE of SWDI')
swdi_w1_gfs, swdi_w2_gfs = gen_week_swdi(gfs)
swdi_w1_convlstm, swdi_w2_convlstm = gen_week_swdi(convlstm)
swdi_w1_mean, swdi_w2_mean = gen_week_swdi(gfs_convlstm_mean)
swdi_w1_tf, swdi_w2_tf = gen_week_swdi(convlstm_pb_tf)
swdi_w1_att_tf, swdi_w2_att_tf = gen_week_swdi(convlstm_pb_att_tf)
swdi_w1_att_tf_mean, swdi_w2_att_tf_mean = gen_week_swdi(gfs_convlstm_pb_att_tf_mean)
swdi_w1_agme, swdi_w2_agme = gen_week_swdi(agme)
"""
urmse_gfs = np.full((len(lat_agme), 2), np.nan)
urmse_convlstm = np.full((len(lat_agme), 2), np.nan)
urmse_smap = np.full((len(lat_agme), 2), np.nan)
urmse_convlstm_pb = np.full((len(lat_agme), 2), np.nan)
urmse_convlstm_pb_tf = np.full((len(lat_agme), 2), np.nan)
urmse_convlstm_pb_tf_mean = np.full((len(lat_agme), 2), np.nan)
urmse_convlstm_mean = np.full((len(lat_agme), 2), np.nan)
urmse_convlstm_pb_att_tf = np.full((len(lat_agme), 2), np.nan)
urmse_convlstm_pb_att_tf_mean = np.full((len(lat_agme), 2), np.nan)
r_gfs = np.full((len(lat_agme), 2), np.nan)
r_convlstm = np.full((len(lat_agme), 2), np.nan)
r_smap = np.full((len(lat_agme), 2), np.nan)
r_convlstm_pb = np.full((len(lat_agme), 2), np.nan)
r_convlstm_pb_tf = np.full((len(lat_agme), 2), np.nan)
r_convlstm_pb_tf_mean = np.full((len(lat_agme), 2), np.nan)
r_convlstm_mean = np.full((len(lat_agme), 2), np.nan)
r_convlstm_pb_att_tf = np.full((len(lat_agme), 2), np.nan)
r_convlstm_pb_att_tf_mean = np.full((len(lat_agme), 2), np.nan)

# calculate week1 r, ubrmse
for i in range(len(lat_agme)):
    lat_obs = lat_agme[i]
    lon_obs = lon_agme[i]
    index_lat = np.argmin(np.abs(lat_gfs-lat_obs))
    index_lon = np.argmin(np.abs(lon_gfs-lon_obs))
    print('distance of gfs and in-situ data')
    print(np.nanmin(np.abs(lat_gfs-lat_obs)), np.nanmin(np.abs(lon_gfs-lon_obs)))
    
    if not (np.isnan(swdi_w1_agme[:, i]).any() or \
        np.isnan(swdi_w1_gfs[:, index_lat, index_lon]).any()):
        urmse_gfs[i, 0] = unbiased_RMSE(swdi_w1_agme[:, i], swdi_w1_gfs[:, index_lat, index_lon])
        r_gfs[i, 0] = np.corrcoef(swdi_w1_agme[:, i], swdi_w1_gfs[:, index_lat, index_lon])[0, 1]

    index_lat = np.argmin(np.abs(lat_smap-lat_obs))
    index_lon = np.argmin(np.abs(lon_smap-lon_obs))
    print('distance of baseline and in-situ data')
    print(np.nanmin(np.abs(lat_smap-lat_obs)), np.nanmin(np.abs(lon_smap-lon_obs)))
    
    if not (np.isnan(swdi_w1_agme[:, i]).any() or \
        np.isnan(swdi_w2_convlstm[:, index_lat, index_lon]).any()):
        urmse_convlstm[i, 0] = \
            unbiased_RMSE(swdi_w1_agme[:, i], swdi_w1_convlstm[:, index_lat, index_lon])
        r_convlstm[i, 0] = \
            np.corrcoef(swdi_w1_agme[:, i], swdi_w1_convlstm[:, index_lat, index_lon])[0, 1]
        urmse_convlstm_pb_att_tf[i, 0] = \
            unbiased_RMSE(swdi_w1_agme[:, i], swdi_w1_att_tf[:, index_lat, index_lon])
        r_convlstm_pb_att_tf[i, 0] = \
            np.corrcoef(swdi_w1_agme[:, i], swdi_w1_att_tf[:, index_lat, index_lon])[0, 1]
        urmse_convlstm_pb_att_tf_mean[i, 0] = \
            unbiased_RMSE(swdi_w1_agme[:, i], swdi_w1_att_tf_mean[:, index_lat, index_lon])
        r_convlstm_pb_att_tf_mean[i, 0] = \
            np.corrcoef(swdi_w1_agme[:, i], swdi_w1_att_tf_mean[:, index_lat, index_lon])[0, 1]
        urmse_convlstm_pb_tf[i, 0] = \
            unbiased_RMSE(swdi_w1_agme[:, i], swdi_w1_tf[:, index_lat, index_lon])
        r_convlstm_pb_tf[i, 0] = \
            np.corrcoef(swdi_w1_agme[:, i], swdi_w1_tf[:, index_lat, index_lon])[0, 1]
        urmse_convlstm_mean[i, 0] = \
            unbiased_RMSE(swdi_w1_agme[:, i], swdi_w1_mean[:, index_lat, index_lon])
        r_convlstm_mean[i, 0] = \
            np.corrcoef(swdi_w1_agme[:, i], swdi_w1_mean[:, index_lat, index_lon])[0, 1]

    #if not (np.isnan(agme[:, t, i]).any() or \
    #    np.isnan(smapl4[:, t, index_lat, index_lon]).any()):

    #    urmse_smap[i, t] = \
    #        unRMSE(agme[:, t, i], smapl4[:, t, index_lat, index_lon])
    #    r_smap[i, t] = \
    #        np.corrcoef(agme[:, t, i], smapl4[:, t, index_lat, index_lon])[0, 1]


# calculate week2 r, ubrmse
for i in range(len(lat_agme)):

    lat_obs = lat_agme[i]
    lon_obs = lon_agme[i]
    
    index_lat = np.argmin(np.abs(lat_gfs-lat_obs))
    index_lon = np.argmin(np.abs(lon_gfs-lon_obs))
    print('distance of gfs and in-situ data')
    print(np.nanmin(np.abs(lat_gfs-lat_obs)), np.nanmin(np.abs(lon_gfs-lon_obs)))
    
    if not (np.isnan(swdi_w2_agme[:, i]).any() or \
        np.isnan(swdi_w2_gfs[:, index_lat, index_lon]).any()):

        urmse_gfs[i, 1] = unbiased_RMSE(swdi_w2_agme[:, i], swdi_w2_gfs[:, index_lat, index_lon])
        r_gfs[i, 1] = np.corrcoef(swdi_w2_agme[:, i], swdi_w2_gfs[:, index_lat, index_lon])[0, 1]

    index_lat = np.argmin(np.abs(lat_smap-lat_obs))
    index_lon = np.argmin(np.abs(lon_smap-lon_obs))
    print('distance of baseline and in-situ data')
    print(np.nanmin(np.abs(lat_smap-lat_obs)), np.nanmin(np.abs(lon_smap-lon_obs)))
    
    if not (np.isnan(swdi_w2_agme[:, i]).any() or \
        np.isnan(swdi_w2_convlstm[:, index_lat, index_lon]).any()):
        urmse_convlstm[i, 1] = \
            unbiased_RMSE(swdi_w2_agme[:, i], swdi_w2_convlstm[:, index_lat, index_lon])
        r_convlstm[i, 1] = \
            np.corrcoef(swdi_w2_agme[:, i], swdi_w2_convlstm[:, index_lat, index_lon])[0, 1]
        urmse_convlstm_pb_att_tf[i, 1] = \
            unbiased_RMSE(swdi_w2_agme[:, i], swdi_w2_att_tf[:, index_lat, index_lon])
        r_convlstm_pb_att_tf[i, 1] = \
            np.corrcoef(swdi_w2_agme[:, i], swdi_w2_att_tf[:, index_lat, index_lon])[0, 1]
        urmse_convlstm_pb_att_tf_mean[i, 1] = \
            unbiased_RMSE(swdi_w2_agme[:, i], swdi_w2_att_tf_mean[:, index_lat, index_lon])
        r_convlstm_pb_att_tf_mean[i, 1] = \
            np.corrcoef(swdi_w2_agme[:, i], swdi_w2_att_tf_mean[:, index_lat, index_lon])[0, 1]
        urmse_convlstm_pb_tf[i, 1] = \
            unbiased_RMSE(swdi_w2_agme[:, i], swdi_w2_tf[:, index_lat, index_lon])
        r_convlstm_pb_tf[i, 1] = \
            np.corrcoef(swdi_w2_agme[:, i], swdi_w2_tf[:, index_lat, index_lon])[0, 1]
        urmse_convlstm_mean[i, 1] = \
            unbiased_RMSE(swdi_w2_agme[:, i], swdi_w2_mean[:, index_lat, index_lon])
        r_convlstm_mean[i, 1] = \
            np.corrcoef(swdi_w2_agme[:, i], swdi_w2_mean[:, index_lat, index_lon])[0, 1]
r_all = np.stack([r_gfs, r_convlstm, r_convlstm_mean, \
    r_convlstm_pb_tf, 
    r_convlstm_pb_att_tf, r_convlstm_pb_att_tf_mean], axis=0)
np.save('r_swdi.npy', r_all)
urmse_all = np.stack([urmse_gfs, urmse_convlstm, urmse_convlstm_mean, \
    urmse_convlstm_pb_tf, 
    urmse_convlstm_pb_att_tf, urmse_convlstm_pb_att_tf_mean], axis=0)
np.save('urmse_swdi.npy', urmse_all)
"""
os.system('mv {} {}'.format('*npy',configs.outputs_path))
# -------------------------------------------------------------------------------------------------

























































# Figures------------------------------------------------------------------------------------------
path = configs.outputs_path
print('[HybridHydro] Start drawing')
# -------------------------------------------------------------------------------------------------
# FIG2. Mean value of R and uRMSE
# -------------------------------------------------------------------------------------------------
r_all = np.load(path+'r.npy')
r_gfs = r_all[0]
r_convlstm = r_all[1]
r_convlstm_mean = r_all[2]
r_convlstm_pb = r_all[3]
r_convlstm_pb_tf = r_all[4]
r_convlstm_pb_att_tf = r_all[5]
r_convlstm_pb_att_tf_mean = r_all[6]
r_smap = r_all[7]
urmse_all = np.load(path+'urmse.npy')
urmse_gfs = urmse_all[0]
urmse_convlstm = urmse_all[1]
urmse_convlstm_mean = urmse_all[2]
urmse_convlstm_pb = urmse_all[3]
urmse_convlstm_pb_tf = urmse_all[4]
urmse_convlstm_pb_att_tf = urmse_all[5]
urmse_convlstm_pb_att_tf_mean = urmse_all[6]
urmse_smap = urmse_all[7]

plt.figure(figsize=(12, 5))
ms=4
colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf',
          '#1a55FF']
t = np.arange(16)

ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
ax1.spines['left'].set_linewidth(1)
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['right'].set_linewidth(1)
ax1.spines['top'].set_linewidth(1)
plt.plot(np.nanmean(r_gfs, axis=0), marker='o', ms=ms, color=colors[1])
plt.plot(np.nanmean(r_convlstm, axis=0), marker='o', ms=ms, color=colors[2])
plt.plot(np.nanmean(r_convlstm_mean, axis=0), marker='o',ms=ms, color=colors[3])
plt.plot(np.nanmean(r_convlstm_pb, axis=0), marker='o', ms=ms, color=colors[5])
plt.plot(np.nanmean(r_convlstm_pb_tf, axis=0), marker='o',ms=ms, color=colors[4])
plt.plot(np.nanmean(r_convlstm_pb_att_tf, axis=0), marker='o',ms=ms, color=colors[-2])
plt.plot(np.nanmean(r_convlstm_pb_att_tf_mean, axis=0), marker='o',ms=ms,  color=colors[-1])
plt.plot(np.nanmean(r_smap, axis=0), linestyle='--', linewidth=2, color='black')
plt.xlabel('Forecast Time Scale (day)', fontsize=12)
plt.ylabel('Mean R', fontsize=12)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], \
    labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])
plt.xlim(0, 15)
plt.legend(['GFS', 'ConvLSTM', 'average', 'drive', 'condition', 'attention', 'ensemble', 'SMAP L4'], fancybox=False, shadow=False, fontsize=8)
plt.text(1, 0.54, 'a)', fontsize=15)

ax2 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)
ax2.spines['left'].set_linewidth(1)
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['right'].set_linewidth(1)
ax2.spines['top'].set_linewidth(1)
plt.plot(np.nanmean(urmse_gfs, axis=0), marker='o', ms=ms, color=colors[1])
plt.plot(np.nanmean(urmse_convlstm, axis=0), marker='o', ms=ms, color=colors[2])
plt.plot(np.nanmean(urmse_convlstm_mean, axis=0), marker='o',ms=ms, color=colors[3])
plt.plot(np.nanmean(urmse_convlstm_pb, axis=0), marker='o', ms=ms, color=colors[5])
plt.plot(np.nanmean(urmse_convlstm_pb_tf, axis=0), marker='o',ms=ms, color=colors[4])
plt.plot(np.nanmean(urmse_convlstm_pb_att_tf, axis=0), marker='o',ms=ms, color=colors[-2])
plt.plot(np.nanmean(urmse_convlstm_pb_att_tf_mean, axis=0), marker='o',ms=ms,  color=colors[-1])
plt.plot(np.nanmean(urmse_smap, axis=0), linestyle='--',linewidth=2,  color='black')
plt.xlabel('Forecast Time Scale (day)', fontsize=12)
plt.ylabel('Mean unbiased RMSE (${m^3/m^3}$)', fontsize=12)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], \
    labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])
plt.text(1, 0.0677, 'b)', fontsize=15)
plt.xlim(0, 15)
plt.subplots_adjust(wspace=0.3)
plt.savefig('figure2.pdf')
print('Figure 2 Completed!')
# -------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------
# FIG3. KDE of R
# -------------------------------------------------------------------------------------------------
plt.figure(figsize=(10,3))

plt.subplot(1, 3, 1)
sns.kdeplot(data=r_gfs[:, 0], shade=False, linewidth=1, color=colors[1])
sns.kdeplot(data=r_convlstm[:, 0], shade=False, linewidth=1, color=colors[2])
sns.kdeplot(data=r_convlstm_mean[:,0], shade=False, linewidth=1, color=colors[3])
sns.kdeplot(data=r_convlstm_pb[:,0], shade=False, linewidth=1, color=colors[5])
sns.kdeplot(data=r_convlstm_pb_tf[:,0], shade=False, linewidth=1, color=colors[4])
sns.kdeplot(data=r_convlstm_pb_att_tf[:,0], shade=False, linewidth=1, color=colors[-2])
sns.kdeplot(data=r_convlstm_pb_att_tf_mean[:,0], shade=False, linewidth=1, color=colors[-1])
plt.xlim(-1, 1)
plt.ylim(0, 2.5)
plt.xlabel('R', fontsize=10)
plt.legend(['GFS', 'ConvLSTM', 'average', 'drive', 'condition', 'attention', 'ensemble'], 
loc='upper left',
fancybox=False, shadow=False, fontsize=8)
plt.title('a) 1-day forecast',fontsize=12)

plt.subplot(1, 3, 2)
sns.kdeplot(data=r_gfs[:, 6], shade=False, linewidth=1, color=colors[1])
sns.kdeplot(data=r_convlstm[:, 6], shade=False, linewidth=1,color=colors[2])
sns.kdeplot(data=r_convlstm_mean[:,6], shade=False, linewidth=1,color=colors[3])
sns.kdeplot(data=r_convlstm_pb[:,6], shade=False,linewidth=1, color=colors[5])
sns.kdeplot(data=r_convlstm_pb_tf[:,6], shade=False,linewidth=1, color=colors[4])
sns.kdeplot(data=r_convlstm_pb_att_tf[:,6], shade=False,linewidth=1, color=colors[-2])
sns.kdeplot(data=r_convlstm_pb_att_tf_mean[:,6], shade=False, linewidth=1,color=colors[-1])
plt.yticks([])
plt.ylabel('')
plt.xlim(-1, 1)
plt.ylim(0, 2.5)
plt.xlabel('R', fontsize=10)
plt.title('b) 7-day forecast',fontsize=12)

plt.subplot(1, 3, 3)
sns.kdeplot(data=r_gfs[:, -1], shade=False, linewidth=1, color=colors[1])
sns.kdeplot(data=r_convlstm[:, -1], shade=False, linewidth=1, color=colors[2])
sns.kdeplot(data=r_convlstm_mean[:,-1], shade=False, linewidth=1, color=colors[3])
sns.kdeplot(data=r_convlstm_pb[:,-1], shade=False, linewidth=1, color=colors[5])
sns.kdeplot(data=r_convlstm_pb_tf[:,-1], shade=False, linewidth=1, color=colors[4])
sns.kdeplot(data=r_convlstm_pb_att_tf[:,-1], shade=False, linewidth=1, color=colors[-2])
sns.kdeplot(data=r_convlstm_pb_att_tf_mean[:,-1], shade=False, linewidth=1, color=colors[-1])
#sns.kdeplot(data=r_smap[:,-1], shade=False, linewidth=1, color='black')
plt.yticks([])
plt.ylabel('')
plt.xlim(-1, 1)
plt.ylim(0, 2.5)
plt.xlabel('R', fontsize=10)
plt.title('c) 16-day forecast',fontsize=12)
plt.savefig('figure3.pdf')
print('Figure 3 Completed!')
# -------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------------
# FIG4. spatial distribution of R
# -------------------------------------------------------------------------------------------------
x1,y1 = np.meshgrid(lon_gfs, lat_gfs)
x0,y0 = np.meshgrid(lon_smap, lat_smap)

# spatial pattern for 1, 7, 16 time scale forecast over different sites
plt.figure(figsize=(9, 10))
NS = 1.5
vmin=-0.2
vmax=0.2
cmap = 'bwr'
mdls = ['GFS', 'ConvLSTM', 'average','condition', 'attention', 'ensemble']
colors = category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, len(mdls)))
#----------------------------------------------------------------
ax = plt.subplot(6, 4, 5)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_gfs[:, 0]-r_convlstm_mean[:, 0], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)
plt.title('1-day', fontsize=12)
plt.ylabel('${\delta}$'+'(GFS)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 6)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_gfs[:, 6]-r_convlstm_mean[:, 6], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 7)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_gfs[:, -1]-r_convlstm_mean[:, -1], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 9)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[:, 0]-r_convlstm_mean[:, 0], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(ConvLSTM)', fontsize=10, rotation=90)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[:, 6]-r_convlstm_mean[:, 6], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 11)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[:, -1]-r_convlstm_mean[:, -1], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc1 = m.scatter(x,y, c=r_convlstm_mean[:, 0], marker='.',s=NS,vmin=0, vmax=1, cmap='rainbow')

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc1, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)
plt.ylabel('average', fontsize=10, rotation=90)
plt.title('1-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_mean[:, 6], marker='.',s=NS, vmin=0, vmax=1, cmap='rainbow')

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('7-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 3)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_mean[:, -1], marker='.',s=NS,vmin=0, vmax=1, cmap='rainbow')

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('16-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 13)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_tf[:, 0]-r_convlstm_mean[:, 0], marker='.',s=NS,vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(condition)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 14)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_tf[:, 6]-r_convlstm_mean[:, 6], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 15)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_tf[:, -1]-r_convlstm_mean[:, -1], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 17)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[:, 0]-r_convlstm_mean[:, 0], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(attention)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 18)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[:, 6]-r_convlstm_mean[:, 6], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 19)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[:, -1]-r_convlstm_mean[:, -1], marker='.',s=NS,vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 21)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf_mean[:, 0]-r_convlstm_mean[:, 0], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(ensemble)', fontsize=10, rotation=90)

axin = ax.inset_axes([0.1, -0.2, 0.5, 0.2])
axin.spines['left'].set_linewidth(0)
axin.spines['bottom'].set_linewidth(0)
axin.spines['right'].set_linewidth(0)
axin.spines['top'].set_linewidth(0)
axin.set_xticks([])
axin.set_yticks([])
axin.text(0, 0, '${\delta}$'+'(*) = '+'R(*)-R(SA-GC)')
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 22)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf_mean[:, 6]-r_convlstm_mean[:, 6], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 23)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf_mean[:, -1]-r_convlstm_mean[:, -1], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
plt.subplots_adjust(hspace=-0.27, wspace=0)
plt.savefig('figure4.pdf')
print('Figure 4 Completed!')
# -------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------
# FIG5. The impact of attention
# -------------------------------------------------------------------------------------------------
# get lat/lon AGME
AGME_auxiliary = np.load(configs.inputs_path+'valid_data/AGME_auxiliary.npy')
lat_agme = AGME_auxiliary[:,0]
lon_agme = AGME_auxiliary[:,1]
fig = plt.figure(figsize=(9, 10))
NS = 1.5
vmin=0#-0.2
vmax=1#0.2
cmap = 'rainbow'
vmin1=-0.2
vmax1=0.2
cmap1 = 'bwr'
mdls = ['GFS', 'ConvLSTM', 'Mean', 'Conditioned','Att-Conditioned', 'Att-TF-Mean']
colors = category_colors = plt.get_cmap('RdYlGn')(
    np.linspace(0.15, 0.85, len(mdls)))
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_gfs[:, 0], marker='o',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('GFS', rotation=90, fontsize=10)
plt.title('1-day', fontsize=12)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_gfs[:, 6], marker='o',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('7-day', fontsize=12)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 3)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_gfs[:, -1], marker='o',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('16-day', fontsize=12)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 5)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[:, 0], marker='o',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('ConvLSTM', rotation=90, fontsize=10)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 6)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[:, 6], marker='o',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 7)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[:, -1], marker='o',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 9)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[:, 0], marker='o',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('attention', rotation=90, fontsize=10)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[:, 6], marker='o',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 11)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[:, -1], marker='o',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 13)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[:, 0]-r_convlstm_pb_tf[:, 0], marker='o',s=NS, vmin=vmin1, vmax=vmax1, cmap=cmap1)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('Impact of attention', rotation=90, fontsize=10)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 14)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[:, 6]-r_convlstm_pb_tf[:, 6], marker='o',s=NS, vmin=vmin1, vmax=vmax1, cmap=cmap1)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 15)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[:, -1]-r_convlstm_pb_tf[:, -1], marker='o',s=NS, vmin=vmin1, vmax=vmax1, cmap=cmap1)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#-------------------------------------------------------------------------------------------------
plt.subplots_adjust(hspace=-0.27, wspace=0)
plt.savefig('figure5.pdf')
print('Figure 5 Completed!')
#-------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------
# FIG6. TCA KDE
# -------------------------------------------------------------------------------------------------
tc_all_bk = np.load(configs.outputs_path+'tc.npy')
tc_all = tc_all_bk
mask_nan = mask_in_china
mask_nan[mask_in_china==0] = np.nan

tc_gfs_bk = tc_all_bk[-1] + mask_nan
tc_convlstm_bk = tc_all_bk[0] + mask_nan
tc_convlstm_mean_bk = tc_all_bk[1] + mask_nan
tc_convlstm_pb_tf_bk = tc_all_bk[2] + mask_nan
tc_convlstm_pb_att_tf_bk = tc_all_bk[3] + mask_nan
tc_convlstm_pb_att_tf_mean_bk = tc_all_bk[4] + mask_nan
tc_gfs = tc_all[-1] + mask_nan
tc_convlstm = tc_all[0] + mask_nan
tc_convlstm_mean = tc_all[1] + mask_nan
tc_convlstm_pb_tf = tc_all[2] + mask_nan
tc_convlstm_pb_att_tf = tc_all[3] + mask_nan
tc_convlstm_pb_att_tf_mean = tc_all[4] + mask_nan

# Figure 
plt.figure(figsize=(9, 10))
cmap = 'gist_rainbow'
cmap = 'Spectral'
cmap = 'jet'
cmap = 'bwr'
vmin=-20
vmax=20
mdls = ['GFS', 'ConvLSTM', 'Mean', 'Conditioned','Att-Conditioned', 'Att-TF-Mean']
colors = category_colors = plt.get_cmap('RdYlGn')(
    np.linspace(0.15, 0.85, len(mdls)))
def diff_snr(x, y): return x-y
#----------------------------------------------------------------
ax = plt.subplot(6, 4, 5)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x, y, mask_in_china * diff_snr(tc_gfs[0], tc_convlstm_mean[0]),vmin=vmin, vmax=vmax, cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(GFS)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 6)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x, y, mask_in_china *diff_snr(tc_gfs[6], tc_convlstm_mean[6]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 7)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x, y, mask_in_china *diff_snr(tc_gfs[-1], tc_convlstm_mean[-1]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 9)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x,y,mask_in_china * diff_snr(tc_convlstm[0], tc_convlstm_mean[0]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(ConvLSTM)', fontsize=10, rotation=90)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china * diff_snr(tc_convlstm[6], tc_convlstm_mean[6]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 11)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm[-1], tc_convlstm_mean[-1]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection

m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_mean_bk[0] + mask_nan, vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('average', fontsize=10, rotation=90)
plt.title('1-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_mean_bk[6]+ mask_nan, vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('7-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 3)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_mean_bk[-1]+ mask_nan, vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('16-day', fontsize=12)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 13)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_tf[0], tc_convlstm_mean[0]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(condition)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 14)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china * diff_snr(tc_convlstm_pb_tf[6], tc_convlstm_mean[6]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 15)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_tf[-1], tc_convlstm_mean[-1]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 17)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_att_tf[0], tc_convlstm_mean[0]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(attention)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 18)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y,mask_in_china *diff_snr(tc_convlstm_pb_att_tf[6], tc_convlstm_mean[6]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 19)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y,mask_in_china *diff_snr(tc_convlstm_pb_att_tf[-1], tc_convlstm_mean[-1]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 21)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_att_tf_mean[0], tc_convlstm_mean[0]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(ensemble)', fontsize=10, rotation=90)
axin = ax.inset_axes([0.1, -0.2, 0.5, 0.2])
axin.spines['left'].set_linewidth(0)
axin.spines['bottom'].set_linewidth(0)
axin.spines['right'].set_linewidth(0)
axin.spines['top'].set_linewidth(0)
axin.set_xticks([])
axin.set_yticks([])
axin.text(0, -0.2, '${\delta}$'+'(*) = '+'SNR(*)-SNR(average)')
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 22)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_att_tf_mean[6], tc_convlstm_mean[6]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 23)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_att_tf_mean[-1], tc_convlstm_mean[-1]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
plt.subplots_adjust(hspace=-0.27, wspace=0)
plt.savefig('figure6.png', dpi=600)
print('Figure 6 Completed!')
#--------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
# Figure 7. SMAP ceiling
#--------------------------------------------------------------------------------------------------
tc_smap = np.load(configs.outputs_path+'tc_smap.npy')

# Figure 
plt.figure(figsize=(9, 10))
cmap = 'gist_rainbow'
cmap = 'Spectral'
cmap = 'bwr'
cmap = 'jet'
vmin=-20
vmax=20
tc_smap_bk = tc_smap
colors = category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, len(mdls)))
#----------------------------------------------------------------
ax = plt.subplot(6, 4, 20)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x, y, tc_smap + mask_nan,vmin=vmin, vmax=vmax, cmap=cmap)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('SMAP L4', fontsize=12)#, fontweight='bold')
#----------------------------------------------------------------
ax = plt.subplot(6, 4, 5)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_bk[0] + mask_nan, vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('ConvLSTM', fontsize=10, rotation=90)
plt.title('1-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 6)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_bk[6]+ mask_nan, vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('7-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 7)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_bk[-1]+ mask_nan, vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('16-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 9)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_pb_att_tf_mean_bk[0] + mask_nan, vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('ensemble', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_pb_att_tf_mean_bk[6]+ mask_nan, vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 11)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_pb_att_tf_mean_bk[-1], vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
plt.subplots_adjust(hspace=-0.27, wspace=0)
plt.savefig('figure7.png', dpi=600)
print('Figure 7 Completed!')
#--------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
# Fig8: Scatter plot of SWDI
#--------------------------------------------------------------------------------------------------
colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf',
          '#1a55FF']

gfs_tmp, agme_tmp, convlstm_tmp, conv_att_tf_mean_tmp, mean_tmp, tf_tmp, att_tf_tmp = \
    [], [], [], [], [], [], [] 
# generate drought array
for i in range(len(lat_agme)):
    lat_obs = lat_agme[i]
    lon_obs = lon_agme[i]
    index_lat = np.argmin(np.abs(lat_gfs-lat_obs))
    index_lon = np.argmin(np.abs(lon_gfs-lon_obs))
    gfs_tmp.append(swdi_w1_gfs[:, index_lat, index_lon])
    agme_tmp.append(swdi_w1_agme[:, i])

    index_lat = np.argmin(np.abs(lat_smap-lat_obs))
    index_lon = np.argmin(np.abs(lon_smap-lon_obs))
    convlstm_tmp.append(swdi_w1_convlstm[:,  index_lat, index_lon])
    conv_att_tf_mean_tmp.append(swdi_w1_att_tf_mean[:, index_lat, index_lon])
    mean_tmp.append(swdi_w1_mean[:, index_lat, index_lon])
    tf_tmp.append(swdi_w1_tf[:, index_lat, index_lon])
    att_tf_tmp.append(swdi_w1_att_tf[:, index_lat, index_lon])
agme_tmp = np.array(agme_tmp).reshape(-1,)
gfs_tmp = np.array(gfs_tmp).reshape(-1,)
convlstm_tmp = np.array(convlstm_tmp).reshape(-1,)
mean_tmp = np.array(mean_tmp).reshape(-1,)
tf_tmp = np.array(tf_tmp).reshape(-1,)
att_tf_tmp = np.array(att_tf_tmp).reshape(-1,)
conv_att_tf_mean_tmp = np.array(conv_att_tf_mean_tmp).reshape(-1,)

#fig
plt.figure(figsize=(8, 3))
plt.subplot(121)
sns.kdeplot(data=agme_tmp, color='black')
sns.kdeplot(data=gfs_tmp, color=colors[1])
sns.kdeplot(data=convlstm_tmp, color=colors[2])
sns.kdeplot(data=mean_tmp, color=colors[3])
sns.kdeplot(data=tf_tmp, color=colors[4])
sns.kdeplot(data=att_tf_tmp, color=colors[6])
sns.kdeplot(data=conv_att_tf_mean_tmp, color=colors[7])
plt.xlim(-20, 10)

plt.legend(['in-situ', 'GFS', 'ConvLSTM', 'average','condition', 'attention', 'ensemble'], fancybox=False, shadow=False, fontsize=6)
plt.title('a) Week-1 forecast', fontsize=10)
plt.xlabel('SWDI', fontsize=10)


gfs_tmp, agme_tmp, convlstm_tmp, conv_att_tf_mean_tmp, mean_tmp, tf_tmp, att_tf_tmp = \
    [], [], [], [], [], [], [] 
# generate drought array
for i in range(len(lat_agme)):
    lat_obs = lat_agme[i]
    lon_obs = lon_agme[i]
    index_lat = np.argmin(np.abs(lat_gfs-lat_obs))
    index_lon = np.argmin(np.abs(lon_gfs-lon_obs))
    gfs_tmp.append(swdi_w2_gfs[:, index_lat, index_lon])
    agme_tmp.append(swdi_w2_agme[:, i])

    index_lat = np.argmin(np.abs(lat_smap-lat_obs))
    index_lon = np.argmin(np.abs(lon_smap-lon_obs))
    convlstm_tmp.append(swdi_w2_convlstm[:,  index_lat, index_lon])
    conv_att_tf_mean_tmp.append(swdi_w2_att_tf_mean[:, index_lat, index_lon])
    mean_tmp.append(swdi_w2_mean[:, index_lat, index_lon])
    tf_tmp.append(swdi_w2_tf[:, index_lat, index_lon])
    att_tf_tmp.append(swdi_w2_att_tf[:, index_lat, index_lon])
agme_tmp = np.array(agme_tmp).reshape(-1,)
gfs_tmp = np.array(gfs_tmp).reshape(-1,)
convlstm_tmp = np.array(convlstm_tmp).reshape(-1,)
mean_tmp = np.array(mean_tmp).reshape(-1,)
tf_tmp = np.array(tf_tmp).reshape(-1,)
att_tf_tmp = np.array(att_tf_tmp).reshape(-1,)
conv_att_tf_mean_tmp = np.array(conv_att_tf_mean_tmp).reshape(-1,)

plt.subplot(122)
sns.kdeplot(data=agme_tmp, color='black')
sns.kdeplot(data=gfs_tmp, color=colors[1])
sns.kdeplot(data=convlstm_tmp, color=colors[2])
sns.kdeplot(data=mean_tmp, color=colors[3])
sns.kdeplot(data=tf_tmp, color=colors[4])
sns.kdeplot(data=att_tf_tmp, color=colors[6])
sns.kdeplot(data=conv_att_tf_mean_tmp, color=colors[7])
plt.title('b) Week-2 forecast', fontsize=10)
plt.yticks([])
plt.ylabel('')
plt.xlabel('SWDI', fontsize=10)
plt.xlim(-20, 10)
plt.savefig('figure8.pdf')
print('Figure 8 Completed!')
# -------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------
# Table 1. drought events probability
#--------------------------------------------------------------------------------------------------
"""
lat_agme = AGME_auxiliary[:,0]
lon_agme = AGME_auxiliary[:,1]

def cal_extreme_events_metrics(y_true, y_pred):
    
    hits = len(np.where((agme_dry>0) & (gfs_dry>0))[0])
    correct_negatives = len(np.where((agme_dry==0) & (gfs_dry==0))[0])
    total = len(agme_dry)
    misses = len(np.where((agme_dry>0) & (gfs_dry==0))[0])
    
    hits = len(np.where(((y_true==1) & (y_pred==1)) | ((y_true==2) & (y_pred==2)))[0]) # accurate
    correct_negatives = len(np.where((y_true==0) & (y_pred==0))[0]) # accurate non 
    total = len(y_true)
    misses = len(np.where(((y_true==1) & (y_pred!=1)) | ((y_true==2) & (y_pred!=2)))[0])

    acc = (hits + correct_negatives)/total
    pod = hits/(hits+misses)
    
    return acc, pod
    
#agme: 342, 16, 1164; 

acc_gfs = []
pod_gfs = []
acc_conv = []
pod_conv = []
acc_atm = []
pod_atm = []
acc_mean = []
pod_mean = []
acc_conv_tf = []
pod_conv_tf = []
acc_conv_att_tf = []
pod_conv_att_tf = []
kg_index = []


lat = np.arange(90, -90, -0.5)
lon = np.arange(-180, 180, 0.5)
acc_all_w1 = np.full((6, 5), np.nan)
pod_all_w1 = np.full((6, 5), np.nan)
acc_mean_w1 = np.full((6, ), np.nan)
pod_mean_w1 = np.full((6, ), np.nan)

# generate drought array
for i in range(len(lat_agme)):
    print(i)

    lat_obs = lat_agme[i]
    lon_obs = lon_agme[i]

    #
    index_lat = np.argmin(np.abs(lat_gfs-lat_obs))
    index_lon = np.argmin(np.abs(lon_gfs-lon_obs))

    gfs_tmp = swdi_w1_gfs[:,  index_lat, index_lon]
    agme_tmp = swdi_w1_agme[:, i]

    agme_dry = np.zeros_like(gfs_tmp)
    gfs_dry = np.zeros_like(gfs_tmp)

    agme_dry[(agme_tmp<0) & (agme_tmp>-5)] = 1
    agme_dry[(agme_tmp<=-5)] = 2
    gfs_dry[(gfs_tmp<0) & (gfs_tmp>-5)] = 1
    gfs_dry[(gfs_tmp<=-5)] = 2

    #
    index_lat = np.argmin(np.abs(lat_smap-lat_obs))
    index_lon = np.argmin(np.abs(lon_smap-lon_obs))

    convlstm_tmp = swdi_w1_convlstm[:,  index_lat, index_lon]
    conv_att_tf_mean_tmp = swdi_w1_att_tf_mean[:, index_lat, index_lon]
    mean_tmp = swdi_w1_mean[:, index_lat, index_lon]
    tf_tmp = swdi_w1_tf[:, index_lat, index_lon]
    att_tf_tmp = swdi_w1_att_tf[:, index_lat, index_lon]
    

    convlstm_dry = np.zeros_like(gfs_tmp)
    conv_att_tf_mean_dry = np.zeros_like(gfs_tmp)
    mean_dry = np.zeros_like(gfs_tmp)
    conv_tf_dry = np.zeros_like(gfs_tmp)
    conv_att_tf_dry = np.zeros_like(gfs_tmp)

    convlstm_dry[(convlstm_tmp<0) & (convlstm_tmp>-5)] = 1
    convlstm_dry[(convlstm_tmp<=-5)] = 2
    conv_att_tf_mean_dry[(conv_att_tf_mean_tmp<0) & (conv_att_tf_mean_tmp>-5)] = 1
    conv_att_tf_mean_dry[(conv_att_tf_mean_tmp<=-5)] = 2
    mean_dry[(mean_tmp<0) & (mean_tmp>-5)] = 1
    mean_dry[(mean_tmp<=-5)] = 2
    conv_tf_dry[(tf_tmp<0) & (tf_tmp>-5)] = 1
    conv_tf_dry[(tf_tmp<=-5)] = 2
    conv_att_tf_dry[(att_tf_tmp<0) & (att_tf_tmp>-5)] = 1
    conv_att_tf_dry[(att_tf_tmp<=-5)] = 2

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, gfs_dry)
    acc_gfs.append(acc)
    pod_gfs.append(pod)

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, convlstm_dry)
    acc_conv.append(acc)
    pod_conv.append(pod)

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, mean_dry)
    acc_mean.append(acc)
    pod_mean.append(pod)

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, conv_tf_dry)
    acc_conv_tf.append(acc)
    pod_conv_tf.append(pod)

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, conv_att_tf_dry)
    acc_conv_att_tf.append(acc)
    pod_conv_att_tf.append(pod)

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, conv_att_tf_mean_dry)
    acc_atm.append(acc)
    pod_atm.append(pod)

    index_lat = np.argmin(np.abs(lat-lat_obs))
    index_lon = np.argmin(np.abs(lon-lon_obs))
    kg_index.append(kg_idx[index_lat, index_lon])



kg_index = np.array(kg_index)
acc_gfs = np.array(acc_gfs)
acc_conv = np.array(acc_conv)
acc_atm = np.array(acc_atm)
acc_mean = np.array(acc_mean)
acc_conv_tf = np.array(acc_conv_tf)
acc_conv_att_tf = np.array(acc_conv_att_tf)

pod_gfs = np.array(pod_gfs)
pod_conv = np.array(pod_conv)
pod_atm = np.array(pod_atm)
pod_mean = np.array(pod_mean)
pod_conv_tf = np.array(pod_conv_tf)
pod_conv_att_tf = np.array(pod_conv_att_tf)

for i in range(acc_all_w1.shape[-1]):
    acc_all_w1[:, i] = [np.nanmean(acc_gfs[np.where(kg_index==i+1)[0]]), \
            np.nanmean(acc_conv[np.where(kg_index==i+1)[0]]), \
                np.nanmean(acc_mean[np.where(kg_index==i+1)[0]]), \
                    np.nanmean(acc_conv_tf[np.where(kg_index==i+1)[0]]), \
                        np.nanmean(acc_conv_att_tf[np.where(kg_index==i+1)[0]]),\
            np.nanmean(acc_atm[np.where(kg_index==i+1)[0]])]

    pod_all_w1[:, i] = [
            np.nanmean(pod_gfs[np.where(kg_index==i+1)[0]]), \
            np.nanmean(pod_conv[np.where(kg_index==i+1)[0]]), \
            np.nanmean(pod_mean[np.where(kg_index==i+1)[0]]), \
            np.nanmean(pod_conv_tf[np.where(kg_index==i+1)[0]]), \
            np.nanmean(pod_conv_att_tf[np.where(kg_index==i+1)[0]]),\
            np.nanmean(pod_atm[np.where(kg_index==i+1)[0]])] 

acc_mean_w1 = [np.nanmean(acc_gfs), \
            np.nanmean(acc_conv), \
                np.nanmean(acc_mean), \
                    np.nanmean(acc_conv_tf), \
                        np.nanmean(acc_conv_att_tf),\
            np.nanmean(acc_atm)]

pod_mean_w1 = [
            np.nanmean(pod_gfs), \
            np.nanmean(pod_conv), \
            np.nanmean(pod_mean), \
            np.nanmean(pod_conv_tf), \
            np.nanmean(pod_conv_att_tf),\
            np.nanmean(pod_atm)]

acc_gfs = []
pod_gfs = []
acc_conv = []
pod_conv = []
acc_atm = []
pod_atm = []
acc_mean = []
pod_mean = []
acc_conv_tf = []
pod_conv_tf = []
acc_conv_att_tf = []
pod_conv_att_tf = []
kg_index = []


lat = np.arange(90, -90, -0.5)
lon = np.arange(-180, 180, 0.5)
acc_all_w2 = np.full((6, 5), np.nan)
pod_all_w2 = np.full((6, 5), np.nan)
acc_mean_w2 = np.full((6, ), np.nan)
pod_mean_w2 = np.full((6, ), np.nan)

# generate drought array
for i in range(len(lat_agme)):
    print(i)

    lat_obs = lat_agme[i]
    lon_obs = lon_agme[i]

    #
    index_lat = np.argmin(np.abs(lat_gfs-lat_obs))
    index_lon = np.argmin(np.abs(lon_gfs-lon_obs))

    gfs_tmp = swdi_w2_gfs[:,  index_lat, index_lon]
    agme_tmp = swdi_w2_agme[:, i]

    agme_dry = np.zeros_like(gfs_tmp)
    gfs_dry = np.zeros_like(gfs_tmp)

    agme_dry[(agme_tmp<0) & (agme_tmp>-5)] = 1
    agme_dry[(agme_tmp<=-5)] = 2
    gfs_dry[(gfs_tmp<0) & (gfs_tmp>-5)] = 1
    gfs_dry[(gfs_tmp<=-5)] = 2

    #
    index_lat = np.argmin(np.abs(lat_smap-lat_obs))
    index_lon = np.argmin(np.abs(lon_smap-lon_obs))

    convlstm_tmp = swdi_w2_convlstm[:,  index_lat, index_lon]
    conv_att_tf_mean_tmp = swdi_w2_att_tf_mean[:, index_lat, index_lon]
    mean_tmp = swdi_w2_mean[:, index_lat, index_lon]
    tf_tmp = swdi_w2_tf[:, index_lat, index_lon]
    att_tf_tmp = swdi_w2_att_tf[:, index_lat, index_lon]
    

    convlstm_dry = np.zeros_like(gfs_tmp)
    conv_att_tf_mean_dry = np.zeros_like(gfs_tmp)
    mean_dry = np.zeros_like(gfs_tmp)
    conv_tf_dry = np.zeros_like(gfs_tmp)
    conv_att_tf_dry = np.zeros_like(gfs_tmp)

    convlstm_dry[(convlstm_tmp<0) & (convlstm_tmp>-5)] = 1
    convlstm_dry[(convlstm_tmp<=-5)] = 2
    conv_att_tf_mean_dry[(conv_att_tf_mean_tmp<0) & (conv_att_tf_mean_tmp>-5)] = 1
    conv_att_tf_mean_dry[(conv_att_tf_mean_tmp<=-5)] = 2
    mean_dry[(mean_tmp<0) & (mean_tmp>-5)] = 1
    mean_dry[(mean_tmp<=-5)] = 2
    conv_tf_dry[(tf_tmp<0) & (tf_tmp>-5)] = 1
    conv_tf_dry[(tf_tmp<=-5)] = 2
    conv_att_tf_dry[(att_tf_tmp<0) & (att_tf_tmp>-5)] = 1
    conv_att_tf_dry[(att_tf_tmp<=-5)] = 2

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, gfs_dry)
    acc_gfs.append(acc)
    pod_gfs.append(pod)

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, convlstm_dry)
    acc_conv.append(acc)
    pod_conv.append(pod)

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, mean_dry)
    acc_mean.append(acc)
    pod_mean.append(pod)

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, conv_tf_dry)
    acc_conv_tf.append(acc)
    pod_conv_tf.append(pod)

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, conv_att_tf_dry)
    acc_conv_att_tf.append(acc)
    pod_conv_att_tf.append(pod)

    # cal A and POD for each points in differnet dry wet condition
    acc, pod = cal_extreme_events_metrics(agme_dry, conv_att_tf_mean_dry)
    acc_atm.append(acc)
    pod_atm.append(pod)

    index_lat = np.argmin(np.abs(lat-lat_obs))
    index_lon = np.argmin(np.abs(lon-lon_obs))
    kg_index.append(kg_idx[index_lat, index_lon])


kg_index = np.array(kg_index)
acc_gfs = np.array(acc_gfs)
acc_conv = np.array(acc_conv)
acc_atm = np.array(acc_atm)
acc_mean = np.array(acc_mean)
acc_conv_tf = np.array(acc_conv_tf)
acc_conv_att_tf = np.array(acc_conv_att_tf)

pod_gfs = np.array(pod_gfs)
pod_conv = np.array(pod_conv)
pod_atm = np.array(pod_atm)
pod_mean = np.array(pod_mean)
pod_conv_tf = np.array(pod_conv_tf)
pod_conv_att_tf = np.array(pod_conv_att_tf)

for i in range(acc_all_w2.shape[-1]):
    acc_all_w2[:, i] = [np.nanmean(acc_gfs[np.where(kg_index==i+1)[0]]), \
            np.nanmean(acc_conv[np.where(kg_index==i+1)[0]]), \
                np.nanmean(acc_mean[np.where(kg_index==i+1)[0]]), \
                    np.nanmean(acc_conv_tf[np.where(kg_index==i+1)[0]]), \
                        np.nanmean(acc_conv_att_tf[np.where(kg_index==i+1)[0]]),\
            np.nanmean(acc_atm[np.where(kg_index==i+1)[0]])]

    pod_all_w2[:, i] = [
            np.nanmean(pod_gfs[np.where(kg_index==i+1)[0]]), \
            np.nanmean(pod_conv[np.where(kg_index==i+1)[0]]), \
            np.nanmean(pod_mean[np.where(kg_index==i+1)[0]]), \
            np.nanmean(pod_conv_tf[np.where(kg_index==i+1)[0]]), \
            np.nanmean(pod_conv_att_tf[np.where(kg_index==i+1)[0]]),\
            np.nanmean(pod_atm[np.where(kg_index==i+1)[0]])]


acc_mean_w2 = [np.nanmean(acc_gfs), \
            np.nanmean(acc_conv), \
                np.nanmean(acc_mean), \
                    np.nanmean(acc_conv_tf), \
                        np.nanmean(acc_conv_att_tf),\
            np.nanmean(acc_atm)]

pod_mean_w2 = [
            np.nanmean(pod_gfs), \
            np.nanmean(pod_conv), \
            np.nanmean(pod_mean), \
            np.nanmean(pod_conv_tf), \
            np.nanmean(pod_conv_att_tf),\
            np.nanmean(pod_atm)]

print(acc_all_w1)
print(acc_all_w2)
print(pod_all_w1)
print(pod_all_w2)

print(acc_mean_w1)
print(acc_mean_w2)
print(pod_mean_w1)
print(pod_mean_w2)

print(kg_index.shape)
np.save('kg_index.npy', kg_index)

print(len(np.where(kg_index==1)[0]))
print(len(np.where(kg_index==2)[0]))
print(len(np.where(kg_index==3)[0]))
print(len(np.where(kg_index==4)[0]))
print(len(np.where(kg_index==5)[0]))
print('Table 1 Completed!')
"""
#--------------------------------------------------------------------------------------------------



"""
# -------------------------------------------------------------------------------------------------
# FIGS1. Spatial distribution of in-situ observation
# -------------------------------------------------------------------------------------------------
# get lat/lon AGME
AGME_auxiliary = np.load(path + 'AGME_auxiliary.npy')
lat_agme = AGME_auxiliary[:,0]
lon_agme = AGME_auxiliary[:,1]

colors = (#'#FF1493', 
            '#FFC0CB',
            '#20B2AA', '#32CD32', '#7CFC00',
            '#00FFFF')#,  '#00BFFF', '#0000FF')
lat = np.arange(90, -90, -0.5)
lon = np.arange(-180, 180, 0.5)

x2, y2 = np.meshgrid(lon, lat)

plt.figure()
ax = plt.subplot(111)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(x2, y2)
levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
sc = m.contourf(x, y, kg_idx, colors=colors, levels=levels, vmin=1, vmax=5)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)

# create proxy artists to make legend
proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
            for pc in sc.collections]
title = ['Tropical', 'Arid', 'Temperate', 'Cold', 'Polar']
plt.legend(proxy, title, loc='upper left', bbox_to_anchor=(0, 1))
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c='gray', marker='.',s=10)
plt.savefig('figureS1.pdf')
print('Figure S1 Completed!')
#--------------------------------------------------------------------------------------------------
"""



"""
# -------------------------------------------------------------------------------------------------
# FIGS2. The impact of conditioned
# -------------------------------------------------------------------------------------------------
fig = plt.figure()
NS = 20
vmin=-0.2
vmax=0.2
cmap = 'bwr'
colors = category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, len(mdls)))
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(1, 2, 1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[:, 0]-r_gfs[:, 0], marker='o',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('a) R(ConvLSTM)-R(GFS)', fontsize=12)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(1, 2, 2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_tf[:, 0]-r_convlstm[:, 0], marker='o',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('b) Impact of condition', fontsize=12)
#-------------------------------------------------------------------------------------------------
plt.savefig('figureS2.pdf')
plt.subplots_adjust(hspace=0)
print('Figure S2 Completed!')
#-------------------------------------------------------------------------------------------------
"""



"""
#-------------------------------------------------------------------------------------------------
# Figure S3: KDE of SNR
#-------------------------------------------------------------------------------------------------
colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf',
          '#1a55FF']

plt.figure(figsize=(10,3))
plt.subplot(1, 3, 1)
sns.kdeplot(data=tc_gfs[0].reshape(-1), shade=False, linewidth=1, color=colors[1])
sns.kdeplot(data=tc_convlstm[0].reshape(-1), shade=False, linewidth=1, color=colors[2])
sns.kdeplot(data=tc_convlstm_mean[0].reshape(-1), shade=False, linewidth=1, color=colors[3])
sns.kdeplot(data=tc_convlstm_pb_tf[0].reshape(-1), shade=False, linewidth=1, color=colors[4])
sns.kdeplot(data=tc_convlstm_pb_att_tf[0].reshape(-1), shade=False, linewidth=1, color=colors[-2])
sns.kdeplot(data=tc_convlstm_pb_att_tf_mean[0].reshape(-1), shade=False, linewidth=1, color=colors[-1])
plt.xlim(-20, 20)
plt.ylim(0, 0.1)
plt.vlines(0, 0, 0.1, linestyles='-.', color='black')
plt.xlabel('SNR')
plt.legend(['GFS', 'ConvLSTM', 'average', 'condition', 'attention', 'ensemble'], 
loc='upper left',
fancybox=False, shadow=False, fontsize=8)
plt.title('a) 1-day forecast',fontsize=10)

plt.subplot(1, 3, 2)
sns.kdeplot(data=tc_gfs[6].reshape(-1), shade=False, linewidth=1, color=colors[1])
sns.kdeplot(data=tc_convlstm[6].reshape(-1), shade=False, linewidth=1,color=colors[2])
sns.kdeplot(data=tc_convlstm_mean[6].reshape(-1), shade=False, linewidth=1,color=colors[3])
sns.kdeplot(data=tc_convlstm_pb_tf[6].reshape(-1), shade=False,linewidth=1, color=colors[4])
sns.kdeplot(data=tc_convlstm_pb_att_tf[6].reshape(-1), shade=False,linewidth=1, color=colors[-2])
sns.kdeplot(data=tc_convlstm_pb_att_tf_mean[6].reshape(-1), shade=False, linewidth=1,color=colors[-1])
plt.yticks([])
plt.ylabel('')
plt.xlim(-20, 20)
plt.ylim(0, 0.1)
plt.vlines(0, 0, 0.1, linestyles='-.', color='black')
plt.xlabel('SNR')
plt.title('b) 7-day forecast',fontsize=10)

plt.subplot(1, 3, 3)
sns.kdeplot(data=tc_gfs[-1].reshape(-1), shade=False, linewidth=1, color=colors[1])
sns.kdeplot(data=tc_convlstm[-1].reshape(-1), shade=False, linewidth=1, color=colors[2])
sns.kdeplot(data=tc_convlstm_mean[-1].reshape(-1), shade=False, linewidth=1, color=colors[3])
sns.kdeplot(data=tc_convlstm_pb_tf[-1].reshape(-1), shade=False, linewidth=1, color=colors[4])
sns.kdeplot(data=tc_convlstm_pb_att_tf[-1].reshape(-1), shade=False, linewidth=1, color=colors[-2])
sns.kdeplot(data=tc_convlstm_pb_att_tf_mean[-1].reshape(-1), shade=False, linewidth=1, color=colors[-1])
plt.yticks([])
plt.ylabel('')
plt.xlim(-20, 20)
plt.ylim(0, 0.1)
plt.vlines(0, 0, 0.1, linestyles='-.', color='black')
plt.xlabel('SNR')
plt.title('c) 16-day forecast',fontsize=10)
plt.savefig('figureS3.pdf')
print('Figure S3 Completed!')
# -------------------------------------------------------------------------------------------------
"""



"""
# -------------------------------------------------------------------------------------------------
# FIGS5. dry to wet
# -------------------------------------------------------------------------------------------------

# load agme observation
AGME_obs = np.load(path + 'AGME_swvl1_2018.npy')/100
print("amge observation shape is {}".format(AGME_obs.shape))
AGME_auxiliary = np.load(path + 'AGME_auxiliary.npy')
lat_agme = AGME_auxiliary[:,0]
lon_agme = AGME_auxiliary[:,1]

r_gfs = r_all[0]
r_convlstm = r_all[1]
r_convlstm_mean = r_all[2]
r_convlstm_pb = r_all[3]
r_convlstm_pb_tf = r_all[4]
r_convlstm_pb_att_tf = r_all[6]
r_convlstm_pb_att_tf_mean = r_all[7]
r_smap = r_all[8]
r_all_mean = np.nanmean(r_all, axis=-1)

a = r_all_mean[0:2]
b = r_all_mean[-2:-1]
c = np.concatenate([a, b], axis=0)
rank1_all = np.full((r_gfs.shape[0],), np.nan)

# rank for gfs, convlstm, last model.
for i in range(r_gfs.shape[0]):
    if np.isnan(c[:, i]).any():
        pass
    else:
        index = np.argsort(c[:, i])
        rank1_all[i] = index[-1]

# rank1 percent
m = np.nanmean(AGME_obs, axis=0)
num = []
percent = []
for i in range(10):
    idx = np.argwhere((m>np.nanquantile(m, i/10)) & (m<np.nanquantile(m, (i+1)/10)))
    num.append(len(idx))

    a = rank1_all[idx]
    b = a[~np.isnan(a)]
    percent.append(len(b[b==2])/(len(b)))
    #plt.plot(np.nanmean(a,  axis=(1, 2)))


plt.figure(figsize=(10, 10))

# -------------------------------------------------------------------------------------------------
#plt.subplot2grid((7, 4), (0, 0), rowspan=1, colspan=2)
#plt.bar(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], \
#    height=percent, color='gray')
#plt.xticks([])
#plt.ylim(0, 1)
#plt.yticks([0., 0.5, 0.75], ['0%', '50%', '75%'])
#plt.text(-0.55, 0.75, 'a', fontweight='bold', fontsize=12)
#plt.ylabel('Rank 1\nPercentage')
# -------------------------------------------------------------------------------------------------
plt.subplot2grid((7, 4), (1, 0), rowspan=2, colspan=2)
d_gfs=[]
d_convlstm=[]
d_conv_tf_att=[]

y1_gfs = []
y1_convlstm=[]
y1_conv_tf_att=[]

y2_gfs = []
y2_convlstm=[]
y2_conv_tf_att=[]

for i in range(10):
    idx = np.argwhere((m>np.nanquantile(m, i/10)) & (m<np.nanquantile(m, (i+1)/10)))
    d_gfs.append(np.nanmean(r_gfs[idx, 0]))
    d_convlstm.append(np.nanmean(r_convlstm[idx, 0]))
    d_conv_tf_att.append(np.nanmean(r_convlstm_pb_att_tf_mean[idx, 0]))

    y1_gfs.append(np.nanquantile(r_gfs[idx, 0], 0.25))
    y1_convlstm.append(np.nanquantile(r_convlstm[idx, 0], 0.25))
    y1_conv_tf_att.append(np.nanquantile(r_convlstm_pb_att_tf_mean[idx, 0], 0.25))

    y2_gfs.append(np.nanquantile(r_gfs[idx, 0], 0.75))
    y2_convlstm.append(np.nanquantile(r_convlstm[idx, 0], 0.75))
    y2_conv_tf_att.append(np.nanquantile(r_convlstm_pb_att_tf_mean[idx, 0], 0.75))

alpha=0.3

plt.plot(d_gfs, color='red')
plt.fill_between(x=np.arange(10), y1=y1_gfs, y2=y2_gfs, color='mistyrose', alpha=alpha)
plt.plot(d_convlstm, color='blue')
plt.fill_between(x=np.arange(10), y1=y1_convlstm, y2=y2_convlstm, color='lightblue',alpha=alpha)
plt.plot(d_conv_tf_att, linestyle='--',color='orange')
plt.fill_between(x=np.arange(10), y1=y1_conv_tf_att, y2=y2_conv_tf_att, color='bisque',alpha=alpha)

plt.xlim(0, 9)
plt.ylim(-0.2, 0.8)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    ['', '', '', '', '', '', '', '', '', ''])
plt.legend(['GFS', 'ConvLSTM', 'Ens-all'], loc='lower right', fontsize=7)
plt.text(0.3, 0.7,  'a', fontweight='bold', fontsize=12)
plt.ylabel('1-day\nR', fontsize=12)
# -------------------------------------------------------------------------------------------------
plt.subplot2grid((7, 4), (3, 0), rowspan=2, colspan=2)
d_gfs=[]
d_convlstm=[]
d_conv_tf_att=[]

y1_gfs = []
y1_convlstm=[]
y1_conv_tf_att=[]

y2_gfs = []
y2_convlstm=[]
y2_conv_tf_att=[]

for i in range(10):
    idx = np.argwhere((m>np.nanquantile(m, i/10)) & (m<np.nanquantile(m, (i+1)/10)))
    d_gfs.append(np.nanmean(r_gfs[idx, 6]))
    d_convlstm.append(np.nanmean(r_convlstm[idx, 6]))
    d_conv_tf_att.append(np.nanmean(r_convlstm_pb_att_tf_mean[idx, 6]))

    y1_gfs.append(np.nanquantile(r_gfs[idx, 6], 0.25))
    y1_convlstm.append(np.nanquantile(r_convlstm[idx, 6], 0.25))
    y1_conv_tf_att.append(np.nanquantile(r_convlstm_pb_att_tf_mean[idx, 6], 0.25))

    y2_gfs.append(np.nanquantile(r_gfs[idx, 6], 0.75))
    y2_convlstm.append(np.nanquantile(r_convlstm[idx, 6], 0.75))
    y2_conv_tf_att.append(np.nanquantile(r_convlstm_pb_att_tf_mean[idx, 6], 0.75))

alpha=0.3

plt.plot(d_gfs, color='red')
plt.fill_between(x=np.arange(10), y1=y1_gfs, y2=y2_gfs, color='mistyrose', alpha=alpha)
plt.plot(d_convlstm, color='blue')
plt.fill_between(x=np.arange(10), y1=y1_convlstm, y2=y2_convlstm, color='lightblue',alpha=alpha)
plt.plot(d_conv_tf_att, linestyle='--',color='orange')
plt.fill_between(x=np.arange(10), y1=y1_conv_tf_att, y2=y2_conv_tf_att, color='bisque',alpha=alpha)
plt.xlim(0, 9)
plt.ylim(-0.2, 0.8)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    ['', '', '', '', '', '', '', '', '', ''])
plt.text(0.3, 0.7,  'c', fontweight='bold', fontsize=12)
plt.ylabel('7-day\nR', fontsize=12)
# -------------------------------------------------------------------------------------------------
plt.subplot2grid((7, 4), (5, 0), rowspan=2, colspan=2)
d_gfs=[]
d_convlstm=[]
d_conv_tf_att=[]

y1_gfs = []
y1_convlstm=[]
y1_conv_tf_att=[]

y2_gfs = []
y2_convlstm=[]
y2_conv_tf_att=[]

for i in range(10):
    idx = np.argwhere((m>np.nanquantile(m, i/10)) & (m<np.nanquantile(m, (i+1)/10)))
    d_gfs.append(np.nanmean(r_gfs[idx, -1]))
    d_convlstm.append(np.nanmean(r_convlstm[idx, -1]))
    d_conv_tf_att.append(np.nanmean(r_convlstm_pb_att_tf_mean[idx, -1]))

    y1_gfs.append(np.nanquantile(r_gfs[idx, -1], 0.25))
    y1_convlstm.append(np.nanquantile(r_convlstm[idx, -1], 0.25))
    y1_conv_tf_att.append(np.nanquantile(r_convlstm_pb_att_tf_mean[idx, -1], 0.25))

    y2_gfs.append(np.nanquantile(r_gfs[idx, -1], 0.75))
    y2_convlstm.append(np.nanquantile(r_convlstm[idx, -1], 0.75))
    y2_conv_tf_att.append(np.nanquantile(r_convlstm_pb_att_tf_mean[idx, -1], 0.75))

alpha=0.3

plt.plot(d_gfs, color='red')
plt.fill_between(x=np.arange(10), y1=y1_gfs, y2=y2_gfs, color='mistyrose', alpha=alpha)
plt.plot(d_convlstm, color='blue')
plt.fill_between(x=np.arange(10), y1=y1_convlstm, y2=y2_convlstm, color='lightblue',alpha=alpha)
plt.plot(d_conv_tf_att, linestyle='--',color='orange')
plt.fill_between(x=np.arange(10), y1=y1_conv_tf_att, y2=y2_conv_tf_att, color='bisque',alpha=alpha)
plt.xlim(0, 9)
plt.ylim(-0.2, 0.8)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',\
         '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'], rotation=45)
plt.text(0.3, 0.7,  'e', fontweight='bold', fontsize=12)
plt.ylabel('16-day\nR', fontsize=12)
# -------------------------------------------------------------------------------------------------
urmse_all = np.load('urmse_all.npy')
urmse_gfs = urmse_all[0]
urmse_convlstm = urmse_all[1]
urmse_convlstm_mean = urmse_all[2]
urmse_convlstm_pb = urmse_all[3]
urmse_convlstm_pb_tf = urmse_all[4]
urmse_convlstm_pb_tf_mean = urmse_all[5]
urmse_convlstm_pb_att_tf = urmse_all[6]
urmse_convlstm_pb_att_tf_mean = urmse_all[7]
urmse_smap = urmse_all[8]


urmse_all_mean = np.nanmean(urmse_all, axis=-1)
a = urmse_all_mean[0:2]
b = urmse_all_mean[-2:-1]
c = np.concatenate([a, b], axis=0)

rank1_all = np.full((r_gfs.shape[0],), np.nan)

for i in range(r_gfs.shape[0]):
    if np.isnan(c[:, i]).any():
        pass
    else:
        index = np.argsort(c[:, i])
        #print(index)
        #print(c[index,i])
        rank1_all[i] = index[0]

m = np.nanmean(AGME_obs, axis=0)
num = []
percent = []
for i in range(10):
    idx = np.argwhere((m>np.nanquantile(m, i/10)) & (m<np.nanquantile(m, (i+1)/10)))
    num.append(len(idx))
    a = rank1_all[idx]
    b = a[~np.isnan(a)]
    percent.append(len(b[b==2])/(len(b)))
    #plt.plot(np.nanmean(a,  axis=(1, 2)))



#plt.subplot2grid((7, 4), (0, 2), rowspan=1, colspan=2)
#plt.bar(['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], \
#    height=percent, color='gray')
#plt.xticks([])
#plt.ylim(0, 1)
#plt.yticks([0., 0.5, 0.75], ['0%', '50%', '75%'])
#plt.text(-0.55, 0.75, 'b', fontweight='bold', fontsize=12)
# -------------------------------------------------------------------------------------------------
plt.subplot2grid((7, 4), (1, 2), rowspan=2, colspan=2)
d_gfs=[]
d_convlstm=[]
d_conv_tf_att=[]

y1_gfs = []
y1_convlstm=[]
y1_conv_tf_att=[]

y2_gfs = []
y2_convlstm=[]
y2_conv_tf_att=[]

for i in range(10):
    idx = np.argwhere((m>np.nanquantile(m, i/10)) & (m<np.nanquantile(m, (i+1)/10)))
    d_gfs.append(np.nanmean(urmse_gfs[idx, 0]))
    d_convlstm.append(np.nanmean(urmse_convlstm[idx, 0]))
    d_conv_tf_att.append(np.nanmean(urmse_convlstm_pb_att_tf_mean[idx, 0]))

    y1_gfs.append(np.nanquantile(urmse_gfs[idx, 0], 0.25))
    y1_convlstm.append(np.nanquantile(urmse_convlstm[idx, 0], 0.25))
    y1_conv_tf_att.append(np.nanquantile(urmse_convlstm_pb_att_tf_mean[idx, 0], 0.25))

    y2_gfs.append(np.nanquantile(urmse_gfs[idx, 0], 0.75))
    y2_convlstm.append(np.nanquantile(urmse_convlstm[idx, 0], 0.75))
    y2_conv_tf_att.append(np.nanquantile(urmse_convlstm_pb_att_tf_mean[idx, 0], 0.75))

alpha=0.3

plt.plot(d_gfs, color='red')
plt.fill_between(x=np.arange(10), y1=y1_gfs, y2=y2_gfs, color='mistyrose', alpha=alpha)
plt.plot(d_convlstm, color='blue')
plt.fill_between(x=np.arange(10), y1=y1_convlstm, y2=y2_convlstm, color='lightblue',alpha=alpha)
plt.plot(d_conv_tf_att, linestyle='--',color='orange')
plt.fill_between(x=np.arange(10), y1=y1_conv_tf_att, y2=y2_conv_tf_att, color='bisque',alpha=alpha)

plt.xlim(0, 9)
plt.ylim(0.02, 0.08)

plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    ['', '', '', '', '', '', '', '', '', ''])
plt.text(0.3, 0.072,  'b', fontweight='bold', fontsize=12)
plt.ylabel('unbiased RMSE', fontsize=12)
# -------------------------------------------------------------------------------------------------
plt.subplot2grid((7, 4), (3, 2), rowspan=2, colspan=2)
d_gfs=[]
d_convlstm=[]
d_conv_tf_att=[]

y1_gfs = []
y1_convlstm=[]
y1_conv_tf_att=[]

y2_gfs = []
y2_convlstm=[]
y2_conv_tf_att=[]

for i in range(10):
    idx = np.argwhere((m>np.nanquantile(m, i/10)) & (m<np.nanquantile(m, (i+1)/10)))
    d_gfs.append(np.nanmean(urmse_gfs[idx, 6]))
    d_convlstm.append(np.nanmean(urmse_convlstm[idx, 6]))
    d_conv_tf_att.append(np.nanmean(urmse_convlstm_pb_att_tf_mean[idx, 6]))

    y1_gfs.append(np.nanquantile(urmse_gfs[idx, 6], 0.25))
    y1_convlstm.append(np.nanquantile(urmse_convlstm[idx, 6], 0.25))
    y1_conv_tf_att.append(np.nanquantile(urmse_convlstm_pb_att_tf_mean[idx, 6], 0.25))

    y2_gfs.append(np.nanquantile(urmse_gfs[idx, 6], 0.75))
    y2_convlstm.append(np.nanquantile(urmse_convlstm[idx, 6], 0.75))
    y2_conv_tf_att.append(np.nanquantile(urmse_convlstm_pb_att_tf_mean[idx, 6], 0.75))

alpha=0.3

plt.plot(d_gfs, color='red')
plt.fill_between(x=np.arange(10), y1=y1_gfs, y2=y2_gfs, color='mistyrose', alpha=alpha)
plt.plot(d_convlstm, color='blue')
plt.fill_between(x=np.arange(10), y1=y1_convlstm, y2=y2_convlstm, color='lightblue',alpha=alpha)
plt.plot(d_conv_tf_att, linestyle='--',color='orange')
plt.fill_between(x=np.arange(10), y1=y1_conv_tf_att, y2=y2_conv_tf_att, color='bisque',alpha=alpha)
plt.xlim(0, 9)
plt.ylim(0.02, 0.08)

plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    ['', '', '', '', '', '', '', '', '', ''])
plt.text(0.3, 0.072,  'd', fontweight='bold', fontsize=12)
plt.ylabel('unbiased RMSE', fontsize=12)
# -------------------------------------------------------------------------------------------------
plt.subplot2grid((7, 4), (5, 2), rowspan=2, colspan=2)
d_gfs=[]
d_convlstm=[]
d_conv_tf_att=[]

y1_gfs = []
y1_convlstm=[]
y1_conv_tf_att=[]

y2_gfs = []
y2_convlstm=[]
y2_conv_tf_att=[]

for i in range(10):
    idx = np.argwhere((m>np.nanquantile(m, i/10)) & (m<np.nanquantile(m, (i+1)/10)))
    d_gfs.append(np.nanmean(urmse_gfs[idx, -1]))
    d_convlstm.append(np.nanmean(urmse_convlstm[idx, -1]))
    d_conv_tf_att.append(np.nanmean(urmse_convlstm_pb_att_tf_mean[idx, -1]))

    y1_gfs.append(np.nanquantile(urmse_gfs[idx, -1], 0.25))
    y1_convlstm.append(np.nanquantile(urmse_convlstm[idx, -1], 0.25))
    y1_conv_tf_att.append(np.nanquantile(urmse_convlstm_pb_att_tf_mean[idx, -1], 0.25))

    y2_gfs.append(np.nanquantile(urmse_gfs[idx, -1], 0.75))
    y2_convlstm.append(np.nanquantile(urmse_convlstm[idx, -1], 0.75))
    y2_conv_tf_att.append(np.nanquantile(urmse_convlstm_pb_att_tf_mean[idx, -1], 0.75))

alpha=0.3

plt.plot(d_gfs, color='red')
plt.fill_between(x=np.arange(10), y1=y1_gfs, y2=y2_gfs, color='mistyrose', alpha=alpha)
plt.plot(d_convlstm, color='blue')
plt.fill_between(x=np.arange(10), y1=y1_convlstm, y2=y2_convlstm, color='lightblue',alpha=alpha)
plt.plot(d_conv_tf_att, linestyle='--',color='orange')
plt.fill_between(x=np.arange(10), y1=y1_conv_tf_att, y2=y2_conv_tf_att, color='bisque',alpha=alpha)
plt.xlim(0, 9)
plt.ylim(0.02, 0.08)

plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
    ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',\
         '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'], rotation=45)
plt.text(0.3, 0.072,  'f', fontweight='bold', fontsize=12)
plt.ylabel('unbiased RMSE', fontsize=12)
# -------------------------------------------------------------------------------------------------
plt.subplots_adjust(wspace=0.7)
plt.savefig('figureS5.pdf')
print('Figure S5 Completed!')
# -------------------------------------------------------------------------------------------------
"""




"""
#--------------------------------------------------------------------------------------------------
# Figure S6: 
#--------------------------------------------------------------------------------------------------
r_noy = np.load('r_noy.npy')
r_noy_drived = np.load('r_noy_drived.npy')
r_lagy = np.load('r_lagy.npy')
r_lagy_drived = np.load('r_lagy_drived.npy')
r_all = np.load('r.npy')
r_convlstm = r_all[1]
r_convlstm_pb = r_all[3]

plt.figure(figsize=(12, 5))
ms=4
colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf',
          '#1a55FF']
ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
ax1.spines['left'].set_linewidth(1)
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['right'].set_linewidth(1)
ax1.spines['top'].set_linewidth(1)
plt.plot(np.nanmean(r_convlstm, axis=0), marker='o', ms=ms, color=colors[2])
plt.plot(np.nanmean(r_convlstm_pb, axis=0), marker='o', ms=ms, color=colors[5])
plt.plot(np.nanmean(r_noy, axis=0), marker='o',ms=ms, color=colors[4])
plt.plot(np.nanmean(r_noy_drived, axis=0), marker='o',ms=ms, color=colors[-2])
plt.plot(np.nanmean(r_lagy, axis=0), marker='o',ms=ms, color=colors[1])
plt.plot(np.nanmean(r_lagy_drived, axis=0), marker='o',ms=ms, color=colors[0])
plt.xlabel('Forecast Time Scale (day)', fontsize=12)
plt.ylabel('Mean R', fontsize=12)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], \
    labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'])
#plt.legend(['GFS', 'ConvLSTM', 'Mean', 'Drived', 'Conditioned', 'Conditioned\nAttention', #'Mean\nConditioned\nAttention', 'SMAP L4'], fancybox=False, shadow=False, fontsize=8)
plt.legend(['ConvLSTM', 'GdC', 'ConvLSTM*', 'condition*', 'ConvLSTM**', 'condition**'], fancybox=False, shadow=False, fontsize=8)
plt.subplots_adjust(wspace=0.3)
plt.savefig('figureS6.pdf')
print('Figure S6 Completed!')
# -------------------------------------------------------------------------------------------------
"""


#--------------------------------------------------------------------------------------------------
# Figure S7: TCA of AMSR2, SMOS
#-------------------------------------------------------------------------------------------
tc_all_bk = np.load(configs.outputs_path+'tc_smos_amsr2.npy')
tc_all = tc_all_bk
mask_nan = mask_in_china
mask_nan[mask_in_china==0] = np.nan

tc_gfs_bk = tc_all_bk[-1] + mask_nan
tc_convlstm_bk = tc_all_bk[0] + mask_nan
tc_convlstm_mean_bk = tc_all_bk[1] + mask_nan
tc_convlstm_pb_tf_bk = tc_all_bk[2] + mask_nan
tc_convlstm_pb_att_tf_bk = tc_all_bk[3] + mask_nan
tc_convlstm_pb_att_tf_mean_bk = tc_all_bk[4] + mask_nan
tc_gfs = tc_all[-1] + mask_nan
tc_convlstm = tc_all[0] + mask_nan
tc_convlstm_mean = tc_all[1] + mask_nan
tc_convlstm_pb_tf = tc_all[2] + mask_nan
tc_convlstm_pb_att_tf = tc_all[3] + mask_nan
tc_convlstm_pb_att_tf_mean = tc_all[4] + mask_nan

# Figure 
plt.figure(figsize=(9, 10))
cmap = 'gist_rainbow'
cmap = 'Spectral'
cmap = 'jet'
cmap = 'bwr'
vmin=-10
vmax=10
mdls = ['GFS', 'ConvLSTM', 'Mean', 'Conditioned','Att-Conditioned', 'Att-TF-Mean']
colors = category_colors = plt.get_cmap('RdYlGn')(
    np.linspace(0.15, 0.85, len(mdls)))
def diff_snr(x, y): return x-y
#----------------------------------------------------------------
ax = plt.subplot(6, 4, 5)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x, y, mask_in_china * diff_snr(tc_gfs[0], tc_convlstm_mean[0]),vmin=vmin, vmax=vmax, cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(GFS)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 6)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x, y, mask_in_china *diff_snr(tc_gfs[6], tc_convlstm_mean[6]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 7)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x, y, mask_in_china *diff_snr(tc_gfs[-1], tc_convlstm_mean[-1]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 9)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x,y,mask_in_china * diff_snr(tc_convlstm[0], tc_convlstm_mean[0]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(ConvLSTM)', fontsize=10, rotation=90)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china * diff_snr(tc_convlstm[6], tc_convlstm_mean[6]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 11)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm[-1], tc_convlstm_mean[-1]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection

m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_mean_bk[0] + mask_nan, vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('average', fontsize=10, rotation=90)
plt.title('1-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_mean_bk[6]+ mask_nan, vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('7-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 3)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, tc_convlstm_mean_bk[-1]+ mask_nan, vmin=-20, vmax=20,  cmap='jet')
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('16-day', fontsize=12)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 13)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_tf[0], tc_convlstm_mean[0]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(condition)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 14)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china * diff_snr(tc_convlstm_pb_tf[6], tc_convlstm_mean[6]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 15)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_tf[-1], tc_convlstm_mean[-1]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 17)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_att_tf[0], tc_convlstm_mean[0]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(attention)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 18)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y,mask_in_china *diff_snr(tc_convlstm_pb_att_tf[6], tc_convlstm_mean[6]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 19)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y,mask_in_china *diff_snr(tc_convlstm_pb_att_tf[-1], tc_convlstm_mean[-1]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 21)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_att_tf_mean[0], tc_convlstm_mean[0]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(ensemble)', fontsize=10, rotation=90)
axin = ax.inset_axes([0.1, -0.2, 0.5, 0.2])
axin.spines['left'].set_linewidth(0)
axin.spines['bottom'].set_linewidth(0)
axin.spines['right'].set_linewidth(0)
axin.spines['top'].set_linewidth(0)
axin.set_xticks([])
axin.set_yticks([])
axin.text(0, -0.2, '${\delta}$'+'(*) = '+'SNR(*)-SNR(average)')
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 22)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_att_tf_mean[6], tc_convlstm_mean[6]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 23)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
#m.drawcoastlines(linewidth=1)
x, y = m(x0, y0)
sc = m.pcolormesh(x,y, mask_in_china *diff_snr(tc_convlstm_pb_att_tf_mean[-1], tc_convlstm_mean[-1]), vmin=vmin, vmax=vmax,  cmap=cmap)
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
plt.subplots_adjust(hspace=-0.27, wspace=0)
plt.savefig('figureS7.png', dpi=600)
print('Figure S7 Completed!')
#--------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------
# Figure S8: TCA(SMOS,AGME)
# -------------------------------------------------------------------------------------------------
r_all = np.load(configs.outputs_path+'tc_smos_agme.npy')
r_gfs = r_all[-1] 
r_convlstm = r_all[0] 
r_convlstm_mean = r_all[1] 
r_convlstm_pb_tf = r_all[2] 
r_convlstm_pb_att_tf = r_all[3] 
r_convlstm_pb_att_tf_mean = r_all[4] 

# spatial pattern for 1, 7, 16 time scale forecast over different sites
plt.figure(figsize=(9, 10))
NS = 1.5
vmin=-10
vmax=10
cmap = 'bwr'
mdls = ['GFS', 'ConvLSTM', 'average','condition', 'attention', 'ensemble']
colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, len(mdls)))
#----------------------------------------------------------------
ax = plt.subplot(6, 4, 5)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_gfs[0]-r_convlstm_mean[0], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)
plt.title('1-day', fontsize=12)
plt.ylabel('${\delta}$'+'(GFS)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 6)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_gfs[6]-r_convlstm_mean[6], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 7)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_gfs[-1]-r_convlstm_mean[-1], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 9)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[0]-r_convlstm_mean[0], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(ConvLSTM)', fontsize=10, rotation=90)
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[6]-r_convlstm_mean[6], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 11)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[-1]-r_convlstm_mean[-1], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc1 = m.scatter(x,y, c=r_convlstm_mean[0], marker='.',s=NS,vmin=vmin, vmax=vmax, cmap='rainbow')

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc1, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)
plt.ylabel('average', fontsize=10, rotation=90)
plt.title('1-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_mean[6], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap='rainbow')

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('7-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 3)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_mean[-1], marker='.',s=NS,vmin=vmin, vmax=vmax, cmap='rainbow')

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('16-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 13)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_tf[0]-r_convlstm_mean[0], marker='.',s=NS,vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(condition)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 14)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_tf[6]-r_convlstm_mean[6], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 15)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_tf[-1]-r_convlstm_mean[-1], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 17)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[0]-r_convlstm_mean[0], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(attention)', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 18)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[6]-r_convlstm_mean[6], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 19)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[-1]-r_convlstm_mean[-1], marker='.',s=NS,vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 21)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf_mean[0]-r_convlstm_mean[0], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('${\delta}$'+'(ensemble)', fontsize=10, rotation=90)

axin = ax.inset_axes([0.1, -0.2, 0.5, 0.2])
axin.spines['left'].set_linewidth(0)
axin.spines['bottom'].set_linewidth(0)
axin.spines['right'].set_linewidth(0)
axin.spines['top'].set_linewidth(0)
axin.set_xticks([])
axin.set_yticks([])
axin.text(0, 0, '${\delta}$'+'(*) = '+'R(*)-R(SA-GC)')
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 22)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf_mean[6]-r_convlstm_mean[6], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 23)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf_mean[-1]-r_convlstm_mean[-1], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
#--------------------------------------------------------------------------------------------------
plt.subplots_adjust(hspace=-0.27, wspace=0)
plt.savefig('figureS8.pdf')
print('Figure S8 Completed!')
# -------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------
# Figure S9. SMAP ceiling of AGME
#--------------------------------------------------------------------------------------------------
r = np.load(configs.outputs_path+'r.npy')
r_smap = r[-1]
r_convlstm = r[1]
r_ens = r[-2]

# Figure 
plt.figure(figsize=(9, 10))
cmap = 'gist_rainbow'
cmap = 'Spectral'
cmap = 'bwr'
cmap = 'jet'
vmin=0
vmax=1
tc_smap_bk = tc_smap
colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, len(mdls)))
#----------------------------------------------------------------
ax = plt.subplot(6, 4, 20)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_smap[:,0], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('SMAP L4', fontsize=12)#, fontweight='bold')
#----------------------------------------------------------------
ax = plt.subplot(6, 4, 18)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)

m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_ens[:,0]-r_smap[:,0], marker='.',s=NS, vmin=-0.2, vmax=0.2, cmap='jet')

axin = ax.inset_axes([0.05, 0.2, 0.2, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('(R(ensemble)-R(SMAP L4))\nfor 1-day forecast', fontsize=10)#, fontweight='bold')
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 9)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_ens[:,0], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.ylabel('ensemble', fontsize=10, rotation=90)
plt.title('1-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_ens[:,6], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('7-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 4, 11)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=55,
            llcrnrlon=70, urcrnrlon=138,
            ax=ax)  # mill projection
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_ens[:,-1], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)

m.readshapefile("/tera03/lilu/work/CLFS/china-shapefiles/china_country",  'china_country', drawbounds=True)

axin1 = ax.inset_axes([0.75, 0.02, 0.3, 0.3])
ll_lons, ll_lat, ur_lon, ur_lat, lat_0, lon_0 = 105, 0, 125, 25, 12, 115
m1 = Basemap(projection='mill',
            llcrnrlat=0, urcrnrlat=25,
            llcrnrlon=105, urcrnrlon=125,
            ax=axin1)  # mill projection
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china', 'china', drawbounds=True)
m1.readshapefile('/tera03/lilu/work/CLFS/china-shapefiles/china_nine_dotted_line', 'dashed9',
                drawbounds=True)
plt.title('16-day', fontsize=12)
#--------------------------------------------------------------------------------------------------
plt.subplots_adjust(hspace=-0.27, wspace=0)
plt.savefig('figureS9.png', dpi=600)
print('Figure S9 Completed!')
#--------------------------------------------------------------------------------------------------















# -------------------------------------------------------------------------------------------------
# FIG10. Ranking
# -------------------------------------------------------------------------------------------------
"""
mdls = ['GFS', 'ConvLSTM', 'Em-GC', 'GcC','AGcC', 'Em-all']

rank1 = np.full_like(r_gfs, np.nan)
print(r_all.shape)
a = r_all[0:3]
b = r_all[4:5]
d = r_all[6:-1]
c = np.concatenate([a, b, d], axis=0)

for t in range(r_gfs.shape[1]):
    for i in range(r_gfs.shape[0]):
        if np.isnan(c[:, i, t]).any():
            pass
        else:
            index = np.argsort(c[:, i, t])
            rank1[i, t] = index[-1]

fig = plt.figure(figsize=(7, 5))
ax = plt.subplot(111)

n = len(rank1[:, 0])
scale = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
category_names = mdls
result = {}

data = np.full((len(scale), len(mdls)), np.nan)
for i in range(len(scale)):

    kk = rank1[:, t-1]
    n = len(kk[~np.isnan(kk)])

    list = []
    t = scale[i]
    tmp = rank1[:, t-1]
    tmp1 = tmp[~np.isnan(tmp)]
    for j in range(len(mdls)):
        list.append((len(tmp1[tmp1==j])/len(tmp1))*100)
    data[i, :] = list
    print(list)
    result[str(scale[i])+'-day'] = list


labels = []
for i in range(1, 17):
    labels.append(str(i))

print(data.shape)
data_cum = data.cumsum(axis=1)
print(data_cum)
category_colors = colors

ax.invert_yaxis()
#ax.invert_xaxis()
ax.xaxis.set_visible(True)
ax.set_xlim(0, np.sum(data, axis=1).max())
category_colors = plt.get_cmap('RdYlGn')(
    np.linspace(0.15, 0.85, data.shape[1]))

for i, (colname, color_) in enumerate(zip(category_names, category_colors)):
    widths = data[:, i]
    starts = data_cum[:, i] - widths
    rects = ax.barh(labels, widths, left=starts, height=0.6,
                    label=colname, color=color_)
    r, g, b, _ = color_
    #text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
    #ax.bar_label(rects, label_type='center', color=text_color)

ax.legend(ncol=7, loc='lower left', mode='expand', bbox_to_anchor=(0, 1.05, 1, 0.2), fontsize=7, shadow=False, fancybox=True)
plt.ylabel('Forecast Time Scale (day)', rotation=90)
plt.ylim(-0.5, 15.5)
plt.xticks([0, 20, 40, 60, 80, 100], ['0%', '20%', '40%', '60%', '80%', '100%'],fontsize=10)
ax1 = ax.twiny()
plt.xticks([100, 80, 60, 40, 20, 0], ['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=10)
plt.savefig('figure10.pdf')
print('Figure 10 Completed!')
"""
#--------------------------------------------------------------------------------------------------






# -------------------------------------------------------------------------------------------------
# FIGS1.  spatial distribution of R of SWDI
# -------------------------------------------------------------------------------------------------
"""
r_all = np.load('r_all_swdi.npy')
r_gfs = r_all[0]
r_convlstm = r_all[1]
r_convlstm_mean = r_all[2]
r_convlstm_pb_tf = r_all[3]
r_convlstm_pb_att_tf = r_all[4]
r_convlstm_pb_att_tf_mean = r_all[5]

urmse_all = np.load('urmse_all_swdi.npy')
urmse_gfs = urmse_all[0]
urmse_convlstm = urmse_all[1]
urmse_convlstm_mean = urmse_all[2]
urmse_convlstm_pb_tf = urmse_all[3]
urmse_convlstm_pb_att_tf = urmse_all[4]
urmse_convlstm_pb_att_tf_mean = urmse_all[5]

# get lat/lon gfs
lat_gfs = np.arange(90, -90.1, -0.25)
lon_gfs = np.arange(0, 359.8, 0.25)
idx_lat = np.where((lat_gfs>14.7) & (lat_gfs<53.5))[0]
idx_lon = np.where((lon_gfs>72.3) & (lon_gfs<135))[0]
lat_gfs = lat_gfs[idx_lat]
lon_gfs = lon_gfs[idx_lon]
x1,y1 = np.meshgrid(lon_gfs, lat_gfs)

# get lat/lon convlstm
x0,y0 = np.meshgrid(lon_smap, lat_smap)

# get lat/lon AGME
AGME_auxiliary = np.load(path + 'AGME_auxiliary.npy')
lat_agme = AGME_auxiliary[:,0]
lon_agme = AGME_auxiliary[:,1]


# spatial pattern for 1, 7, 16 time scale forecast over different sites
fig = plt.figure(figsize=(4.5, 7.3))
NS = 1.5
vmin=-0.2
vmax=0.2
cmap = 'bwr'
#
mdls = ['GFS', 'ConvLSTM', 'Mean', 'Conditioned','Att-Conditioned', 'Att-TF-Mean']
colors = category_colors = plt.get_cmap('RdYlGn')(
    np.linspace(0.15, 0.85, len(mdls)))
#['black', 'pink', 'orange', 'green', 'blue', 'purple', 'yellow', 'red']
#----------------------------------------------------------------
ax = plt.subplot(6, 3, 4)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_gfs[:, 0]-r_convlstm_mean[:, 0], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)
plt.ylabel('GFS', fontsize=10, rotation=90)#, fontweight='bold')
plt.title('1-day', fontsize=12)
axin = ax.inset_axes([0.08, 0.85, 0.4, 0.07])
cbar = plt.colorbar(sc, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)
cbar.set_label('${\delta}$'+'R',fontsize=7)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 5)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_gfs[:, -1]-r_convlstm_mean[:, -1], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 6)
sns.kdeplot(data=r_gfs[:,0], shade=True, color="g")
sns.kdeplot(data=r_gfs[:,-1], shade=True, color="orange")
plt.xticks([])
plt.xlim(-1, 1)
plt.ylim(0, 2)
plt.yticks([])
plt.ylabel('')
#-------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 7)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[:, 0]-r_convlstm_mean[:, 0], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)
plt.ylabel('ConvLSTM', rotation=90, fontsize=10)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 8)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm[:, -1]-r_convlstm_mean[:, -1], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 9)
sns.kdeplot(data=r_convlstm[:,0], shade=True, color="g")
sns.kdeplot(data=r_convlstm[:,-1], shade=True, color="orange")
plt.xticks([])
plt.yticks([])
plt.ylabel('')
plt.xlim(-1, 1)
plt.ylim(0, 2)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 1)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc1 = m.scatter(x,y, c=r_convlstm_mean[:, 0], marker='.',s=NS,vmin=0, vmax=1, cmap='rainbow')
plt.ylabel('Em-GC', fontsize=10, rotation=90)
axin = ax.inset_axes([0.08, 0.85, 0.4, 0.07])
cbar = plt.colorbar(sc1, cax=axin, orientation='horizontal')
axin.tick_params(axis='x', labelsize=5)
cbar.set_ticks([0, 0.5, 1], ['0', '0.5', '1'])
cbar.set_label('R',fontsize=7)
plt.title('Week 1', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 2)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_mean[:, -1], marker='.',s=NS, vmin=0, vmax=1, cmap='rainbow')
plt.title('Week 2', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 3)
sns.kdeplot(data=r_convlstm_mean[:,0], shade=True, color="g")
sns.kdeplot(data=r_convlstm_mean[:,-1], shade=True, color="orange")
plt.xticks([])
plt.yticks([])
plt.ylabel('')
plt.xlim(-1, 1)
plt.ylim(0, 2)
plt.title('KDE curve', fontsize=12)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 10)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_tf[:, 0]-r_convlstm_mean[:, 0], marker='.',s=NS,vmin=vmin, vmax=vmax,  cmap=cmap)
plt.ylabel('GcC', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 11)
ax.spines['left'].set_linewidth(0)
ax.spines['bottom'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_tf[:, -1]-r_convlstm_mean[:, -1], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 12)
sns.kdeplot(data=r_convlstm_pb_tf[:,0], shade=True, color="g")
sns.kdeplot(data=r_convlstm_pb_tf[:,-1], shade=True, color="orange")
plt.xticks([])
plt.yticks([])
plt.ylabel('')
plt.xlim(-1, 1)
plt.ylim(0, 2)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 13)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[:, 0]-r_convlstm_mean[:, 0], marker='.',s=NS, vmin=vmin, vmax=vmax,  cmap=cmap)
plt.ylabel('AGcC', fontsize=10, rotation=90)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 14)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf[:, -1]-r_convlstm_mean[:, -1], marker='.',s=NS,vmin=vmin, vmax=vmax, cmap=cmap)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 15)
sns.kdeplot(data=r_convlstm_pb_att_tf[:,0], shade=True, color="g")
sns.kdeplot(data=r_convlstm_pb_att_tf[:,-1], shade=True, color="orange")
plt.xticks([])
plt.yticks([])
plt.ylabel('')
plt.xlim(-1, 1)
plt.ylim(0, 2)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 16)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf_mean[:, 0]-r_convlstm_mean[:, 0], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)
plt.ylabel('Em-all', fontsize=10, rotation=90)
axin = ax.inset_axes([0.1, -0.2, 0.5, 0.2])
axin.spines['left'].set_linewidth(0)
axin.spines['bottom'].set_linewidth(0)
axin.spines['right'].set_linewidth(0)
axin.spines['top'].set_linewidth(0)
axin.set_xticks([])
axin.set_yticks([])
axin.text(0, 0, '${\delta}$'+'R = '+'R(*)-R(Mean)')
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 17)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
m = Basemap(projection='mill',
            llcrnrlat=14.7, urcrnrlat=53.5,
            llcrnrlon=75, urcrnrlon=130,
            ax=ax)  # mill projection
m.drawcoastlines(linewidth=1)
x, y = m(lon_agme,lat_agme)
sc = m.scatter(x,y, c=r_convlstm_pb_att_tf_mean[:, -1]-r_convlstm_mean[:, -1], marker='.',s=NS, vmin=vmin, vmax=vmax, cmap=cmap)
#--------------------------------------------------------------------------------------------------
ax = plt.subplot(6, 3, 18)
ax.set_yticks([])
ax1 = ax.twinx()
sns.kdeplot(data=r_convlstm_pb_att_tf_mean[:,0], shade=True, color="g")
sns.kdeplot(data=r_convlstm_pb_att_tf_mean[:,-1], shade=True, color="orange")
ax1.set_xlim(-1, 1)
ax1.set_ylim(0, 2)
ax1.set_ylabel('Density')
ax1.set_xlabel('R')
plt.text(-0.09, -0.8, 'R')
plt.legend(['Week 1', 'Week 2'], ncol=1, loc=2 ,fontsize=4.5, shadow=False, fancybox=True)
#--------------------------------------------------------------------------------------------------
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('figureS1.pdf')
print('Figure S1 Completed!')
# -------------------------------------------------------------------------------------------------
"""
