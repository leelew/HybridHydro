import numpy as np
import os
import h5py


def get_KG(path):
    data = h5py.File(path+'Beck_KG_V1_present_0p5.mat', 'r')
    kg = np.transpose(data['KG'])

    kg_idx = np.full_like(kg, np.nan)
    # exclude Tropical area cuz only few points
    # set more rough spatial
    for i in range(1, 4):
        kg_idx[kg == i] = 1 # tropical
    for i in range(4, 8):
        kg_idx[kg == i] = 2 # arid
    for i in range(8, 17):
        kg_idx[kg == i] = 3 # temperate
    for i in range(17, 29):
        kg_idx[kg == i] = 4 # cold
    for i in range(29, 31):
        kg_idx[kg == i] = 5 # polar
    return kg_idx

# generate week 1 week 2 swdi
def gen_week_swdi(input): #（365，16，448，672）
    tmp = input[:, 0] # (342, 448, 672)
    q05, q95 = np.nanquantile(tmp, 0.05, axis=0), np.nanquantile(tmp, 0.95, axis=0) #(448, 672)
    sm_week1 = np.nanmean(input[:, :8], axis=1) # (342, 448, 672)
    sm_week2 = np.nanmean(input[:, 8:], axis=1)
    swdi_week1 = (sm_week1 - q95) / (q95-q05) * 10
    swdi_week2 = (sm_week2 - q95) / (q95-q05) * 10 # (342, 448, 672)
    return swdi_week1, swdi_week2

def unbiased_RMSE(y_true, y_pred):
    predmean = np.nanmean(y_pred)
    targetmean = np.nanmean(y_true)
    predanom = y_pred-predmean
    targetanom = y_true - targetmean
    return np.sqrt(np.nanmean((predanom-targetanom)**2))

def init_fold(work_path):
    if not os.path.exists(work_path+"input"):
        os.mkdir(work_path+"input")
    if not os.path.exists(work_path+"output"):
        os.mkdir(work_path+"output")
    if not os.path.exists(work_path+"logs"):
            os.mkdir(work_path+"logs")
    if not os.path.exists(work_path+"output/saved_model"):
        os.mkdir(work_path+"output/saved_model")
    if not os.path.exists(work_path+"output/forecast"):
        os.mkdir(work_path+"output/forecast")
