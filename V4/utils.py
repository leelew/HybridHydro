import numpy as np
import os


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
