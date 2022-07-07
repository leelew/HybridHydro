# ==============================================================================
# Read SMAP and CLDAS, and grid match inputs and outputs.
#
# author: Lu Li, 2021/09/27
# ==============================================================================

import numpy as np
#from data.read_cldas import read_preprocessed_daily_cldas_forcing
from data.read_smap import read_daily_smap, read_preprocessed_daily_smap_force


def grid_match(X, y, Xlat, Xlon, Xres, ylat, ylon, yres):

    if y.ndim != 4 & X.ndim != 4:
        raise ValueError("must 4D")
    else:
        N, Nlat, Nlon, _ = y.shape

    matched_X = np.full((N, Nlat, Nlon, X.shape[-1]), np.nan)

    for i in range(len(ylat)):
        for j in range(len(ylon)):

            # grid match index
            lat, lon = ylat[i], ylon[j]
            lat_idx = np.where((Xlat < (lat + yres / 2))
                               & (Xlat > (lat - yres / 2)))[0]
            lon_idx = np.where((Xlon < (lon + yres / 2))
                               & (Xlon > (lon - yres / 2)))[0]

            print(lat_idx)
            print(lon_idx)
            print(X[:, lat_idx, lon_idx, :].shape)
            # average mapping
            matched_X[:, i, j, :] = np.nanmean(X[:, lat_idx, lon_idx, :],
                                               axis=(-2))

    return matched_X, y


def grid_match_xy(X_path, y_path, begin_date, end_date):

    y, ylat, ylon = read_daily_smap(input_path=y_path,
                                    begin_date=begin_date,
                                    end_date=end_date)

    X, Xlat, Xlon = read_preprocessed_daily_smap_force(input_path=X_path,
                                                       begin_date=begin_date,
                                                       end_date=end_date)

    #assert y.shape[0] == X.shape[0]

    #X, y = grid_match(X, y, Xlat, Xlon, 0.0625, ylat,
    #                  ylon, 0.09)  # 9km, 0.0625

    return X, y
