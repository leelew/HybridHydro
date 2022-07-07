


import numpy as np



class SoilMoistureIndex():
    """Drought indices based on only soil moisture.
    
    
    Reference:
    [Narasimhan and Srinivasan et al. 2005]: Development and evaluation of 
                                             Soil Moisture Deficit Index(SMDI) and Evapotranspiration Deficit Index (ETDI) for agricultural drought monitoring.
    
    [Yidraw et al. 2008]: GRACE satellite observations of terrestrial moisture 
                          changes for drought characterization in the Canadian Prairie.
    """

    def __init__(self) -> None:
        pass

    def fit():
        pass

def SWDI(historical, inputs, predict=None):
    """Soil moisture deficit index(SMDI) from [Narasimhan and Srinivasan et al. 2005]."""
    if predict is not None:
        s, t, h, w, _ = predict.shape

        swdi_regions = np.full((int(s/7), h, w), np.nan)
        for i in range(h):
            for j in range(w):
                print(i,j)
                # 
                percentile95 = np.nanpercentile(historical[:, 0, i, j], 95)
                percentile05 = np.nanpercentile(historical[:, 0, i, j], 5)

                # 
                swdi_year = []
                
                for week in range(int(s/7)):
                    # see eq.8 in Narasimhan and Srinivasan et al. 2005
                    sw = np.nanmean(predict[:,:,:,:,0], axis=1)
                    sw = sw[week*7, i, j]
                    
                    swdi = (sw-percentile95)/(percentile95-percentile05)*10
                    swdi_year.append(swdi)

                swdi_regions[:, i, j] = np.array(swdi_year)

    else:
        t, _, h, w = inputs.shape

        swdi_regions = np.full((int(t/7), h, w), np.nan)
        for i in range(h):
            for j in range(w):
                print(i,j)
                # 
                percentile95 = np.nanpercentile(historical[:, 0, i, j], 95)
                percentile05 = np.nanpercentile(historical[:, 0, i, j], 5)

                # 
                swdi_year = []
                for week in range(int(t/7)):
                    # see eq.8 in Narasimhan and Srinivasan et al. 2005
                    sw = np.nanmean(inputs[week*7:week*7+7, 0, i, j])
                    
                    swdi = (sw-percentile95)/(percentile95-percentile05)*10
                    swdi_year.append(swdi)

                swdi_regions[:, i, j] = np.array(swdi_year)
    
    return swdi_regions



def SMDI(historical, inputs, predict=None):
    """Soil moisture deficit index(SMDI) from [Narasimhan and Srinivasan et al. 2005]."""
    t, _, h, w = inputs.shape

    smdi_regions = np.full((int(t/7), h, w), np.nan)
    for i in range(h):
        for j in range(w):
            print(i,j)
            # generate median, max, min for each grid
            max = np.nanmax(historical[:, 0, i, j])
            min = np.nanmin(historical[:, 0, i, j])
            median = np.nanmedian(historical[:, 0, i, j])

            # 
            smdi_year = []
            for week in range(int(t/7)):
                # see eq.8 in Narasimhan and Srinivasan et al. 2005
                sw = np.nanmean(inputs[week*7:week*7+7, 0, i, j])
                
                if sw <= median: 
                    sd = 100*(sw - median)/(median - min)
                elif sw > median:
                    sd = 100*(sw - median)/(max - median)

                if week == 0:
                    smdi = sd/50
                else:
                    smdi += sd/50
                print(smdi)
                smdi_year.append(smdi)

            smdi_regions[:, i, j] = np.array(smdi_year)

    return smdi_regions


if __name__ == '__main__':
    historical = np.load('/hard/lilu/inputs/SMAP_L4/y_train_l4_1_2015-2017.npy')
    #inputs = np.load('/hard/lilu/inputs/SMAP_L4/y_valid_l4_1_2018.npy')
    #historical = np.load('/hard/lilu/Compare/ERA5-land_swvl1_2015-2017.npy')
    inputs = np.load('/hard/lilu/Compare/ERA5-land_swvl1_2018.npy')
    predict = np.load('/hard/lilu/y_test_pred_1.npy')
    #smdi_regions = SMDI(historical, inputs)
    swdi_regions = SWDI(historical, inputs, predict=predict)

    #np.save('SMAP_L4_SMDI_2018.npy', smdi_regions)
    #np.save('SMAP_L4_SWDI_2018.npy', swdi_regions)
    #np.save('ERA5_Land_SWDI_2018.npy', swdi_regions)
    np.save('Predict_SWDI_2018.npy', swdi_regions)

            
