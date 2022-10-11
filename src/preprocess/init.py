import glob
import json
import os
import sys; sys.path.append('../../HybridHydro/')
import numpy as np

from readers import RawSMAPReader


class AuxManager():
    def __init__(self):
        pass

    def init(self, raw_data_path, auxiliary_data_path, lat_lower, lat_upper,
             lon_left, lon_right):

        # init land mask and latitude, longitude
        l = glob.glob(raw_data_path + '/2015.05.31/' + 'SMAP*h5',
                      recursive=True)
        data, lat, lon = RawSMAPReader(lat_lower, lat_upper, lon_left,
                                       lon_right)(l[0])
        mask = self.get_mask(data)
        print(lat.shape)
        print(lon.shape)

        # init attribute and land mask
        aux = {
            'Nlat': lat.shape[0],
            'Nlon': lon.shape[1],
            'mask': mask.tolist(),
            'lat_2d': lat.tolist(),
            'lon_2d': lon.tolist(),
            'lon_left': [0, 112, 224, 336, 448, 560, 
                         0, 112, 224, 336, 448, 560, 
                         0, 112, 224, 336, 448, 560, 
                         0, 112, 224, 336, 448, 560],
            'lat_low': [0, 0, 0, 0, 0, 0, 
                        112, 112, 112, 112, 112, 112, 
                        224, 224, 224, 224, 224, 224,
                        336, 336, 336, 336, 336, 336]
        }

        full_path = os.path.join(auxiliary_data_path, 'auxiliary.json')
        with open(full_path, 'w') as f:
            json_string = json.dumps(aux)
            f.write(json_string)

    @staticmethod
    def update(auxiliary_data_path, key, value):

        with open(auxiliary_data_path + 'auxiliary.json', 'r') as f:
            aux = json.load(f)

        aux[key] = value

        with open(auxiliary_data_path + 'auxiliary.json', 'w') as f:
            json_string = json.dumps(aux)
            f.write(json_string)

    @staticmethod
    def get_mask(data):
        mask = np.ones_like(data)
        mask[np.isnan(data)] = 0
        return np.array(mask)


if __name__ == '__main__':
    Init(raw_data_path='/hard/lilu/SMAP_L4/SMAP_L4/',
         auxiliary_data_path='/hard/lilu/SMAP_L4/test/',
         lat_lower=10,
         lat_upper=20,
         lon_left=30,
         lon_right=40)
