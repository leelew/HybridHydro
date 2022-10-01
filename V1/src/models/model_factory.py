import sys; sys.path.append('../')
from models import convlstm, unet, persistence


def model(configs):
    networks_map = {
            'convlstm': convlstm.convlstm_v1,
            'unet': unet.unet9,
            'persistence': persistence.Persistence,
            'convlstm_tf': convlstm.convlstm_teacher_forcing
        }

    if configs.model_name in networks_map:
        network = networks_map[configs.model_name](configs)
    else:
        raise ValueError('Name of network unknown %s' % configs.model_name)
        
    return network


