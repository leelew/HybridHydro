import sys; sys.path.append('../')
from models import convlstm


def model(configs):
    networks_map = {
            'convlstm': convlstm.convlstm,
            'convlstm_drive': convlstm.convlstm_drive,
            'convlstm_condition': convlstm.convlstm_condition,
            'convlstm_att_condition': convlstm.convlstm_att_condition
        }
    if configs.model_name in networks_map:
        network = networks_map[configs.model_name](configs)
    else:
        raise ValueError('Name of network unknown %s' % configs.model_name)
    return network


