import sys; sys.path.append('../')
from models import convlstm


def model(configs):
    networks_map = {
            'convlstm': convlstm.convlstm,
            'convlstm_drive': convlstm.convlstm_drive,
            'convlstm_condition': convlstm.convlstm_condition,
            'convlstm_linear_att_condition': convlstm.convlstm_linear_att_condition,
            'convlstm_se_att_condition': convlstm.convlstm_se_att_condition,
            'convlstm_condition_forcing': convlstm.convlstm_condition_forcing,
            'convlstm_condition_forcing_gfs': convlstm.convlstm_condition_forcing_gfs,
            'convlstm_condition_p': convlstm.convlstm_condition_p,
            'convlstm_condition_p_gfs': convlstm.convlstm_condition_p_gfs
        }
    if configs.model_name in networks_map:
        network = networks_map[configs.model_name](configs)
    else:
        raise ValueError('Name of network unknown %s' % configs.model_name)
    return network


