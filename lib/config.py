import yaml
import os
from easydict import EasyDict as edict


cfg = edict()

cfg.REG_PARAM_DEF = ((1.6471,4.3302),   # gasoline
                     (0.6564,0.7964),   # hybrid
                     (1.4216,5.9909))   # diesel

def cfg_from_file(filename):
    if filename[0] != '/':
        filename = os.path.abspath('.')+'/'+filename
    with open(filename, 'r', encoding='utf-8') as f:
        y = yaml.load(f)
        cfg.update(y)

cfg_from_file('experiments/cfgs/rank_car.yml')