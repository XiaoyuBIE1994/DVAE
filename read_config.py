#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

from configparser import ConfigParser

# Re-write configure class, enable to distinguish betwwen upper and lower letters
class myconf(ConfigParser):
    def __init__(self,defaults=None):
        ConfigParser.__init__(self,defaults=None)
    def optionxform(self, optionstr):
        return optionstr


def read_config(config_file):
    cfg = myconf()
    cfg.read(config_file)
    cfg_str = cfg.get('user', 'local_host')
    cfg_boolen = cfg.getboolean('STFT', 'trim')
    cfg_int = cfg.getint('STFT', 'fs')
    cfg_float = cfg.getfloat('STFT', 'wlen_sec')
    cfg_list = [int(i) for i in cfg.get('network', 'hidden_dim_encoder').split(',')]
    print(cfg_str)
    print(cfg_boolen)
    print(cfg_int)
    print(cfg_float)
    print(cfg_list)
    print(type(cfg_list))
    return


if __name__ == '__main__':
    import sys
    config_dict = read_config(sys.argv[1])
    