#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from configparser import ConfigParser

class myconf(ConfigParser):
    def __init__(self,defaults=None):
        ConfigParser.__init__(self, defaults=None)
    def optionxform(self, optionstr):
        return optionstr

cfg_file = 'config/cfg_dkf.ini'
cfg = myconf()
cfg.read(cfg_file)

a = cfg.get('Network', 'dense_x_gx').split(',')
print(a)
print(type(a))
print(len(a))
# dense_x_gx = [int(i) for i in cfg.get('Network', 'dense_x_gx').split(',')]