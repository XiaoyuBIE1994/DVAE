#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""

import logging

def get_logger(file_path = 'log.txt', handle = 1):
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create Handler
    # type 1: file handler
    # type 2: stream handler
    if handle == 1:
        log_handler = logging.FileHandler(file_path, mode='a', encoding='UTF-8')
    elif handle == 2:
        log_handler = logging.StreamHandler()
    else:
        log_handler = logging.FileHandler(file_path, mode='w', encoding='UTF-8')

    # Set formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)

    # Add to logger
    logger.addHandler(log_handler)

    return logger
    

if __name__ == '__main__':
    logger = get_logger('log.log', 2)
    logger.debug('debug xxx')
    logger.info('info xxx')
    logger.warning('warn xxx')
    logger.error('error xxx')
    logger.critical('critical xxx')
    logger.debug('%s is customize infomation' % 'this')