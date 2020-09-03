#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import numpy as np
import soundfile as sf


def rmse_frame():

    def get_result(file_est, file_ref):
        x_est, _ = sf.read(file_est)
        x_ref, _ = sf.read(file_ref)
        # align
        len_x = len(x_est)
        x_ref = x_ref[:len_x]
        # scaling
        alpha = np.sum(x_est*x_ref) / np.sum(x_est**2)
        # x_est_ = np.expand_dims(x_est, axis=1)
        # alpha = np.linalg.lstsq(x_est_, x_ref, rcond=None)[0][0]
        x_est_scaled = alpha * x_est
        return np.sqrt(np.square(x_est_scaled - x_ref).mean())

    return get_result