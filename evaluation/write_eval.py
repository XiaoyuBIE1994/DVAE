#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt
"""

import os
import numpy as np

# write evaluation in files
def write_eval(eval_dic, audio_list, save_path):
    with open(save_path, 'w') as f:
        eval_list = list(eval_dic.keys())
        headline = ['filename'] + eval_list
        for item in headline:
            f.write('{}\t'.format(item))
        f.write('\n')
        for n, file in enumerate(audio_list):
            _, filename = os.path.split(file)
            f.write(filename)
            for eval_method in eval_list:
                f.write('\t{}'.format(eval_dic[eval_method][n]))
            f.write('\n')


# write result in tex table format
def write_eval_latex(result_dir, eval_dic_list, tag_list, audio_list):
    eval_list = list(eval_dic_list[0].keys())
    for eval_item in eval_list:
        if eval_item in ['stoi', 'pesq', 'pemoQ']:
            judging = 'max'
        else:
            judging = 'min'
        file_name = os.path.join(result_dir, '{}.tex'.format(eval_item))
        with open(file_name, 'w') as f:
            f.write('\\begin{table}[H]\n')             # \begin{table}[H]
            f.write('\\centering\n')                   # \centering
            f.write('\\begin{tabular}{r l l l l l}\n') # \begin{tabular}{rlllll}
            f.write('\\toprule\n')                     # \toprule
            f.write('filename')
            for item in tag_list:
                f.write(' & {}'.format(item))
            f.write(' \\\\\n')
            f.write('\\midrule\n') # \midrule
            
            for n, audio_file in enumerate(audio_list):
                _, audio_name = os.path.split(audio_file)
                f.write(audio_name)
                eval_models = []
                for i in range(len(eval_dic_list)):
                    eval_models.append(float(eval_dic_list[i][eval_item][n]))
                if judging == 'min':
                    idx_least = np.argmin(np.array(eval_models))
                else:
                    idx_least = np.argmax(np.array(eval_models))
                # bold the best eval model
                for j in range(len(eval_models)):
                    if j == idx_least:
                        f.write(' & \\textbf{{{}}}'.format(eval_dic_list[j][eval_item][n]))
                    else:
                        f.write(' & {}'.format(eval_dic_list[j][eval_item][n]))
                f.write(' \\\\\n')
            f.write('\\bottomrule\n')     # \bottomrule
            f.write('\\end{tabular}\n') # \end{tabular}
            f.write('\\caption{ }\n')   # \caption{ }
            f.write('\\label{ }\n')     # \label{ }
            f.write('\\end{table}')     # \end{table}