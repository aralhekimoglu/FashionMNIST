from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp

import torch

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def print_parameter_count(model):
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.view(-1).shape[0]
    if total_params//10**6!=0:
        str_b = 'M'
        total_params/=10**6
    elif total_params//10**3!=0:
        str_b = 'K'
        total_params/=10**3
    else:
        str_b = ''
    print("Total number of parameters: {:.2f} {}".format(total_params,str_b))