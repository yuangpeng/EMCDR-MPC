# -*- encoding: utf-8 -*-
'''
@File    :   utils.py.py    
@Contact :   initialwooo@gmai.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/4/1 20:51   wuyuhan      1.0         None
'''

import torch
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)