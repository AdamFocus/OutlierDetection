from __future__ import division
from __future__ import print_function

import pandas as pd
import os
import sys
from time import time
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pickle
from tqdm import tqdm
import warnings

PATH = "result"
fns = os.listdir(PATH)

result=[]

for i in range(len(fns)):
    try:
        fn = fns[i]
        with open(os.path.join('result', fn), 'rb') as f:
            score = pickle.load(f)
        
        for k, v in score.items():
            row = [fn, k] + v
            result.append(row)
    except Exception as e:
        pass
result = pd.DataFrame(result, columns=['filename', 'models', 'roc', 'f1_score', 'precision', 'recall'])
result.to_csv('result.csv', index=False)

# 计算每个模型在所有文件上的指标平均值
roc = result.groupby('models')['roc', 'f1_score', 'precision', 'recall'].apply(lambda x: x.mean())
# 按ROC值排序
roc = roc.sort_values(by='roc')

roc.plot.barh(figsize=(8, 7))
plt.show()