from __future__ import division
from __future__ import print_function

import os
import sys
from time import time


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager

# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP

import pandas as pd
import os

import pickle
from tqdm import tqdm
import warnings
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")


# 预处理数据
def pre_process(data):
    data.drop(['point.id', 'motherset', 'origin','diff.score'], axis=1, inplace=True)
    data['ground.truth'] = data['ground.truth'].map({'nominal': 0, 'anomaly': 1})

    data.drop('original.label', inplace=True, axis=1)


    y = data['ground.truth']
    X = data.drop('ground.truth', axis=1)

    return X, y



# 定义分类器
def load_classifiers(outliers_fraction):
    outliers_fraction = min(0.5, outliers_fraction)
    random_state = np.random.RandomState(42)
    # Define nine outlier detection tools to be compared
    classifiers = {
        'Angle-based Outlier Detector (ABOD)':
            ABOD(contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor (CBLOF)':
            CBLOF(contamination=outliers_fraction,
                  check_estimator=False, random_state=random_state),
        'Feature Bagging':
            FeatureBagging(LOF(n_neighbors=35),
                           contamination=outliers_fraction,
                           random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS(
            contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,
                                    random_state=random_state, behaviour="new"),
        'K Nearest Neighbors (KNN)': KNN(
            contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',
                           contamination=outliers_fraction),
        'Local Outlier Factor (LOF)':
            LOF(n_neighbors=35, contamination=outliers_fraction),
        'Minimum Covariance Determinant (MCD)': MCD(
            contamination=outliers_fraction, random_state=random_state),
        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
        'Principal Component Analysis (PCA)': PCA(
            contamination=outliers_fraction, random_state=random_state)
    }
    return classifiers


# 训练和测试

PATH = "benchmarks"
fns = os.listdir(PATH)

if not os.path.exists('result'):
    os.mkdir('result')

for i in tqdm(range(len(fns))):
    fn = fns[i]

    if os.path.exists('result/{}'.format(fn)):
        continue

    data = pd.read_csv(os.path.join(PATH, fn))
    X, y = pre_process(data)

    outliers_fraction = y.sum() / len(y)
    classifiers = load_classifiers(outliers_fraction)

    all_scores = {}
    for i, (name, clf) in enumerate(classifiers.items()):
        try:
            clf.fit(X)
            y_pred = clf.predict(X)

            scores = []
            scores.append(roc_auc_score(y, y_pred))
            scores.append(f1_score(y, y_pred))
            scores.append(precision_score(y, y_pred))
            scores.append(recall_score(y, y_pred))
            all_scores[name] = scores
        except Exception as e:
            pass

    with open('result/{}'.format(fn), 'wb') as f:
        pickle.dump(all_scores, f)
