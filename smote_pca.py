# ！/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© Copyright 2018 The Author. All Rights Reversed.
------------------------------------------------------------
File Name: 
Author : zhangtao 
Time:
Description：
------------------------------------------------------------
"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN


import BPNN
print(__doc__)

def loaddata(filepath):
    f = open(filepath, 'r')
    X = []
    label = []
    for line in f:
        line = line.rstrip('\n')
        sVals = line.split(' ')
        label.append(sVals[-1])
        X.append(sVals[0:10])

    X = np.array(X)
    X = X.astype(float)
    label = np.array(label)
    label = label.astype(float)

    return X, label

def processdata(traindata):
    resultList = []

    for line in traindata:
        # print('line', line)
        if line[-1] == 1:
            line = np.hstack([line[0:2],[0,1]])
        elif line[-1] == 0:
            line = np.hstack([line[0:2],[1,0]])
        else:
            print('error')

        fVals = list(map(np.float32, line))  # [1.0, 2.0, 3.0]

        resultList.append(fVals)  # [[1.0, 2.0, 3.0] , [4.0, 5.0, 6.0]]
    print('shape:', np.asarray(resultList, dtype=np.float32).shape)
    return np.asarray(resultList, dtype=np.float32)  # not necessary

def smote_pca_train(filepath):
    X, y = loaddata(filepath)
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 1
        else:
            y[i] = 0
    pca = PCA(n_components=2, copy=False)
    pca.fit(X)
    print('pca.explained_variance_ratio_', pca.explained_variance_ratio_)
    print('pca.explained_variance_', pca.explained_variance_)
    print('pca.n_components_', pca.n_components_)
    print(X)
    print(len(X), len(y))
    X = pca.transform(X)

    print(X, y)
    print(y)
    print(y.shape)

    print(type(X),type(y))
    sm = SMOTEENN()
    X_resampled, y_resampled = sm.fit_sample(X, y)
    court1 = 0
    for i in y_resampled:
        if i == 1:
            court1 += 1
    print('比例：', court1/len(y_resampled))

    plt.scatter(X_resampled[:, 0], X_resampled[:, 1], marker='.',s= 1, c=y_resampled)
    plt.legend()

    plt.axis('tight')
    plt.show()
    print(np.shape(X_resampled),np.shape(y_resampled))
    data = []
    for i in range(len(y_resampled)):
        data.append(np.hstack((X_resampled[i],y_resampled[i])))

    return processdata(data)

def smote_pca_test(filepath):
    X, y = loaddata(filepath)
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 1
        else:
            y[i] = 0
    pca = PCA(n_components=2, copy=False)
    pca.fit(X)
    print('pca.explained_variance_ratio_', pca.explained_variance_ratio_)
    print('pca.explained_variance_', pca.explained_variance_)
    print('pca.n_components_', pca.n_components_)
    print(X)
    print(len(X), len(y))
    X = pca.transform(X)

    print(X, y)
    print(y)
    print(y.shape)

    print(type(X),type(y))

    plt.scatter(X[:, 0], X[:, 1], marker='.', s=1, c=y)
    plt.legend()

    plt.axis('tight')
    plt.show()
    data = []
    for i in range(len(y)):
        data.append(np.hstack((X[i], y[i])))

    return processdata(data)

if __name__ == '__main__':
    smote_pca_train('train.data')




