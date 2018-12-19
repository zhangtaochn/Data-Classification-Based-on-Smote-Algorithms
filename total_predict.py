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
import numpy as np
from sklearn.metrics import classification_report


def loadFile(df):
    resultList = []
    f = open(df, 'r')
    for line in f:
        resultList.append(line)
    f.close()
    print('shape:', np.asarray(resultList, dtype=np.float32).shape)
    return np.asarray(resultList, dtype=np.float32)  # not necessary

Decision_tree_balanced = loadFile('result\Decision_tree_balanced.txt')
BPNN1 = loadFile('result\BPNN1.txt')
BPNN_PCA_smote = loadFile('result\BPNN_PCA_smote.txt')

# y_test is the true label
y_test = loadFile('result\y_test.txt')

predict_result = []
for i in range(len(y_test)):
    if Decision_tree_balanced[i] + BPNN1[i] + BPNN_PCA_smote[i] >1:
        predict_result.append(1)
    else:
        predict_result.append(0)

print(classification_report(y_test, predict_result, target_names=['-1', '1']))