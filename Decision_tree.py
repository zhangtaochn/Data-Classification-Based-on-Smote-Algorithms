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
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

''''' 数据读入 '''
def readdata(filenpath):
    data = []
    labels = []
    with open(filenpath) as ifile:
        for line in ifile:
            tokens = line.strip().split(' ')
            data.append([float(tk) for tk in tokens[:-10]])
            labels.append(tokens[-1])
    x = np.array(data)
    print(x)
    labels = np.array(labels)
    print('label', labels)
    y = np.zeros(labels.shape)

    # ''''' 标签转换为0/1 '''
    y[labels == '1'] = 1

    return x, y
#, class_weight='balanced'



# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train, y_train = readdata(r'E:\Windows\Desktop\pro1\IrisData-master\train.data')
x_test, y_test = readdata(r'E:\Windows\Desktop\pro1\IrisData-master\test.data')
''''' 使用信息熵作为划分标准，对决策树进行训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
print(clf)
clf.fit(x_train, y_train)

''''' 把决策树结构写入文件 '''
with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
print(clf.feature_importances_)

'''''测试结果的打印'''
answer = clf.predict(x_test)
print(x_test)
print('answer:', answer)
print('y_test:', y_test)
print(np.mean(answer == y_test))


'''''准确率与召回率'''
precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))

print('precision, recall, thresholds:', precision, recall, thresholds)
#answer = clf.predict_proba(x_test)
answer = clf.predict(x_test)
print(classification_report(y_test, answer, target_names=['-1', '1']))

f = open("result\Decision_tree.txt", 'w+', encoding='utf8')
for i in answer:
    f.write(str(i)+'\n')
f.close()

print(y_test)

f = open("result\y_test.txt", 'w+', encoding='utf8')
for i in y_test:
    f.write(str(i)+'\n')
f.close()
