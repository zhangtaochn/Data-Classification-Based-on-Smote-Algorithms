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
import random
import math
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import smote_pca

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


def loadFile(df):
    # load a comma-delimited text file into an np matrix
    resultList = []
    f = open(df, 'r')

    i = -50000
    for line in f:
        i += 1
        if i == 500:
            break
        line = line.rstrip('\n')
        sVals = line.split(' ')
        if sVals[-1] == '1':
            sVals = sVals[:-8] + ['0', '1']
        elif sVals[-1] == '-1':
            sVals = sVals[:-8] + ['1', '0']
        else:
            print('error')

        fVals = list(map(np.float32, sVals))  # [1.0, 2.0, 3.0]

        resultList.append(fVals)  # [[1.0, 2.0, 3.0] , [4.0, 5.0, 6.0]]
    f.close()
    print('shape:', np.asarray(resultList, dtype=np.float32).shape)
    return np.asarray(resultList, dtype=np.float32)  # not necessary


# end loadFile

def showVector(v, dec):
    fmt = "%." + str(dec) + "f"  # like %.4f
    for i in range(len(v)):
        x = v[i]
        if x >= 0.0: print(' ', end='')
        print(fmt % x + '  ', end='')
    print('')


def showMatrix(m, dec):
    fmt = "%." + str(dec) + "f"  # like %.4f
    for i in range(len(m)):
        for j in range(len(m[i])):
            x = m[i, j]
            if x >= 0.0: print(' ', end='')
            print(fmt % x + '  ', end='')
        print('')


def showMatrixPartial(m, numRows, dec, indices):
    fmt = "%." + str(dec) + "f"  # like %.4f
    lastRow = len(m) - 1
    width = len(str(lastRow))
    for i in range(numRows):
        if indices == True:
            print("[", end='')
            print(str(i).rjust(width), end='')
            print("] ", end='')

        for j in range(len(m[i])):
            x = m[i, j]
            if x >= 0.0: print(' ', end='')
            print(fmt % x + '  ', end='')
        print('')
    print(" . . . ")

    if indices == True:
        print("[", end='')
        print(str(lastRow).rjust(width), end='')
        print("] ", end='')
    for j in range(len(m[lastRow])):
        x = m[lastRow, j]
        if x >= 0.0: print(' ', end='')
        print(fmt % x + '  ', end='')
    print('')


# -----

class NeuralNetwork:

    def __init__(self, numInput, numHidden, numOutput, seed):
        self.ni = numInput
        self.nh = numHidden
        self.no = numOutput

        self.iNodes = np.zeros(shape=[self.ni], dtype=np.float32)
        self.hNodes = np.zeros(shape=[self.nh], dtype=np.float32)
        self.oNodes = np.zeros(shape=[self.no], dtype=np.float32)

        self.ihWeights = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)
        self.hoWeights = np.zeros(shape=[self.nh, self.no], dtype=np.float32)

        self.hBiases = np.zeros(shape=[self.nh], dtype=np.float32)
        self.oBiases = np.zeros(shape=[self.no], dtype=np.float32)

        self.rnd = random.Random(seed)  # allows multiple instances
        self.initializeWeights()

    def setWeights(self, weights):
        if len(weights) != self.totalWeights(self.ni, self.nh, self.no):
            print("Warning: len(weights) error in setWeights()")

        idx = 0
        for i in range(self.ni):
            for j in range(self.nh):
                self.ihWeights[i, j] = weights[idx]
                idx += 1

        for j in range(self.nh):
            self.hBiases[j] = weights[idx]
            idx += 1

        for j in range(self.nh):
            for k in range(self.no):
                self.hoWeights[j, k] = weights[idx]
                idx += 1

        for k in range(self.no):
            self.oBiases[k] = weights[idx]
            idx += 1

    def getWeights(self):
        tw = self.totalWeights(self.ni, self.nh, self.no)
        result = np.zeros(shape=[tw], dtype=np.float32)
        idx = 0  # points into result

        for i in range(self.ni):
            for j in range(self.nh):
                result[idx] = self.ihWeights[i, j]
                idx += 1

        for j in range(self.nh):
            result[idx] = self.hBiases[j]
            idx += 1

        for j in range(self.nh):
            for k in range(self.no):
                result[idx] = self.hoWeights[j, k]
                idx += 1

        for k in range(self.no):
            result[idx] = self.oBiases[k]
            idx += 1

        return result

    def initializeWeights(self):
        numWts = self.totalWeights(self.ni, self.nh, self.no)
        wts = np.zeros(shape=[numWts], dtype=np.float32)
        lo = -0.01
        hi = 0.01
        for idx in range(len(wts)):
            wts[idx] = (hi - lo) * self.rnd.random() + lo
        self.setWeights(wts)

    def computeOutputs(self, xValues):
        hSums = np.zeros(shape=[self.nh], dtype=np.float32)
        oSums = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(self.ni):
            self.iNodes[i] = xValues[i]

        for j in range(self.nh):
            for i in range(self.ni):
                hSums[j] += self.iNodes[i] * self.ihWeights[i, j]

        for j in range(self.nh):
            hSums[j] += self.hBiases[j]

        for j in range(self.nh):
            self.hNodes[j] = self.hypertan(hSums[j])

        for k in range(self.no):
            for j in range(self.nh):
                oSums[k] += self.hNodes[j] * self.hoWeights[j, k]

        for k in range(self.no):
            oSums[k] += self.oBiases[k]

        softOut = self.softmax(oSums)
        for k in range(self.no):
            self.oNodes[k] = softOut[k]

        result = np.zeros(shape=self.no, dtype=np.float32)
        for k in range(self.no):
            result[k] = self.oNodes[k]

        return result

    def train(self, trainData, maxEpochs, learnRate):
        hoGrads = np.zeros(shape=[self.nh, self.no], dtype=np.float32)  # hidden-to-output weights gradients
        obGrads = np.zeros(shape=[self.no], dtype=np.float32)  # output node biases gradients
        ihGrads = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)  # input-to-hidden weights gradients
        hbGrads = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden biases gradients

        oSignals = np.zeros(shape=[self.no], dtype=np.float32)  # output signals: gradients w/o assoc. input terms
        hSignals = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden signals: gradients w/o assoc. input terms

        epoch = 0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)
        numTrainItems = len(trainData)
        indices = np.arange(numTrainItems)  # [0, 1, 2, . . n-1]  # rnd.shuffle(v)

        while epoch < maxEpochs:
            self.rnd.shuffle(indices)  # scramble order of training items
            for ii in range(numTrainItems):
                idx = indices[ii]

                for j in range(self.ni):
                    x_values[j] = trainData[idx, j]  # get the input values
                for j in range(self.no):
                    t_values[j] = trainData[idx, j + self.ni]  # get the target values
                self.computeOutputs(x_values)  # results stored internally

                # 1. compute output node signals
                for k in range(self.no):
                    derivative = (1 - self.oNodes[k]) * self.oNodes[k]  # softmax
                    oSignals[k] = derivative * (self.oNodes[k] - t_values[k])  # E=(t-o)^2 do E'=(o-t)

                # 2. compute hidden-to-output weight gradients using output signals
                for j in range(self.nh):
                    for k in range(self.no):
                        hoGrads[j, k] = oSignals[k] * self.hNodes[j]

                # 3. compute output node bias gradients using output signals
                for k in range(self.no):
                    obGrads[k] = oSignals[k] * 1.0  # 1.0 dummy input can be dropped

                # 4. compute hidden node signals
                for j in range(self.nh):
                    sum = 0.0
                    for k in range(self.no):
                        sum += oSignals[k] * self.hoWeights[j, k]
                    derivative = (1 - self.hNodes[j]) * (1 + self.hNodes[j])  # tanh activation
                    hSignals[j] = derivative * sum

                # 5 compute input-to-hidden weight gradients using hidden signals
                for i in range(self.ni):
                    for j in range(self.nh):
                        ihGrads[i, j] = hSignals[j] * self.iNodes[i]

                # 6. compute hidden node bias gradients using hidden signals
                for j in range(self.nh):
                    hbGrads[j] = hSignals[j] * 1.0  # 1.0 dummy input can be dropped

                # update weights and biases using the gradients

                # 1. update input-to-hidden weights
                for i in range(self.ni):
                    for j in range(self.nh):
                        delta = -1.0 * learnRate * ihGrads[i, j]
                        self.ihWeights[i, j] += delta

                # 2. update hidden node biases
                for j in range(self.nh):
                    delta = -1.0 * learnRate * hbGrads[j]
                    self.hBiases[j] += delta

                    # 3. update hidden-to-output weights
                for j in range(self.nh):
                    for k in range(self.no):
                        delta = -1.0 * learnRate * hoGrads[j, k]
                        self.hoWeights[j, k] += delta

                # 4. update output node biases
                for k in range(self.no):
                    delta = -1.0 * learnRate * obGrads[k]
                    self.oBiases[k] += delta

            epoch += 1

            if epoch % 10 == 0:
                mse = self.meanSquaredError(trainData)
                print("epoch = " + str(epoch) + " error = %0.4f " % mse)
        # end while

        result = self.getWeights()
        return result

    # end train

    def accuracy(self, tdata):  # train or test data matrix
        temp_list = []

        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(len(tdata)):  # walk thru each data item
            for j in range(self.ni):  # peel off input values from curr data row
                x_values[j] = tdata[i, j]
            for j in range(self.no):  # peel off tareget values from curr data row
                t_values[j] = tdata[i, j + self.ni]

            y_values = self.computeOutputs(x_values)  # computed output values)
            # print('y_values', y_values)
            if y_values[0] > y_values[1]:
                temp_list.append(0)
            else:
                temp_list.append(1)
        return temp_list

    def meanSquaredError(self, tdata):  # on train or test data matrix
        sumSquaredError = 0.0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)

        for ii in range(len(tdata)):  # walk thru each data item
            for jj in range(self.ni):  # peel off input values from curr data row
                x_values[jj] = tdata[ii, jj]
            for jj in range(self.no):  # peel off tareget values from curr data row
                t_values[jj] = tdata[ii, jj + self.ni]

            y_values = self.computeOutputs(x_values)  # computed output values

            for j in range(self.no):
                err = t_values[j] - y_values[j]
                sumSquaredError += err * err  # (t-o)^2

        return sumSquaredError / len(tdata)

    @staticmethod
    def hypertan(x):
        if x < -20.0:
            return -1.0
        elif x > 20.0:
            return 1.0
        else:
            return math.tanh(x)

    @staticmethod
    def softmax(oSums):
        result = np.zeros(shape=[len(oSums)], dtype=np.float32)
        m = max(oSums)
        divisor = 0.0
        for k in range(len(oSums)):
            divisor += math.exp(oSums[k] - m)
        for k in range(len(result)):
            result[k] = math.exp(oSums[k] - m) / divisor
        return result

    @staticmethod
    def totalWeights(nInput, nHidden, nOutput):
        tw = (nInput * nHidden) + (nHidden * nOutput) + nHidden + nOutput
        return tw


# end class NeuralNetwork

def main(traindataMatrix, testDataMatrix):

    trainDataMatrix = processdata(traindataMatrix)
    print("\nBegin NN back-propagation \n")
    numInput = 2
    numHidden = 7
    numOutput = 2
    print("\nCreating a %d-%d-%d neural network " %
          (numInput, numHidden, numOutput))
    nn = NeuralNetwork(numInput, numHidden, numOutput, seed=3)
    print("\nTrain data: ")
    showMatrixPartial(trainDataMatrix, 3, 6, True)
    print("\nTest data: ")
    showMatrixPartial(testDataMatrix, 3, 6, True)

    maxEpochs = 50
    learnRate = 0.005
    print("\nSetting maxEpochs = " + str(maxEpochs))
    print("Setting learning rate = %0.3f " % learnRate)
    print("\nStarting training")
    nn.train(trainDataMatrix, maxEpochs, learnRate)
    print("Training complete")
    accTest = nn.accuracy(testDataMatrix)
    list_test = []
    for num in testDataMatrix:
        temp = list(num[-2:])
        if temp == [0.0, 1.0]:
            list_test.append(1)
        elif temp == [1.0, 0.0]:
            list_test.append(0)
        else:
            print('error')
    print('\n')
    # print('accTest', accTest)
    print(len(list_test), len(accTest))

    print(classification_report(list_test, accTest, target_names=['-1', '1']))

    f = open("result\BPNN.txt", 'w+', encoding='utf8')
    for i in range(len(accTest)):
        f.write(str(accTest[i]) + '\n')
    f.close()



if __name__ == "__main__":

    trainDataMatrix = loadFile('train.data')
    testDataMatrix = loadFile('test.data')
    main(trainDataMatrix, testDataMatrix)

