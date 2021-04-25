
import pandas as pd
import numpy as np
import os
import math
import sklearn
import xgboost
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from numpy.random import choice


class onePixel:
    def __init__(self, targetModel, weight):

        self.targetModel = targetModel
        self.weight = weight
        trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
        testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
        self.columns = trainingData[0].columns
        X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
        X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
        X = np.vstack((X_trainraw, X_testraw))
        y = np.hstack((y_trainraw, y_testraw))
        _, X_sample, _, y_sample = train_test_split(X, y, test_size=0.01, random_state=0, stratify=y)



    def perturb(self, X):
        for i in range(0,len(X)):
            f = choice(len(X[0]), 1, p=weight[i])
            feature = X[i][f]
            if type(feature) == str:
                feature = feature[::-1]
            else:
                feature = feature * (-1)
            X[i][f] = feature
        return self.targetModel.predict(X)


    def test_accuracy(self, X, y):
        print(f"Accuracy - Test: {accuracy_score(y, self.perturb(X))}")
        test_auc = roc_auc_score(y, self.targetModel.predict_proba(X)[:, 1])
        print(f"AUC - Test: {test_auc}")


if __name__ == '__main__':
    
       # dummy target model to test against
        trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
        testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
        X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
        X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
        X = np.vstack((X_trainraw, X_testraw))
        y = np.hstack((y_trainraw, y_testraw))
    


        rf = pickle.load(open(os.path.join("fitted_models","rf_0.929"),"rb"))
        weight=pickle.load(open("shap.txt","rb"))
        onePixel = onePixel(rf, weight)
        onePixel.test_accuracy(X,y)
