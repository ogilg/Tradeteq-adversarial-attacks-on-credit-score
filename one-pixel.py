
import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from numpy.random import choice
import pickle
import random

class onePixel:
    def __init__(self, weight, targetModel):

        self.targetModel = targetModel
        self.weight=weight

        trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
        testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
        self.columns = trainingData[0].columns
        X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
        X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
        X = np.vstack((X_trainraw, X_testraw))
        y = np.hstack((y_trainraw, y_testraw))
        _, X_sample, _, y_sample = train_test_split(X, y, test_size=0.01, random_state=0, stratify=y)



    def perturb(self, X, weight):
        col = choice(range(0,len(X)), 1, weight)
        feature = X[col]
        if type(feature) == str:
            feature = feature[::-1]
        else:
            feature = feature * (-1)
        X[col] = feature
        return self.targetModel.predict(X)


    def test_accuracy(self, X, y, weight):
        print(f"Accuracy - Test: {accuracy_score(y, self.perturb(X, weight))}")
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
    

        xg = pickle.load(open(os.path.join("fitted_models", "xg_0.904"), "rb"))
        rf = pickle.load(open(os.path.join("fitted_models","rf_0.929"),"rb"))
        weight = pickle.load(open("importance.txt","rb"))
    
        onePixel = onePixel(xg,weight)
        onePixel.test_accuracy(X,y)
    
    