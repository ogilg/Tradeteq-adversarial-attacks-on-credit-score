
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
import matplotlib.pyplot as plt


class randomized:
    def __init__(self, targetModel, weight,n):

        self.targetModel = targetModel
        self.weight = weight
        self.n = n
        self.adv=[]
        trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
        testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
        self.columns = trainingData[0].columns
        X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
        X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
        X = np.vstack((X_trainraw, X_testraw))
        y = np.hstack((y_trainraw, y_testraw))


    def perturb(self, X, y):
        adv = []
        codomain = X.transpose()
        co = []
        for i in range(0, 115):
            co.append((codomain[i].min(), codomain[i].max()))
        n=self.n

        success=0
        for i in range(0,len(X)):
            list = choice(len(X[0]), n, replace=False, p=weight[i])
            if y[i]==0:
                it=0
                x = X[i]
                scale = 2
                print(scale)
                x_curr = x.copy()
                while it<1000 :
                    x_new = x.copy()
                    it=it+1
                    for f in list:
                        (l,u) = co[f]
                        t = (u - l) / scale
                        ft = x_new[f] + np.random.uniform(-t, t)
                        if ft > u: ft = u
                        if ft < l: ft = l
                        x_new[f]=ft
                    y_curr = self.targetModel.predict([x_new])[0]
                    if y_curr==1:
                        scale=scale*2
                        x_curr=x_new
                if scale>2: success=success+1
                x=x_curr
                print(i, scale) # scale > 2 implies that an adversarial example is generated
        self.adv=adv
        return self.targetModel.predict(X)

    def test_accuracy(self, X, y):
        d=self.perturb(X, y)
        print(f"Accuracy - Test: {accuracy_score(y, self.perturb(X,y))}")
        test_auc = roc_auc_score(y, self.targetModel.predict_proba(X)[:, 1])
        print(f"AUC - Test: {test_auc}")
        return test_auc


if __name__ == '__main__':
    
       # dummy target model to test against
        trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
        testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
        X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
        X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
        X = np.vstack((X_trainraw, X_testraw))
        y = np.hstack((y_trainraw, y_testraw))

        rf = pickle.load(open(os.path.join("fitted_models","rf_0.929"),"rb"))

        weight = pickle.load(open("weight.txt","rb"))
        print("loaded")
        random_rf = randomized(rf, weight, 40)
        random_rf.test_accuracy(X, y)
