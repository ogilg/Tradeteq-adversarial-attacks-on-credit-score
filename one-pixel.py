
import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

class onepixel:
    def _init_(self, targetModel):
        self.lrModel=LogisticRegression()
        self.targetModel = targetModel
        trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
		testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
		self.columns = trainingData[0].columns
		X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
		X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
		X = np.vstack((X_trainraw, X_testraw))
		y = np.hstack((y_trainraw, y_testraw))
		_, X_sample, _, y_sample = train_test_split(X, y, test_size=0.01, random_state=0, stratify=y)
		self.train(X_sample, y_sample, 1, 4)
        
    
    def test_accuracy(self, X, y):
        print(f"Accuracy - Test: {accuracy_score(y, self.lrModel.predict(X))}")
        test_auc = roc_auc_score(y, self.lrModel.predict_proba(X)[:, 1])
        print(f"AUC - Test: {test_auc}")

    def train(self, X, y, num_iters=10000, epsilon=0.2, targeted=False):
        
        
        
if _name_== '__main__':
    
    # dummy target model to test against
    trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
    testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
    X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
    X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
    X = np.vstack((X_trainraw, X_testraw))
    y = np.hstack((y_trainraw, y_testraw))
    
    # lr = LogisticRegression(max_iter=10000)
    # lr.fit(X, y)
    xg = pickle.load(open(os.path.join("fitted_models", "xg_0.927.pkl"), "rb"))
    
    
    
    
