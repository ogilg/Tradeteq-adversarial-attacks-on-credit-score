import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

#dummy target model to test against
trainingData = pickle.load(open("train_preproc.p", "rb"))
testData = pickle.load(open("test_preproc.p", "rb"))
X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
X = np.vstack((X_trainraw, X_testraw))
y = np.hstack((y_trainraw, y_testraw))
lr = LogisticRegression()
lr.fit(X, y)


class subModel:
	def __init__(self, targetModel):
		self.lrModel = LogisticRegression()
		self.targetModel = targetModel

		#get stratified sample for jacobian dataset augmentation
		trainingData = pickle.load(open("train_preproc.p", "rb"))
		testData = pickle.load(open("test_preproc.p", "rb"))
		X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
		X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
		X = np.vstack((X_trainraw, X_testraw))
		y = np.hstack((y_trainraw, y_testraw))
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0, stratify=y)

		self.train(X_test, y_test, 1, 4)

	def train(self, X, y, alpha, reps): #alpha is step-size, reps is number of augmentations
		if(reps > 0):
			n = X.shape[1]

			#fit model
			self.lrModel.fit(X, y)

			#get coefficients
			weightCoeffs = self.lrModel.coef_[0]
			intercept = self.lrModel.intercept_[0]
			augmentedX = []
			augmentedY = []

			for x in X:
				#perform augmentation for each vector x in current dataset
				jacobian = []
				e = math.exp(-intercept-np.dot(weightCoeffs, x))
				for i in range(0, n):
					partialDeriv = (weightCoeffs[i]*e)/((1+e)**2)
					jacobian.append(partialDeriv)
				xNew = np.sign(jacobian)*alpha + x
				augmentedX.append(xNew)
				
			#get class labels for new augmented vectors
			augmentedY = self.targetModel.predict(augmentedX)

			#append augmented vectors to dataset
			X = np.vstack((X, augmentedX))
			y = np.hstack((y, augmentedY))
	
			reps -= 1
			self.train(X, y, alpha, reps)
		


	def test_accuracy(self, X, y):
		count = 0
		success = 0
		for i in range(0, len(X)):
			if(self.lrModel.predict([X[i]])[0] == y[i]):
				success += 1
			count += 1
		return (success/count)*100

subModel = subModel(lr)
print(subModel.test_accuracy(X, y))