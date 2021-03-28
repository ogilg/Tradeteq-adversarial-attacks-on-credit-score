import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


class subModel:
	def __init__(self, targetModel):
		self.lrModel = LogisticRegression()
		self.targetModel = targetModel

		#get stratified sample for jacobian dataset augmentation
		trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
		testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
		self.columns = trainingData[0].columns
		X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
		X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
		X = np.vstack((X_trainraw, X_testraw))
		y = np.hstack((y_trainraw, y_testraw))
		_, X_sample, _, y_sample = train_test_split(X, y, test_size=0.01, random_state=0, stratify=y)
		self.train(X_sample, y_sample, 1, 4)

	def train(self, X, y, alpha, reps): #alpha is step-size, reps is number of augmentations
		if(reps > 0):
			n = X.shape[1]

			#fit model
			self.lrModel.fit(X, y)

			#get coefficients
			weightCoeffs = self.lrModel.coef_[0]
			intercept = self.lrModel.intercept_[0]
			augmentedX = pd.DataFrame(columns=self.columns)

			for x in X:
				#perform augmentation for each vector x in current dataset
				jacobian = []
				e = math.exp(-intercept-np.dot(weightCoeffs, x))
				for i in range(0, n):
					partialDeriv = (weightCoeffs[i]*e)/((1+e)**2)
					jacobian.append(partialDeriv)
				xNew = np.sign(jacobian)*alpha + x
				augmentedX = augmentedX.append(pd.DataFrame([xNew], columns=self.columns))
			#get class labels for new augmented vectors
			augmentedY = self.targetModel.predict(augmentedX)


			#append augmented vectors to dataset
			X = np.vstack((X, augmentedX))
			y = np.hstack((y, augmentedY))
	
			reps -= 1
			self.train(X, y, alpha, reps)

	def test_accuracy(self, X, y):
		print(f"Accuracy - Test: {accuracy_score(y, self.lrModel.predict(X))}")
		test_auc = roc_auc_score(y, self.lrModel.predict_proba(X)[:, 1])
		print(f"AUC - Test: {test_auc}")


if __name__ == '__main__':
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

	print("target model ready")

	subModel = subModel(xg)
	print("sub model trained")
	subModel.test_accuracy(X, y)