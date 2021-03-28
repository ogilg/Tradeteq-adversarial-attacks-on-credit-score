import pandas as pd
import numpy as np
<<<<<<< HEAD
import os
import math
from sklearn.metrics import accuracy_score, roc_auc_score
=======
import math
>>>>>>> origin/substitute_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

<<<<<<< HEAD
=======
#dummy target model to test against
trainingData = pickle.load(open("train_preproc.p", "rb"))
testData = pickle.load(open("test_preproc.p", "rb"))
X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
X = np.vstack((X_trainraw, X_testraw))
y = np.hstack((y_trainraw, y_testraw))
lr = LogisticRegression()
lr.fit(X, y)

>>>>>>> origin/substitute_model

class subModel:
	def __init__(self, targetModel):
		self.lrModel = LogisticRegression()
		self.targetModel = targetModel

		#get stratified sample for jacobian dataset augmentation
<<<<<<< HEAD
		trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
		testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
		self.columns = trainingData[0].columns
=======
		trainingData = pickle.load(open("train_preproc.p", "rb"))
		testData = pickle.load(open("test_preproc.p", "rb"))
>>>>>>> origin/substitute_model
		X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
		X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
		X = np.vstack((X_trainraw, X_testraw))
		y = np.hstack((y_trainraw, y_testraw))
<<<<<<< HEAD
		_, X_sample, _, y_sample = train_test_split(X, y, test_size=0.01, random_state=0, stratify=y)
		self.train(X_sample, y_sample, 1, 4)
=======
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0, stratify=y)

		self.train(X_test, y_test, 1, 4)
>>>>>>> origin/substitute_model

	def train(self, X, y, alpha, reps): #alpha is step-size, reps is number of augmentations
		if(reps > 0):
			n = X.shape[1]

			#fit model
			self.lrModel.fit(X, y)

			#get coefficients
			weightCoeffs = self.lrModel.coef_[0]
			intercept = self.lrModel.intercept_[0]
<<<<<<< HEAD
			augmentedX = pd.DataFrame(columns=self.columns)
=======
			augmentedX = []
>>>>>>> origin/substitute_model
			augmentedY = []

			for x in X:
				#perform augmentation for each vector x in current dataset
				jacobian = []
				e = math.exp(-intercept-np.dot(weightCoeffs, x))
				for i in range(0, n):
					partialDeriv = (weightCoeffs[i]*e)/((1+e)**2)
					jacobian.append(partialDeriv)
				xNew = np.sign(jacobian)*alpha + x
<<<<<<< HEAD
				augmentedX = augmentedX.append(pd.DataFrame([xNew], columns=self.columns))
			#get class labels for new augmented vectors
			augmentedY = self.targetModel.predict(augmentedX)


=======
				augmentedX.append(xNew)
				
			#get class labels for new augmented vectors
			augmentedY = self.targetModel.predict(augmentedX)

>>>>>>> origin/substitute_model
			#append augmented vectors to dataset
			X = np.vstack((X, augmentedX))
			y = np.hstack((y, augmentedY))
	
			reps -= 1
			self.train(X, y, alpha, reps)
<<<<<<< HEAD

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
=======
		


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
>>>>>>> origin/substitute_model
