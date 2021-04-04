import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization
from keras import optimizers
import tensorflow as tf


class subModel:
    def __init__(self, targetModel):

        self.targetModel = targetModel
        # get stratified sample for jacobian dataset augmentation
        trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
        testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
        self.columns = trainingData[0].columns
        X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
        X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
        X = np.vstack((X_trainraw, X_testraw))
        y = np.hstack((y_trainraw, y_testraw))
        X = X.astype(np.float32)
        _, X_sample, _, y_sample = train_test_split(X, y, test_size=0.01, random_state=0, stratify=y,shuffle=True)

        self.NN_Model = self.Create_NN_Model(input_dim=X_trainraw.shape[1])
        self.train(X_sample, y_sample, 0.5, 2)

    def train(self, X, y, alpha, reps):  # alpha is step-size, reps is number of augmentations
        X = X.astype(np.float32)
        # we have changed some code here
        if (reps > 0):
            n = X.shape[1]

            # fit model
            self.NN_Model.fit(X, y, epochs = 10)

            augmentedX = pd.DataFrame(columns=self.columns)

            for i,x in enumerate(X):
                print(str(i) + "/"+str(len(X)),"reps:"+str(reps))
                # perform augmentation for each vector x in current dataset
                # print(x) x in form [x1,x2,...,x115]
                x = tf.Variable([x],dtype=tf.float32)
                with tf.GradientTape() as tape:
                    y_pred = x
                    for layer in self.NN_Model.layers:
                        y_pred = layer(y_pred)
                J = tape.jacobian(y_pred, x)
                # print(J) shape= (1,1,1,115)
                xNew = np.sign(J) * alpha + x
                xNew = np.reshape(xNew,(n,))
                #print(xNew)
                augmentedX = augmentedX.append(pd.DataFrame([xNew], columns=self.columns))
            # get class labels for new augmented vectors
            augmentedY = self.targetModel.predict(augmentedX)

            # append augmented vectors to dataset
            X = np.vstack((X, augmentedX))
            y = np.hstack((y, augmentedY))

            reps -= 1
            print("reps: "+str(reps))
            self.train(X, y, alpha, reps)


    def test_accuracy(self, X, y):
        X = np.asarray(X).astype('float32')
        y_pred_keras = self.NN_Model.predict(X).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, y_pred_keras)
        auc_keras = auc(fpr_keras, tpr_keras)
        print(f"AUC - Test: {auc_keras}")


    def Create_NN_Model(self, input_dim):
        opt = optimizers.Adam(lr=0.0001)  # hyper parameter
        METRICS = [tf.keras.metrics.AUC()]
        model = Sequential()
        model.add(BatchNormalization())
        # Try some weight initialization, with different weight
        model.add(Dense(10, input_dim=input_dim,
                        kernel_initializer='normal',
                        # kernel_regularizer=regularizers.l2(0.02),
                        activation="relu"))
        model.add(Dense(4,activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=METRICS)
        return model


if __name__ == '__main__':
    # dummy target model to test against
    xg = pickle.load(open(os.path.join("fitted_models", "rf_0.929.pkl"), "rb"))
    trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
    testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
    X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
    X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
    X = np.vstack((X_trainraw, X_testraw))
    y = np.hstack((y_trainraw, y_testraw))

    
    print("target model ready")

    subModel = subModel(xg)
    print("sub model trained")
    subModel.test_accuracy(X, y)
    # AUC test = 0.816