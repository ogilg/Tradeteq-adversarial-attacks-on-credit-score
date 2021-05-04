import tensorflow as tf
import pickle
import os
import numpy as np
import pandas as pd
from keras.models import model_from_json
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split

class FGSM:
    def __init__(self, model):
        self.vic_model = model  # the victim model
        self.loss_object = tf.keras.losses.BinaryCrossentropy()  # use binary cross-entropy
        trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
        self.columns = trainingData[0].columns

    def to_df_shape(self, input_X):
        input_X = np.reshape(input_X.numpy(), (input_X.shape[0],))
        input_X = pd.DataFrame([input_X], columns=self.columns)
        return input_X

    def create_adversarial_pattern(self, input_X, input_label):
        input_X = tf.Variable(input_X, dtype=tf.float32)
        input_X = tf.reshape(input_X, (1, 115))

        input_label = tf.one_hot(input_label, 2)
        input_label = tf.reshape(input_label, (1, 2))

        with tf.GradientTape() as tape:
            tape.watch(input_X)
            prediction = self.vic_model(input_X)
            loss = self.loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_X)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad


if __name__ == '__main__':
    # dummy target model to test against
    trainingData = pickle.load(open(os.path.join("Processing", "train_preproc.p"), "rb"))
    testData = pickle.load(open(os.path.join("Processing", "test_preproc.p"), "rb"))
    X_trainraw, y_trainraw = trainingData[0].to_numpy(), trainingData[1]
    X_testraw, y_testraw = testData[0].to_numpy(), testData[1]
    X = np.vstack((X_trainraw, X_testraw))
    y = np.hstack((y_trainraw, y_testraw))
    print(y[0])
    print("Data loading ready")

    xg = pickle.load(open(os.path.join("fitted_models", "rf_0.929.pkl"), "rb"))

    # load json and create model
    modelFileName = os.path.join("fitted_models", 'NN_Model_0.811.json')
    weightFileName = os.path.join("fitted_models", 'NN_weight_0.811.h5')
    json_file = open(modelFileName, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    nn = model_from_json(loaded_model_json)
    # load weights into new model
    nn.load_weights(weightFileName)
    print("Loaded model from disk")
    # We left here as the nn model needs to be callable in order to fix the gradient-tape problem in finding gradient
    print("Model reading ready")
    # Set the substitute victim model nn

    FGSM_attack = FGSM(nn)

    #Generate up to 100 adversarial samples by iterating through dataset

    adv_samples = []
    step_sizes = []
    for i in range(0, len(X)):
        if(len(adv_samples) == 100):
            print("Reached 100")
            break
        if(y[i] == 0 and xg.predict([X[i]])[0] == 0):
            perturbations = FGSM_attack.create_adversarial_pattern(X[i], y[i])
            ep = 0.1
            done = False
            while(not done and ep <= 1):
                adv_x = X[i] - ep * perturbations
                adv_label = xg.predict(adv_x)
                if(adv_label[0] == 1):
                    print(".")
                    adv_samples.append(adv_x.numpy()[0])
                    step_sizes.append(ep)
                    done = True
                else:
                    ep += 0.1
            

    with open("adv_samples.pkl", "wb") as f:
        pickle.dump(adv_samples, f)
    with open("step_sizes.pkl", "wb") as f:
        pickle.dump(step_sizes, f)

    print(len(adv_samples))