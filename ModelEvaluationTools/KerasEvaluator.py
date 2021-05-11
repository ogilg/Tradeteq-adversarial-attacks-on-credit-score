from ModelEvaluationTools.ModelEvaluator import ModelEvaluator
import pickle
import os
import numpy as np

"""
Implements ModelEvaluator
Evaluates keras models 
"""

class KerasEvaluator(ModelEvaluator):

    def __init__(self, model, model_name):
        super().__init__(model)
        self.model_name = model_name

    def fit(self, X_train, y_train, batch_size=None, epochs=1,
             validation_data=None, shuffle=True):
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data, shuffle=shuffle)
        self.fitted = True

    def predict(self, X):
        pred = self.model.predict(X)
        print(pred)
        arr = pred.ravel()
        print(arr)
        return np.rint(arr)

    def predict_proba(self, X):
        return self.model.predict(X)

    """
    Saves model with AUC score in the name of the file
    Example file name: rf_0.85 for a random forest model with 0.85 AUC
    """
    def save_model(self):
        assert self.fitted
        filename = os.path.join("fitted_models", self.model_name+'_'+str(round(self.most_recent_auc, 3)))
        self.model.save(filename)
