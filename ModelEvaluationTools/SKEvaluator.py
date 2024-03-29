from ModelEvaluationTools.ModelEvaluator import ModelEvaluator
import pickle
import os

"""
Implements ModelEvaluator
Evaluates sk-learn models 
"""
class SKEvaluator(ModelEvaluator):

    def __init__(self, model, model_name):
        super().__init__(model)
        self.model_name = model_name

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.fitted = True

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        try:
            return self.model.predict_proba(X)[:,1]
        except AttributeError as e:
            print(e)
            print("Type of model is: ", self.type)

    """
    Saves model with AUC score in the name of the file
    Example file name: rf_0.85 for a random forest model with 0.85 AUC
    """
    def save_model(self):
        assert self.fitted
        main_dir = os.path.dirname(os.path.dirname(__file__))
        filename = os.path.join(main_dir, "fitted_models", self.model_name+'_'+str(round(self.most_recent_auc, 3)))
        pickle.dump(self.model, open(filename, 'wb'))
