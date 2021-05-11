from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, roc_auc_score
"""
Abstract class for general model agnostic evaluator
"""
class ModelEvaluator(ABC):
    """
    model: untrained model
    """
    def __init__(self, model):
        self.model = model
        self.type = type(model)
        self.fitted = False

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    """"
    input: data matrix x
    output: prediction vector y
    """
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    """
    Evaluates model accuracy and AUC on test set
    returns AUC test score
    """
    def evaluate(self, X_test, y_test):
        print(f"Accuracy - Test: {accuracy_score(y_test, self.predict(X_test))}")
        test_auc = roc_auc_score(y_test, self.predict_proba(X_test))
        print(f"AUC - Test: {test_auc}")
        self.most_recent_auc = test_auc
        return test_auc
    """
    Saves model weights and parameters to some file in the fitted models folder
    Pre: model has been fitted and evaluated (enforced)
    """
    @abstractmethod
    def save_model(self):
        pass