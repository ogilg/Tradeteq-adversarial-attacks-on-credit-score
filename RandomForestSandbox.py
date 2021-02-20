import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool


class ModelEvaluator:
    """
    model: untrained model
    """
    def __init__(self, model):
        
        self.model = model
        
    
    def fit(self, X_train, y_train):
        if type(self.model) == 'catboost.core.CatBoostClassifier':
            train_pool = Pool(X_train, y_train)
            self.model.fit(train_pool)
        else:
            self.model.fit(X_train, y_train)
            
    def evaluate_model(self, X_test, y_test):
        
    