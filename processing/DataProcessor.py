import pandas as pd
import numbers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from pandas.api.types import is_numeric_dtype

class DataProcessor:

    def __init__(self, data_file):
        self._fetch_data(data_file)

    def _fetch_data(self, data_file):
        self.df = pd.read_pickle(data_file)

    def remove_columns(self, cols):
        self.df = self.df.drop(cols, axis=1)

    def _is_numeric(self, col_name):
        i = 0
        column = self.df[col_name].values
        while i < len(column) and column[i] is None:
            i += 1
        if i == len(column):
            return "Cannot infer column type"
        return isinstance(column[i], numbers.Number)


    def _extract_categorical_feature_names(self):
        return [col_name for col_name in self.df.columns if not self._is_numeric(col_name)]

    def _convert_col_to(self, col_name, type_constructor):
        return list(map(lambda x: type_constructor(x) if x is not None else None, self.df[col_name].values))

    """
    Makes sure all columns are of a single type 
    """
    def clean_column_datatypes(self, to_num_cols, to_str_cols):
        for col in to_num_cols:
            self.df[col] = self._convert_col_to(col, float)

        # Change None to string 'Nan' so that it is encoded as string
        for cat in self._extract_categorical_feature_names():
            self.df[cat] = self.df[cat].apply(lambda x: 'Nan' if x is None else x)

        # Some columns contained both strings and floats so we convert everything to str
        for col in to_str_cols:
            self.df[col] = self._convert_col_to(col, str)
       
    def ReplaceDateFeatures(self,data_convert):
        for dateBundle in data_convert:
            #Note here dateBundle in the format: ["NewDuration Name", "Post Name", "Prev Name"]
            self.df[dateBundle[0]] = self.df[dateBundle[1]] - self.df[dateBundle[2]]
    
    def PcdFeature(self):
        #Dealt with pcd feature
        #Generate the pcd dictionary for fast checking
        pcdDict = dict()
        for i in self.df["pcd"]:
            pcdDict[i] = pcdDict.get(i, 0) + 1
        self.df["OtherCompInPcd"] = self.df.apply (lambda row: pcdDict[row["pcd"]],axis = 1)
        self.remove_columns("pcd")
    
    def _is_exponential_feature(self,feature_col):
        if not is_numeric_dtype(feature_col) or feature_col.values[0] in [True, False] or feature_col.dropna().min() < 0:
            return False
        num_nans = feature_col.isnull().sum()
        num_unique_values = len(feature_col.unique())
        # does the feature have enough different value 
        has_enough_unique_values = num_unique_values > 50
        has_big_range = (feature_col.max() - feature_col.dropna().min()) > 500
        return has_big_range and has_enough_unique_values
    
    def ExponentialData(self):
        #Assumption: all the exponential column are the one starting with F
        exponential_features = [col_name for col_name in self.df.columns if self._is_exponential_feature(self.df[col_name])]
        #Assertion of the picking as numerical data.
        for feature in exponential_features:
        # This part works by approximating the log value
        # plus 1 to approximate the data in order to avoid divison by 0 error
            self.df[feature]=np.log10(self.df[feature]+1)
            
    def fillNan(self):
        # for all numeric values, fill the mean instead 
        self.df = self.df.copy()
        for col in self.df:
            # select only integer or float dtypes
            if self.df[col].dtype in ("int", "float"):
                self.df[col] = self.df[col].fillna(self.df[col].mean())
       
    
    def encode_data(self):
        self.le = LabelEncoder()
        cat_features = self._extract_categorical_feature_names()
        non_cat_features_df = self.df.drop(cat_features, axis=1)
        cat_features_encoded = self.df[cat_features].apply(self.le.fit_transform)
        encoded_data = pd.concat([non_cat_features_df, cat_features_encoded], axis=1)
        encoded_data = encoded_data.reset_index(drop=True)
        return encoded_data

    def get_train_test(self, encoded_data, test_size=0.2):
        # remove 'isfailed' column which is what we want to predict
        X, y = encoded_data.drop('isfailed', axis=1), encoded_data['isfailed'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=2020)

        return X_train, X_test, y_train, y_test
    
    
    
    def storePickle(self,name):
        # Store the current pandas form in pickle
        with open(name, 'wb') as f:
            pickle.dump(self.df, f)
        
    def preprocess_data(self, to_num_cols, to_str_cols,date_convert):
        self.clean_column_datatypes(to_num_cols, to_str_cols)
        self.ReplaceDateFeatures(date_convert)
        self.PcdFeature()
        self.ExponentialData()
        self.fillNan()
        self.storePickle(name = "Mod_Co_600K_Jul2019_6M.pkl")
        encoded_data = self.encode_data()
        X_train, X_test, y_train, y_test = self.get_train_test(encoded_data)
        return X_train, X_test, y_train, y_test
