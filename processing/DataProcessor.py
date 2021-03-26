import pickle

import joblib
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from Processing.LogScaler import LogScaler
from Processing.LabelEncoderExt import LabelEncoderExt

zero_info_features = ["CompanyId", "CompanyNumber","CompanyName","imd"]
only_one_value_features = ["Filled", "LimitedPartnershipsNumGenPartners", "LimitedPartnershipsNumLimPartners",\
                          "Status20190701","CompanyStatus"]
complicated_features = ["RegAddressAddressLine1", "RegAddressAddressLine2", "RegAddressCareOf", "RegAddressCounty", \
                        "RegAddressPOBox", "RegAddressPostCode", "RegAddressPostTown","oa11", "PreviousName_1CompanyName"]
to_num_cols = ["AccountsAccountRefDay", "AccountsAccountRefMonth", "oac1", "Field282", "Field2815"] # convert Fields for speedup
to_str_cols = ["ru11ind"]
dates_to_convert=[["dAccountsTimeGap","dAccountsNextDueDate","dAccountsLastMadeUpDate"],\
              ["dConfStmtTimeGap","dConfStmtNextDueDate","dConfStmtLastMadeUpDate"],\
              ["dReturnsTimeGap","dReturnsNextDueDate","dReturnsLastMadeUpDate"]]

class DataProcessor:

    def __init__(self, data):
        self.df = data
        self.df = self._remove_columns(zero_info_features + only_one_value_features + complicated_features)
        self._extract_feature_types(self.df)
        self.clean_column_datatypes(to_num_cols, to_str_cols)

    def _remove_columns(self, cols):
        to_remove_cols = []
        # check that columns are actually present
        for col in cols:
            if col in self.df.columns:
                to_remove_cols.append(col)
        return self.df.drop(to_remove_cols, axis=1)

    def _extract_feature_types(self, df):
        self.cat_features = []
        self.exp_features = []
        self.numeric_features = []

        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                self.cat_features.append(col)
            else:
                if self._is_exponential_feature(df[col]):
                    self.exp_features.append(col)
                else:
                    self.numeric_features.append(col)


    def _convert_col_to(self, col_name, type_constructor):
        return list(map(lambda x: type_constructor(x) if x is not None else None, self.df[col_name].values))

    """
    Makes sure all columns are of a single type 
    """
    def clean_column_datatypes(self, to_num_cols, to_str_cols):
        for col in to_num_cols:
            self.df[col] = self._convert_col_to(col, float)

        # Change None to string 'Nan' so that it is encoded as string
        for cat in self.cat_features:
            self.df[cat] = self.df[cat].apply(lambda x: 'Nan' if x is None else x)

        # Some columns contained both strings and floats so we convert everything to str
        for col in to_str_cols:
            self.df[col] = self._convert_col_to(col, str)

    # adds time gap feature to specified date features
    def add_time_gap_features(self, df, dates_to_convert):
        for dateBundle in dates_to_convert:
            #Note here dateBundle in the format: ["NewDuration Name", "Post Name", "Prev Name"]
            df[dateBundle[0]] = df[dateBundle[1]] - df[dateBundle[2]]
        return df
    
    def add_others_in_pcd_feature(self, df):
        #Dealt with pcd feature
        #Generate the pcd dictionary for fast checking
        if "pcd" in df.columns:
            pcdDict = dict()
            for i in df["pcd"]:
                pcdDict[i] = pcdDict.get(i, 0) + 1
            df["OtherCompInPcd"] = df.apply(lambda row: pcdDict[row["pcd"]], axis = 1)
            df = df.drop(['pcd'], axis=1)
        return df

    def _is_exponential_feature(self, feature_col):
        if not is_numeric_dtype(feature_col) or feature_col.values[0] in [True, False] or feature_col.dropna().min() < 0:
            return False
        num_unique_values = len(feature_col.unique())
        # does the feature have enough different value 
        has_enough_unique_values = num_unique_values > 50
        has_big_range = (feature_col.max() - feature_col.dropna().min()) > 500
        return has_big_range and has_enough_unique_values
            
    def impute_nans(self, df):
        # for all numeric values, fill the mean instead 
        df = df.copy()
        for col in df:
            # select only integer or float dtypes
            if df[col].dtype in ("int", "float"):
                df[col] = df[col].fillna(df[col].mean())
        return df

    def create_column_operations(self, X, scale):
        col_operations = {}
        for cat_col in self.cat_features:
            col_operations[cat_col] = LabelEncoderExt().fit(X[cat_col])

        for exp_col in self.exp_features:
            col_operations[exp_col] = LogScaler(scale).fit(X[exp_col])

        for num_feature in self.numeric_features:
            col_operations[num_feature] = None

        return col_operations

    def get_train_test(self, data, test_size=0.2):
        # remove 'isfailed' column which is what we want to predict
        X, y = data.drop('isfailed', axis=1), data['isfailed'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=2020, stratify = y)

        return X_train, X_test, y_train, y_test
        
    def clean_data(self, df):
        df = self.add_time_gap_features(df, dates_to_convert)
        df = self.add_others_in_pcd_feature(df)
        assert 'pcd' not in df.columns

        return df

    def preprocess_data(self):
        X = self.clean_data(self.df)
        # load list of column operations
        col_operations = joblib.load('column_operations.pkl')
        # apply the transformations to the corresponding column
        for col in X.columns:
            if col not in col_operations.keys():
                print("Column from data not present in preprocessing pipeline")
                return
            if col_operations[col] is not None:
                X[col] = col_operations[col].transform(X[col])
            print(col)
        X = self.impute_nans(X)
        return X

    def build_column_transformers(self, scale = True):
        df = self.clean_data(self.df)
        X_train, X_test, y_train, y_test = self.get_train_test(df)
        self._extract_feature_types(X_train)
        assert 'pcd' not in self.cat_features
        # Create and fit the transformers for each column and store them in col_operations.pkl
        self.col_operations = self.create_column_operations(X_train, scale)
        joblib.dump(self.col_operations, 'column_operations.pkl')
        return X_train, X_test, y_train, y_test


