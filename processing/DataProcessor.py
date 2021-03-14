import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataProcessor:

    def __init__(self, data_file):
        self._fetch_data(data_file)

    def _fetch_data(self, data_file):
        self.df = pd.read_pickle(data_file)

    def remove_columns(self, cols):
        self.df = self.df.drop(cols, axis=1)

    def _extract_categorical_feature_names(self):
        return [col_name for col_name in self.df.columns if not is_numeric_dtype(col_name)]

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


    def preprocess_data(self, to_num_cols, to_str_cols):
        self.clean_column_datatypes(to_num_cols, to_str_cols)
        encoded_data = self.encode_data()
        X_train, X_test, y_train, y_test = self.get_train_test(encoded_data)
        return X_train, X_test, y_train, y_test
