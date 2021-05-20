import os
import pickle
import pandas as pd
import numpy as np
import math

import skopt
from skopt.plots import plot_convergence
from sklearn.model_selection import train_test_split

from processing.ModelWrapper import ModelWrapper
from processing.PostcodeEncoder import PostcodeEncoder

encoder = PostcodeEncoder()

# load training data
# used to find initial points to use, and to get column names
# also used to calculate standard deviation - switch this to a subset of data
data_dir = os.path.dirname(os.getcwd())
with open(os.path.join(data_dir, "Processing", r"train_preproc.p"), 'rb') as data_file:
    train_data = pickle.load(data_file)
X_train, y_train = train_data
# forgot in preprocessing: convert from bool to int
X_train['hasGNotice'] = X_train['hasGNotice'].apply(int)

# load sample of data to calculate standard deviation
_, X_sample, _, y_sample = train_test_split(X_train, y_train, test_size=0.01, random_state=0, stratify=y_train)


# the features derived from pcd - we don't attack these features directly
derived_features = ["oseast1m", "osnrth1m", "cty", "lat", "long", "ru11ind", "oac11",
                    "country", "oac1", "oac2", "imdu", "OtherCompInPcd"]

incomplete_columns = X_sample.columns.drop(derived_features)
incomplete_columns = incomplete_columns.insert(incomplete_columns.size, 'pcd')
#print(incomplete_columns)

# construct feature type lists
integer_features = []
for i in range(0,X_sample.dtypes.size):
    if(X_sample.dtypes[i] == np.dtype('int64')):
        integer_features.append(X_sample.dtypes.index[i])
        
boolean_features = []
for i in range(0,X_sample.dtypes.size):
    if (X_sample.dtypes[i] == np.dtype('int32')) or (X_sample.dtypes[i] == np.dtype('bool')) :
        boolean_features.append(X_sample.dtypes.index[i])

boolean_features.remove("eAccountsAccountCategory")

integer_features.append("eAccountsAccountCategory")

# categorical features that are not hierarchical in nature
cat_features = ["AccountsAccountCategory", "CompanyCategory", "RegAddressCountry", "pcd"]
cat_features_index = [incomplete_columns.get_loc(col_name) for col_name in cat_features]


# Finding min/max for categorical features:
min_max_cat = pd.DataFrame(index=["min", "max"], columns=cat_features)
for col_name in cat_features:
    if col_name == 'pcd':
        min_max_cat[col_name]["max"] = encoder.num_postcodes()
        min_max_cat[col_name]["min"] = 0
    else:
        min_max_cat[col_name]["max"] = X_sample[col_name].max(axis=0)
        min_max_cat[col_name]["min"] = X_sample[col_name].min(axis=0)

class BaysOpt():

    def __init__(self, model):
        self.model = model
        self.wrapper = ModelWrapper(model)
        self.sd = pd.DataFrame(index=["sd"], columns=X_sample.columns)
        for col_name in X_sample.columns:
            self.sd[col_name] = X_sample[col_name].std(axis=0)

    # takes original data point as dataframe with derived fields removed
    def construct_derived_search_space(self, original):
        print("constructing search space")
        search_space = list()

        for col_name in original.index:
            if col_name in cat_features:
                search_space.append(tuple(min_max_cat[col_name]))
            elif col_name in boolean_features:
                search_space.append((0,1))
            else:
                original_value = original[col_name]
                col_sd = self.sd.at["sd", col_name]
                # ensures that integers stay integers
                if col_name in integer_features:
                    lower = math.floor(original_value-0.1*col_sd)
                    upper = math.ceil(original_value+0.1*col_sd)
                else:
                    lower = original_value-0.1*col_sd
                    upper = original_value+0.1*col_sd
                #print(original_value)
                #print((lower, upper))
                search_space.append((lower, upper))
        #print(len(search_space))
        return search_space

    # standardise weighting of features in distance function
    # weight of 1 for categorical features
    weighting_vector = np.array([1/(col.mean()+0.01) for col in [X_sample[col_name] for col_name in incomplete_columns.drop('pcd')]])
    # add entry for pcd
    weighting_vector = np.append(weighting_vector, 1)
    for index in cat_features_index:
        weighting_vector[index] = 1

    def distance_function(self, X, company):
        d = abs(X-company)
        for index in cat_features_index:
            if(X[index] != company[index]):
                d[index] = 1
            else:
                d[index] = 0
        return (d @ self.weighting_vector)

    # objective function incorporating both failed_prob and delta - smoother function
    def objective(self, X, true_company):
       encoder = PostcodeEncoder()
       X_df = self.build_df(X)
       X_df['pcd'].iloc[0] = encoder.search_decode(X_df['pcd'].iloc[0])
       delta = self.distance_function(X, true_company)
       failed_prob = self.wrapper.predict_proba(X_df)[0,1]
       if failed_prob < 0.5:
           # model has been tricked
           obj = 10*failed_prob + delta
       else:
           # model has not been tricked
           obj = 100 * failed_prob + delta
       return obj

    def build_df(self, X):
        if(len(X) == len(X_sample.columns)):
           return pd.DataFrame(data=[X], columns=X_sample.columns)
        elif(len(X) == len(incomplete_columns)):
           return pd.DataFrame(data=[X], columns=incomplete_columns)
        else:
           raise ValueError("Objective function recieved an unexpected number of columns")

    def get_arr(X):
        return X.values[0]

    # idea: make objective optional? allows for experiments, just run the best one when
    # I find it
    def attack(self, starting_sample):
        x0 = list(starting_sample)

        def compute_objective(X):
            return self.objective(X, starting_sample)

        search_space = self.construct_derived_search_space(starting_sample)
        print("starting attack now")

        res = skopt.gp_minimize(compute_objective, search_space, x0 = x0,
                                y0 = compute_objective(x0), n_calls = 50)
        return res

