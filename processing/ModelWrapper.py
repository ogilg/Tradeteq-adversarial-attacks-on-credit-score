import os
import pandas as pd
# prevents warnings from not using .loc
pd.options.mode.chained_assignment = None
import numpy as np
import pickle
import joblib
from processing.PostcodeEncoder import format_pcd


"""
these are the fields we will derive from the postcode
["oseast1m", "osnrth1m", "cty", "lat", "long", "ru11ind", "oac11",
"country", "oac1", "oac2", "imdu", "OtherCompInPcd"]
"""


if __name__=="main":
    data_dir = os.cwd()
else:
    data_dir = os.path.dirname(__file__)

ons_data_folder = os.path.join(os.path.dirname(data_dir), "client_start_folder", "ONS", "data")
NSPL_file = "NSPL_AUG_2019_UK.csv"
ons_data = pd.read_csv(os.path.join(ons_data_folder, NSPL_file),
                       usecols=["pcd", "oseast1m", "osnrth1m", "cty", "oa11",
                                "lat", "long", "ru11ind", "oac11",
                                "imd"])

with open(os.path.join(data_dir, "feature_types.p"), "rb") as file:
    cat, num, bool = pickle.load(file)

# Import data that has already been preprocessed
with open(os.path.join(data_dir, r"train_data.p"), 'rb') as data_file:
    train_data = pickle.load(data_file)
X_train, y_train = train_data[0], train_data[1]

with open(os.path.join(data_dir, r"test_data.p"), 'rb') as data_file:
    test_data = pickle.load(data_file)
X_test, y_test = test_data[0], test_data[1]

col_ops = joblib.load(os.path.join(data_dir, "column_operations.pkl"))

class ModelWrapper:

    def __init__(self, model):
        self.model = model
        # store order of features
        # maybe not the right way to do this - better to get this from the model somehow?
        self.columns = X_train.columns.tolist()
        #self.columns.remove('pcd')

    def predict(self, X):
        return self.model.predict(self._complete_data(X)[self.columns])

    def predict_proba(self, X):
        return self.model.predict_proba(self._complete_data(X)[self.columns])

    # takes incomplete data with pcd and completes it
    def _complete_data(self, X):
        # round booleans to 0 or 1
        X = round_booleans(X).reset_index(drop=True)
        if len(X.columns) != len(self.columns) or X.columns != self.columns:
            try:
                derived_df = self._construct_derived_df(X["pcd"])
                X = pd.concat([X, derived_df], axis=1)
            except Exception as error:
                print("Data fill-in process is not compatible with given data")
                raise error
        if "pcd" in X.columns:
            X = X.drop(["pcd"], axis=1)
        return X

    def _construct_derived_df(self, X_pcd):
        # postcode no longer needs decoding - done already
        # decoded_pcd = col_ops["pcd"].label_encoder.inverse_transform(X_pcd.astype(int))
        # create df with derived fileds from ons data
        derived_df = []
        for pcd in X_pcd:
            derived_df.append(self._pcd_to_derived(pcd))
        derived_df = pd.concat(derived_df)
        # equalise imd into universal imd ("imdu") across different countries
        derived_df = self._add_imdu_col(derived_df)
        derived_df = derived_df.drop(["oa11", "imd"], axis=1)
        # reencode the derived fields
        derived_df = self._encode_derived(derived_df).reset_index(drop=True)
        return derived_df

    def _pcd_to_derived(self, pcd):
        derived = ons_data.loc[ons_data["pcd"] == pcd]
        print(derived)
        derived["country"] = derived.cty.map(cty_map)
        derived["cty"] = derived.cty.map(lambda x: "Nan" if pd.isnull(x) else x)
        derived["oac11"] = derived.oac11.map(lambda x: "Nan" if pd.isnull(x) else x)
        derived["oac1"] = derived.oac11.map(lambda x: "Nan" if (x == "Nan") else x[0])
        derived["oac2"] = derived.oac11.map(lambda x: "Nan" if (x == "Nan") else x[0:2])
        derived["ru11ind"] = derived.ru11ind.map(lambda x: "Nan" if pd.isnull(x) else x)
        # fix for oseast1m, osnrth1m sometimes being NaN
        # if we put the example through preprocessing, the pipeline would take care of this
        # numbers are the mean of each from the testing data
        oseastmean = 456447.45768833335
        osnrthmean = 271566.95671666664
        derived['oseast1m'] = derived.oseast1m.map(lambda x: oseastmean if np.isnan(x) 
                                                     else x)
        derived['osnrth1m'] = derived.osnrth1m.map(lambda x: osnrthmean if np.isnan(x) 
                                                     else x)
        try:
            derived["OtherCompInPcd"] = pcdDict[pcd]
        except KeyError:
            derived["OtherCompInPcd"] = 0
        print(derived)
        return derived

    # output argument
    def _add_imdu_col(self, derived):
        ctylist = "ENWS"
        imdu = []
        for i in range(0, derived.shape[0]):
            ctyidx = ctylist.index(derived.iloc[i]["country"])
            imdu.append(regr[ctyidx, 0] + regr[ctyidx, 1] * derived.iloc[i]["imd"])
        derived["imdu"] = imdu
        return derived

    def _encode_derived(self, derived):
        derived =derived.drop(["pcd"], axis=1)
        for col in derived.columns:
            if col_ops[col] is not None:
                derived[col] = col_ops[col].transform(derived[col])
        return derived

def round_booleans(X):
    for bool_feature in bool:
        if bool_feature != "isfailed":
            X[bool_feature] = X[bool_feature].apply(round)
    return X

# No longer using this - we load in a dict created when preprocessing
def count_companies_with_same_pcd(full_data):
    pcdDict = dict()
    for pcd in full_data['pcd']:
        actual_pcd = format_pcd(pcd)
        pcdDict[actual_pcd] = pcdDict.get(actual_pcd, 0) + 1
    return pcdDict

def cty_map(cty):
    if type(cty) == str:
        country = cty[0]
        if country == "L" or country == "M":  # map Channel Islands to England
            country = 'E'
    else:
        country = 'E'
    return country


# imd is not comparable across England, Wales, Scotland and NI
# Using regressions of a universal IMD on country imd from UK_indices_of_multiple_deprivation-a_way_to_make_c.pdf
rO = np.array([[138., 634], [100., 651.], [110., 651.], [0., 0.]])
rI = np.array([[706., 41.], [682., 58.], [678., 56.], [1., 1.]])
rA = np.array([[145., 634.], [111., 606.], [121., 618.], [0., 0.]])
rB = np.array([[583., 191.], [546., 100.], [505., 272.], [1., 1.]])
# regression coefficients by country are:
regr = np.zeros([4, 2])

for idx, country in enumerate(rO):
    A = (rA[idx] - rO[idx]) / (rI[idx] - rO[idx])
    B = (rB[idx] - rO[idx]) / (rI[idx] - rO[idx])
    regr[idx] = [A[1], (B[1] - A[1]) / (B[0] - A[0])]


with open(os.path.join(data_dir, "pcd_dict.p"), "rb") as file:
    pcdDict = pickle.load(file)
