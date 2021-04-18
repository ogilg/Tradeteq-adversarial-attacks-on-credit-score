import os
import pandas as pd
import numpy as np
import tqdm

"""
pcd : "oseast1m", "osnrth1m", "cty", "oa11",
                   "lat", "long", "ru11ind", "oac11",
                   "imd", "country"])

"""

ons_data_folder = os.path.join("..", "client_start_folder", "ONS", "data")
NSPL_file = "NSPL_AUG_2019_UK.csv"

ons_data = pd.read_csv(os.path.join(ons_data_folder, NSPL_file),
                       usecols=["pcd", "oseast1m", "osnrth1m", "cty", "oa11",
                                "lat", "long", "ru11ind", "oac11",
                                "imd"])

class ModelWrapper:
    def __init__(self, model, columns):
        self.model = model
        self.columns = columns

    def predict(self, X):
        return self.model.predict(self._complete_data(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._complete_data(X))

    def _complete_data(self, X):
        if len(X.columns) != len(self.columns):# or X.columns != self.columns:
            try:
                derived_data = X.pcd.map(self._pcd_to_derived)
                derived_df = pd.concat(derived_data)
                X = pd.concat([X, derived_df], axis=1)
            except Exception as error:
                print("Data fill-in process is not compatible with given data")
                raise error
        return X

    def _pcd_to_derived(self, pcd):
        derived = ons_data.loc[ons_data.pcd == pcd].to_frame().T
        print(derived)
        derived["country"] = derived.cty.map(cty_map)
        derived["oac1"] = derived.oac11.map(lambda x: x[0])
        derived["oac2"] = derived.oac11.map(lambda x: x[0:2])
        derived = self._add_imdu_col(derived)
        return derived

    # output argument
    def _add_imdu_col(self, derived):
        ctylist = "ENWS"
        derived["imdu"] = np.nan
        for i in tqdm.tqdm(range(0, derived.shape[0])):
            ctyidx = ctylist.index(derived.loc[i, "country"])
            derived.set_value(i, "imdu", regr[ctyidx, 0] + regr[ctyidx, 1] * derived.loc[i, "imd"])
        return derived


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




