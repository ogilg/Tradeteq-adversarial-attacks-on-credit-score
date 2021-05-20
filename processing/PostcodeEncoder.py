import pandas as pd
import os
import joblib

class PostcodeEncoder():
    data_dir = os.path.dirname(__file__)
 
    ons_data_folder = os.path.join(os.path.dirname(data_dir), "client_start_folder",
                                       "ONS", "data")

    NSPL_file = "NSPL_AUG_2019_UK.csv"
    ons_data = pd.read_csv(os.path.join(ons_data_folder, NSPL_file),
                               usecols=["pcd"])

    col_ops = joblib.load(os.path.join(data_dir, "column_operations.pkl"))

    def num_postcodes(self):
        return len(self.ons_data)-1

    # converts postcode string (eg AB1 AB1) into encoding used by models
    # input: pcd : String 
    #def model_encode(self, pcd):
    #    return self.col_ops['pcd'].transform([pcd])[0]

    # convert encoding used by models into postcode string
    # input: pcd : Int
    #def model_decode(self, pcd):
    #    return self.col_ops['pcd'].inverse_transform([pcd])[0]

    # convert postcode string in search encoding
    # input: pcd : String 
    def search_encode(self, pcd):
        return pd.Index(self.ons_data['pcd']).get_loc(format_pcd(pcd))

    # convert search encoding into postcode string
    # input: pcd : Int
    def search_decode(self, pcd):
        return self.ons_data['pcd'][pcd]

def format_pcd(pcd):
    second_half = pcd[(len(pcd)-3):]
    first_half = pcd[:(len(pcd)-3)]
    if len(first_half) == 2:
        first_half += "  "
    elif len(first_half) == 3:
        first_half += " "
    return(first_half + second_half)