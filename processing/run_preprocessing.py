import os
import pickle
import pandas as pd
from Processing.DataProcessor import DataProcessor

if __name__ == '__main__':
    # Read the sample file
    data_dir = os.path.dirname(os.getcwd())
    co_file = os.path.join(data_dir, "Co_600K_Jul2019_6M.pkl")
    data = pd.read_pickle(co_file)
    data_proc = DataProcessor(data)
    X_train, X_test, y_train, y_test = data_proc.build_column_transformers()

    with open(r"train_data.p", "wb") as train_file:
        pickle.dump([X_train, y_train], train_file)
    with open(r"test_data.p", "wb") as test_file:
        pickle.dump([X_test, y_test], test_file)

    test_preproc = DataProcessor(X_train)
    X_train_preproc = test_preproc.preprocess_data()

    test_preproc = DataProcessor(X_test)
    X_test_preproc = test_preproc.preprocess_data()

    with open(r"train_preproc.p", "wb") as train_file:
        pickle.dump([X_train_preproc, y_train], train_file)
    with open(r"test_preproc.p", "wb") as test_file:
        pickle.dump([X_test_preproc, y_test], test_file)
