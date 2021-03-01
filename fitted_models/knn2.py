from sklearn import preprocessing, impute
import pandas as pd
import numpy as np
from sklearn import preprocessing, datasets, impute
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

co_file = "Co_600K_Jul2019_6M.pkl"
df = pd.read_pickle(co_file)

# In case you want to drop any features before fitting model because it takes ages
# Don't theoretically need to drop anything except most likely names as label encoding will have an extremely large no of unique values
# dontInclude = [4,5,10,34,36,49,56,57,58,59,60,61,62,63,64,117,123,125,129,130]
# dontInclude2 = list(range(12,50)) + [4,5,10,56,57,58,59,60,61,62,63,64,117,123,125,129,130]
# dontInclude3 = list(range(12, len(list(df)))) + [4,5,10] + [2,3,6,7,8,9]
target_df = df["isfailed"]
dontInclude4 = list(range(12, 83)) + [4,5,10,117,123,125,129]
df.drop(df.columns[dontInclude4], axis=1, inplace=True)


num_cols = df._get_numeric_data().columns
categorical_columns = list(set(df.columns) - set(num_cols))

cols = list(df)

# Impute data (using pandas)
for column in cols:
	col_data = df[column]
	missing_data = sum(col_data.isna())
	if(missing_data > 0):
		if(column in categorical_columns):
			try:
				col_mode = col_data.value_counts().index[0]
				col_data.fillna(col_mode, inplace=True)
				df[column] = col_data
			except:
				print(column)
		else:
			col_median = col_data.median()
			col_data.fillna(col_median, inplace=True)
			df[column] = col_data

# Label encode categorical columns (using pandas)
for col in categorical_columns:
	df[col] = df[col].astype("category")
	df[col] = df[col].cat.codes

# target_df = df["isfailed"]
# del df["isfailed"]
# del df["Filled"]

# numpy conversion
target = target_df.to_numpy()
data = df.to_numpy()

# Fitting model
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)
print("About to fit")
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_scaled_train, y_train)
print("Fitted")
y_pred = knn.predict(X_scaled_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))