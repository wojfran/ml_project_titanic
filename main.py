import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


csv_path = os.path.join(os.getcwd(), "datasets", "train.csv")
training_set = pd.read_csv(csv_path, index_col=False)

print(training_set.head())

training_set.hist(bins=50, figsize=(20,15))
# plt.show()

training_set = training_set.drop("Name", axis=1)
training_set = training_set.drop("Cabin", axis=1)
training_set = training_set.drop("Ticket", axis=1)
training_set["Sex"] = np.where(training_set["Sex"] == "female", True, False)
training_set["Survived"] = np.where(training_set["Survived"] == 1, True, False)

print(training_set.head())
print(training_set.columns.values)
print(training_set.info())
print(training_set.describe())
print(training_set[training_set.isna().any(axis=1)].to_string())

training_set = training_set.dropna(subset=["Embarked"])

imputer = SimpleImputer(strategy="median")
training_set_num = training_set.drop(training_set.columns[[8]], axis=1)
training_set_str = training_set[['Embarked']]
imputer.fit(training_set_num)
X = imputer.transform(training_set_num)
training_set_num = pd.DataFrame(X, columns=training_set_num.columns, index=training_set_num.index)

# print(training_set_str.info())

training_set = pd.concat([training_set_num, training_set_str], axis=1)

print(training_set.head())
