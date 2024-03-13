import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


csv_path = os.path.join(os.getcwd(), "datasets", "train.csv")
training_set = pd.read_csv(csv_path, index_col=False)


training_set.hist(bins=50, figsize=(20,15))
# plt.show()

training_set = training_set.drop("Name", axis=1)
training_set = training_set.drop("Cabin", axis=1)
training_set = training_set.drop("Ticket", axis=1)
training_set["Sex"] = np.where(training_set["Sex"] == "female", True, False)
training_set["Survived"] = np.where(training_set["Survived"] == 1, True, False)


training_set = training_set.dropna(subset=["Embarked"])

training_set_num = training_set.drop(training_set.columns[[8]], axis=1)
training_set_str = training_set[['Embarked']]

imputer = SimpleImputer(strategy="median")
imputer.fit(training_set_num)
X = imputer.transform(training_set_num)
training_set_num = pd.DataFrame(X, columns=training_set_num.columns, index=training_set_num.index)

training_set_1hot = training_set[["Pclass", "SibSp", "Parch", "Embarked"]]

cat_encoder = OneHotEncoder()
training_set_1hot = cat_encoder.fit_transform(training_set_1hot)

training_set_1hot = training_set_1hot.toarray()

training_set_1hot = pd.DataFrame(training_set_1hot, columns = cat_encoder.get_feature_names_out())

print(training_set_1hot.head(25))

training_set_num = training_set_num.drop("Pclass", axis=1)
training_set_num = training_set_num.drop("SibSp", axis=1)
training_set_num = training_set_num.drop("Parch", axis=1)

training_set = pd.concat([training_set_num, training_set_1hot], axis=1)

print(training_set.info())

corr_matrix = training_set.corr()
print(corr_matrix["Survived"].sort_values(ascending=False))
