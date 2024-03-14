import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB

def transform_input_data(set, training_set):
    set = set.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis=1)

    set["Sex"] = np.where(set["Sex"] == "female", True, False)
    set["Survived"] = np.where(set["Survived"] == 1, True, False)

    set = set.dropna(subset=["Embarked"])
    set = set.reset_index(drop=True)

    training_set = training_set.dropna(subset=["Embarked"])
    training_set = training_set.reset_index(drop=True)

    training_set_num = training_set.drop(["PassengerId", "Survived", "Name", "Cabin", 
                                      "Ticket", "Sex", "Embarked"], axis=1)
    set_num = set.drop(["Survived", "Embarked", "Sex"], axis=1)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(training_set_num)
    X = imputer.transform(set_num)
    set_num = pd.DataFrame(X, columns=set_num.columns, index=set_num.index)

    scaler = StandardScaler()
    scaler.fit(training_set_num)
    X = scaler.transform(set_num)

    training_set_1hot = training_set[["Pclass", "SibSp", "Parch", "Embarked"]]
    set_1hot = set[["Pclass", "SibSp", "Parch", "Embarked"]]

    cat_encoder = OneHotEncoder()
    cat_encoder.fit_transform(training_set_1hot)
    set_1hot = cat_encoder.transform(set_1hot)
    set_1hot = set_1hot.toarray()
    set_1hot = pd.DataFrame(set_1hot, columns = cat_encoder.get_feature_names_out())

    set_num = set_num.drop(["Pclass", "SibSp", "Parch"], axis=1)

    labels = set["Survived"]

    set = pd.concat([set_num, set["Sex"], set_1hot], axis=1)

    return set, labels