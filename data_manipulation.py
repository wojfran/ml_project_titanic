from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Function which extrapolates the amount of distnct titles
# belonging to passengers in the array with their names
# and returns a dictionary with the titles as keys and the
# amount of passengers with that title as values
def extract_titles(names):
    titles = defaultdict(int)
    for name in names:
        title = name.split(",")[1].split(".")[0].strip()
        titles[title] += 1
    return titles

# Function which extracts the titles from the names of the passengers]
# and groups them into cathegories which would encapsulate all the titles
# given that this is the total amount of each distinct title:
# {'Mr': 517, 'Mrs': 125, 'Miss': 182, 'Master': 40, 'Don': 1, 'Rev': 6, 
# 'Dr': 7, 'Mme': 1, 'Ms': 1, 'Major': 2, 'Lady': 1, 'Sir': 1, 'Mlle': 2, 
#'Col': 2, 'Capt': 1, 'the Countess': 1, 'Jonkheer': 1}
# The cathegories are: Common, Proffession and Child
def group_title(name: str) -> str:
    common = ["Mr", "Mrs", "Miss", "Ms", "Mme", "Mlle"]
    profession = ["Don", "Rev", "Dr",  "Major", "Lady", "Sir", "Col", "Capt", "the Countess", "Jonkheer"]
    child = ["Master"]
    title = name.split(",")[1].split(".")[0].strip()
    if title in common:
        return "Common"
    elif title in profession:
        return "Profession"
    elif title in child:
        return "Child"
    else:
        return "Common"
    
# Function which groups groups the passenger by age into cathegories
# The cathegories are: Child, Young, Adult and Elder, if age is null
# the passenger is grouped as an Adult
def group_age(age: int) -> str:
    if age < 18:
        return "Child"
    elif age < 40:
        return "Young"
    elif age < 60:
        return "Adult"
    elif age >= 60:
        return "Elder"
    else:
        return "Adult"
    
# Function which groups the pasenger based on the fare paid
# The cathegories are: Low, Medium and High
def group_fare(fare: float) -> str:
    if fare < 10:
        return "Low"
    elif fare < 30:
        return "Medium"
    else:
        return "High"

def transform_input_data(data_set: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_set = data_set.drop(["PassengerId", "Cabin", "Ticket"], axis=1)
    data_set = data_set.dropna(subset=["Embarked"])
    data_set = data_set.reset_index(drop=True)

    data_set["Title"] = data_set["Name"].apply(group_title)
    data_set = data_set.drop("Name", axis=1)

    data_set["Age"] = data_set["Age"].apply(group_age)

    data_set["Fare"] = data_set["Fare"].apply(group_fare)

    data_set["Sex"] = np.where(data_set["Sex"] == "female", True, False)
    data_set["Survived"] = np.where(data_set["Survived"] == 1, True, False)

    dummy_values = [[0, 0, 0, 'C', 'Common', 'Adult', 'Low'],
                    [1, 1, 1, 'Q', 'Common', 'Adult', 'Low'],
                    [2, 2, 2, 'S', 'Common', 'Adult', 'Low'],
                    [3, 3, 3, 'S', 'Common', 'Adult', 'Low'],
                    [3, 4, 4, 'S', 'Common', 'Adult', 'Low'],
                    [3, 5, 5, 'S', 'Common', 'Adult', 'Low'],
                    [3, 6, 6, 'S', 'Common', 'Adult', 'Low'],
                    [3, 7, 7, 'S', 'Common', 'Adult', 'Low'],
                    [3, 8, 8, 'S', 'Common', 'Adult', 'Low'],
                    [3, 8, 9, 'C', 'Common', 'Adult', 'Low']]
    dummy_df = pd.DataFrame(dummy_values, columns=["Pclass", "SibSp", "Parch", "Embarked", "Title", "Age", "Fare"])
    data_set_1hot = data_set[["Pclass", "SibSp", "Parch", "Embarked", "Title", "Age", "Fare"]]
    dummy_1hot = pd.concat([dummy_df, data_set_1hot], axis=0, ignore_index=True)

    cat_encoder = OneHotEncoder(handle_unknown='ignore')
    cat_encoder.fit(dummy_1hot)  
    data_set_1hot = cat_encoder.transform(data_set_1hot)
    data_set_1hot = data_set_1hot.toarray()
    data_set_1hot = pd.DataFrame(data_set_1hot, columns = cat_encoder.get_feature_names_out())

    labels = data_set["Survived"]

    data_set = data_set.drop(["Pclass", "SibSp", "Parch", "Embarked", "Title", "Age", "Fare", "Survived"], axis=1)

    data_set = pd.concat([data_set, data_set_1hot], axis=1)

    return data_set, labels
