import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import data_manipulation as dm

training_set_path = os.path.join(os.getcwd(), "datasets", "train.csv")
training_set_original = pd.read_csv(training_set_path, index_col=False)

test_set_path = os.path.join(os.getcwd(), "datasets", "test.csv")
test_set = pd.read_csv(test_set_path, index_col=False)

test_set_labels_path = os.path.join(os.getcwd(), "datasets", "gender_submission.csv")
test_set_labels = pd.read_csv(test_set_labels_path, index_col=False)

test_set = pd.merge(test_set, test_set_labels, on='PassengerId')

training_set, training_set_labels = dm.transform_input_data(training_set_original, training_set_original)
test_set, test_set_labels = dm.transform_input_data(test_set, training_set_original)

# print(training_set[training_set.isnull().any(axis=1)])
# print(training_set.to_string())




mnb = MultinomialNB().fit(training_set, training_set_labels)
print("score on test: " + str(mnb.score(test_set, test_set_labels)))
print("score on test: " + str(mnb.score(training_set, training_set_labels)))