import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import data_manipulation as dm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

training_set_path = os.path.join(os.getcwd(), "datasets", "train.csv")
training_set = pd.read_csv(training_set_path, index_col=False)

test_set_path = os.path.join(os.getcwd(), "datasets", "test.csv")
test_set = pd.read_csv(test_set_path, index_col=False)

test_set_labels_path = os.path.join(os.getcwd(), "datasets", "gender_submission.csv")
test_set_labels = pd.read_csv(test_set_labels_path, index_col=False)

test_set = pd.merge(test_set, test_set_labels, on='PassengerId')

training_set, training_set_labels = dm.transform_input_data(training_set)
test_set, test_set_labels = dm.transform_input_data(test_set)

mnb = MultinomialNB()
log_reg = LogisticRegression(max_iter=1000)
svc = SVC()
rfc = RandomForestClassifier()

mnb_score = cross_val_score(mnb, training_set, training_set_labels, cv=10, scoring="accuracy")
log_reg_score = cross_val_score(log_reg, training_set, training_set_labels, cv=10, scoring="accuracy")
svc_score = cross_val_score(svc, training_set, training_set_labels, cv=10, scoring="accuracy")
rfc_score = cross_val_score(rfc, training_set, training_set_labels, cv=10, scoring="accuracy")

mnb.fit(training_set, training_set_labels)
log_reg.fit(training_set, training_set_labels)
svc.fit(training_set, training_set_labels)
rfc.fit(training_set, training_set_labels)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print("\n")

print("MultinomialNB")
display_scores(mnb_score)
print("Logistic Regression")
display_scores(log_reg_score)
print("SVC")
display_scores(svc_score)
print("Random Forest")
display_scores(rfc_score)

mnb_predictions = mnb.predict(test_set)
log_reg_predictions = log_reg.predict(test_set)
svc_predictions = svc.predict(test_set)
rfc_predictions = rfc.predict(test_set)

mnb_accuracy = np.mean(mnb_predictions == test_set_labels)
log_reg_accuracy = np.mean(log_reg_predictions == test_set_labels)
svc_accuracy = np.mean(svc_predictions == test_set_labels)
rfc_accuracy = np.mean(rfc_predictions == test_set_labels)

print("MultinomialNB accuracy:", mnb_accuracy)
print("Logistic Regression accuracy:", log_reg_accuracy)
print("SVC accuracy:", svc_accuracy)
print("Random Forest accuracy:", rfc_accuracy)

