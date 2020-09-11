import numpy as np
import pickle
import pandas as pd
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer



features = genfromtxt('features.csv', delimiter=',')
target = genfromtxt('labels.csv', delimiter=',')


training_features, testing_features, training_target, testing_target = \
            train_test_split(features, target, random_state=None)

imputer = SimpleImputer(strategy="median")
imputer.fit(features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was: 0.8751173708920188
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.8500000000000001, min_samples_leaf=8, min_samples_split=15, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)



diff = results - testing_target  # comparing predicted label and the target
num_of_errors = np.sum(np.absolute(diff))   # count wrong predictions
print("Num of prediction errors: ", num_of_errors)

pickled_pip = pickle.dumps(exported_pipeline)
reloaded_pip = pickle.loads(pickled_pip)
results2 = reloaded_pip.predict(testing_features)

diff2 = results2 - testing_target  # comparing predicted label and the target
num_of_errors2 = np.sum(np.absolute(diff2))   # count wrong predictions
print("Num of prediction errors2: ", num_of_errors2)






print("prediction is finished")