import numpy as np
import os
from joblib import dump, load
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle

training_features = genfromtxt('training_features_imputed.csv', delimiter=',')
testing_features = genfromtxt('testing_features_imputed.csv', delimiter=',')
training_target = genfromtxt('training_target_imputed.csv', delimiter=',')
testing_target = genfromtxt('testing_target_imputed.csv', delimiter=',')



# reloaded_pip = load('exported_pipeline.joblib')
files = os.listdir(os.getcwd() + '/exported_pipelines')
for file in files:
    reloaded_pip = pickle.load(open('exported_pipelines/'+file, 'rb'))

    results = reloaded_pip.predict(testing_features)
    diff = results - testing_target  # comparing predicted label and the target
    num_of_errors = np.sum(np.absolute(diff))   # count wrong predictions
    print(file, " |  Correct Prediction: ", (len(results)-num_of_errors)/len(results)*100, "%")




