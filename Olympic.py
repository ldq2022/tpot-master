from numpy import genfromtxt
from tpot import TPOTClassifier
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
import pickle

with open("log.txt", "w") as f:
    f.write('')



features = genfromtxt('features.csv', delimiter=',')
target = genfromtxt('labels.csv', delimiter=',')

tpot = TPOTClassifier(generations=3,
                      population_size=5,
                      verbosity=2,
                      max_time_mins=2,
                      template="Transformer-Classifier")

tpot.fit(features, target)
# print(tpot.score(X_test, y_test))
best_pipelines = list(tpot.pareto_front_fitted_pipelines_.values())[0]
dump(best_pipelines, 'exported_pipelines/best.joblib')
pickle.dump(best_pipelines, open('exported_pipelines/best.pkl', 'wb'))
tpot.export('tpot_olympics_pipeline.py')

