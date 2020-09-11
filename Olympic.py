from numpy import genfromtxt
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

with open("log.txt", "w") as f:
    f.write('')

# x_train = genfromtxt('x_train.csv', delimiter=',')
# y_train = genfromtxt('y_train.csv', delimiter=',')

features = genfromtxt('features.csv', delimiter=',')
target = genfromtxt('labels.csv', delimiter=',')

tpot = TPOTClassifier(generations=10,
                      population_size=5,
                      verbosity=2,
                      max_time_mins=2,
                      template="Transformer-Classifier")

tpot.fit(features, target)
# print(tpot.score(X_test, y_test))
tpot.export('tpot_olympics_pipeline.py')

