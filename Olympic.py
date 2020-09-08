import csv

from numpy import genfromtxt
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

x_train = genfromtxt('x_train.csv', delimiter=',')
y_train = genfromtxt('y_train.csv', delimiter=',')

tpot = TPOTClassifier(population_size=5, verbosity=2, max_time_mins=5)

tpot.fit(x_train, y_train)
# print(tpot.score(X_test, y_test))
tpot.export('tpot_olympics_pipeline.py')
print("---TPOT Finished---")
