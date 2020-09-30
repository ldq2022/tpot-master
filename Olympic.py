from numpy import genfromtxt
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from tabulate import tabulate
from pathlib import Path
import pickle
import pandas as pd
import os
import shutil


# create new path to save pipelines
if Path('exported_pipelines').exists():
    shutil.rmtree('exported_pipelines', ignore_errors=False, onerror=None)


features = genfromtxt('features.csv', delimiter=',')
target = genfromtxt('labels.csv', delimiter=',')

tpot = TPOTClassifier(generations=3,
                      population_size=4,
                      verbosity=2,
                      max_time_mins=2,
                      template="Selector-Classifier")

tpot.fit(features, target)
# print(tpot.score(X_test, y_test))

# save best pipeline as pkl
best_pipelines = list(tpot.pareto_front_fitted_pipelines_.values())[0]
os.makedirs('exported_pipelines/best')
pickle.dump(best_pipelines, open('exported_pipelines/best/best.pkl', 'wb'))


# visualization of all the evaluated pipelines
my_dict = list(tpot.evaluated_individuals_.items())

model_scores = pd.DataFrame()
for model in my_dict:
    model_name = model[0].replace(',', ',\n') + '\n'
    model_info = str(model[1]).replace(',', ',\n') + '\n'
    cv_score = model[1].get('internal_cv_score')  # Pull out cv_score as a column (i.e., sortable)
    model_path = model[1].get('model_path')
    model_scores = model_scores.append({'model': model_name,
                                        'cv_score': cv_score,
                                        'model_info': model_info,
                                        'model_path': model_path
                                        },
                                       ignore_index=True)

model_scores = model_scores.sort_values('cv_score', ascending=False)

print()
print("Pipelines Summary:")
print(tabulate(model_scores, tablefmt="fancy_grid", headers="keys"))


# save a template for reusing the best pipeline:
tpot.export('tpot_olympics_pipeline.py')
