import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import time
import xgboost
import sklearn

from bayes_opt import BayesianOptimization
import xgboost as xgb

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.filterwarnings('ignore')

from preproc import Preprocessor

DATA_DIR = os.getcwd()

X = pd.read_csv(f'{DATA_DIR}/training_set_values.csv')
y = pd.read_csv(f'{DATA_DIR}/training_set_labels.csv')
y['status_group'] = y['status_group'].astype('category').cat.codes

status_group_cats = { 0: 'functional', 1: 'functional needs repair', 2: 'non functional'}

p = Preprocessor(X, top_n=100)
X = p.preprocess(X, train=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y['status_group'])
dtrain = xgb.DMatrix(X_train[p.colset], label=y_train)
dtest = xgb.DMatrix(X_test[p.colset])


def xgb_evaluate(n_estimators, max_depth, gamma, eta, colsample_bytree, subsample):
    params = {
              'gpu_id': 0,
              'tree_method': 'gpu_hist',
              'objective': 'multi:softmax',
              'max_depth': int(max_depth),
              'subsample': float(subsample),
              'eta': float(eta),
              'num_class': 3,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=int(n_estimators), nfold=3)    
   
    # Bayesian optimization only knows how to maximize, not minimize
    return -1.0 * cv_result['test-mlogloss-mean'].iloc[-1]


start = time.time()
xgb_bo = BayesianOptimization(xgb_evaluate, {'n_estimators': (100, 1000),
                                             'max_depth': (3, 15),
                                             'gamma': (0, 1),
                                             'eta': (0.01, 0.1),
                                             'colsample_bytree': (0.3, 0.9),
                                             'subsample': (0.5, 1.0)})
# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
init_points = 10
n_iter = 50
xgb_bo.maximize(init_points=init_points, n_iter=n_iter, acq='ei')
duration = -np.round(start - time.time()) / 60

print(f"Finished {init_points}/{n_iter} init/iter optimization in {duration} minutes.")
params = xgb_bo.max['params']

# params= {
#   'colsample_bytree': 0.3615,
#   'eta': 0.06958,
#   'gamma': 0.2715,
#   'max_depth': 13,
#   'n_estimators': 662,
#   'subsample': 1
# }

params['max_depth'] = int(params['max_depth'])
params['num_class'] = 3
params['objective'] = 'multi:softmax'
params['gpu_id'] = 0
params['tree_method'] = 'gpu_hist'
# Train a new model with the best parameters from the search
model = xgb.train(params, dtrain, num_boost_round=int(params['n_estimators']))

# Evaluate
y_pred = model.predict(dtest)
y_train_pred = model.predict(dtrain)

print("Train Classification")
print(sklearn.metrics.classification_report(y_train, y_train_pred))

print("Test Classification")
print(sklearn.metrics.classification_report(y_test, y_pred))

# print("Train Accuracy: " + str(sklearn.metrics.accuracy_score(y_train, y_train_pred)))
# print("Test Accuracy: " + str(sklearn.metrics.accuracy_score(y_test, y_pred)))

fscores = pd.DataFrame({'X': list(model.get_fscore().keys()),
                        'Y': list(model.get_fscore().values())})
fscores.to_csv(f'{DATA_DIR}/f_scores.csv', index=False)

# Holdout
X_holdout = pd.read_csv(f'{DATA_DIR}/test_set_values.csv')
X_holdout = p.preprocess(X_holdout)
assert set(X_holdout.columns) == set(X.columns)
dholdout = xgb.DMatrix(X_holdout[p.colset])
y_pred_holdout = model.predict(dholdout)
y_pred_holdout = pd.DataFrame({'id': X_holdout['id'], 'status_group': y_pred_holdout})
y_pred_holdout['status_group'] = y_pred_holdout['status_group'].apply(lambda x: status_group_cats[int(x)])
y_pred_holdout.to_csv(f"{DATA_DIR}/submission.csv", index=False)
