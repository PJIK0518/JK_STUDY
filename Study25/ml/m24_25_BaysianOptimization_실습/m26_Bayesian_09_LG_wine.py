# m10_09.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#       >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#       >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#       >> 데이터의 소실 없이 훈련 가능

import warnings

warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

#1. 데이터
DS = load_wine()

x = DS.data
y = DS.target
x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
import warnings
import numpy as np
import random
import time

RS = 44
np.random.seed(RS)
random.seed(RS)

warnings.filterwarnings('ignore')
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier, LGBMRegressor
bayesian_params = {
    'learning_rate': (0.001, 0.1),
    'max_depth': (3, 10),
    'num_leaves': (24, 40),
    'min_child_samples': (10, 200),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'reg_lambda': (0.0, 100.0),
    'reg_alpha': (0.0, 10.0),
    'max_bin': (9, 500)
}

#2. 모델
def LGBM(learning_rate, max_depth, num_leaves, min_child_samples,
        subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    model = LGBMClassifier(learning_rate=learning_rate,
                            max_depth=int(max_depth),
                            num_leaves=int(num_leaves),
                            min_child_samples=int(min_child_samples),
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            reg_lambda=reg_lambda,
                            reg_alpha=reg_alpha,
                            max_bin=int(max_bin),
                            random_state=132,
                            n_jobs=-1,
                            verbose=0,
                            early_stopping_rounds=20,
                            verbosity=-1, 
    )
    
    #3. 훈련
    model.fit(x_trn, y_trn,
              eval_set = [(x_tst, y_tst)])
    
    
    #4. 평가 예측
    y_prd = model.predict(x_tst)
    
    return accuracy_score(y_tst, y_prd)

Optimizer = BayesianOptimization(
    f = LGBM,     
    pbounds = bayesian_params,  
    random_state = 333,
)
Optimizer.maximize(init_points = 15, 
                   n_iter = 50)   

print(Optimizer.max)        
# LGBM  {'target': 1.0,
# {'target': 1.0,
#  'params': {'colsample_bylevel': 0.7716455435126173,
#             'colsample_bytree': 0.8644753639495034, 'gamma': 0.08440723019894236,
#             'learning_rate': 0.03370354148846345, 'max_bin': 190.04241601832527,
#             'max_depth': 3.3381256602464306, 'min_child_samples': 29.86073575276946,
#             'min_child_weight': 5.774438265201903, 'n_estimator': 198.16132580542174,
#             'num_leaves': 37.975829671340975, 'reg_alpha': 0.5161163930274903, 
#             'reg_lambda': 6.048133573318914, 'subsample': 0.6193700568007332}}       