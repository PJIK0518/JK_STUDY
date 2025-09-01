# m10_02.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#       >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#       >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#       >> 데이터의 소실 없이 훈련 가능

import warnings

warnings.filterwarnings('ignore')

import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import random

RS = 44
np.random.seed(RS)
random.seed(RS)

warnings.filterwarnings('ignore')

#1. 데이터
DS = fetch_california_housing()

x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import warnings
import numpy as np
import random

### BayesianOptimization

from bayes_opt import BayesianOptimization
bayesian_params = {
    'n_estimator'       : (100,500),
    'learning_rate'     : (0.001, 0.1),
    'max_depth'         : (3, 10),
    'num_leaves'        : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight'  : (1, 50),
    'gamma'             : (0,5),
    'subsample'         : (0.5, 1),
    'colsample_bytree'  : (0.5, 1),
    'colsample_bylevel' : (0.5, 1),
    'max_bin'           : (9, 500),
    'reg_lambda'        : (0.0, 100),       # 정규화와 관련된 수치들
    'reg_alpha'         : (0.0, 10),        #
}

#2. 모델
def XGB(n_estimator, learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight,
        gamma, subsample, colsample_bytree, colsample_bylevel, max_bin, reg_lambda, reg_alpha):
    model = XGBRegressor(n_estimator = n_estimator,
                        learning_rate = learning_rate,
                        max_depth = int(max_depth),
                        num_leaves = num_leaves,
                        min_child_samples = min_child_samples,
                        min_child_weight = min_child_weight,
                        gamma= gamma,
                        subsample = subsample,
                        colsample_bytree = colsample_bytree,
                        colsample_bylevel = colsample_bylevel,
                        max_bin = int(max_bin),
                        reg_lambda = reg_lambda,
                        reg_alpha = reg_alpha,
                        n_jods = -1,
                        early_stopping_rounds=20,
                        eval_metric='rmse')

    #3. 훈련
    model.fit(x_trn, y_trn,
              eval_set = [(x_tst, y_tst)],
              verbose = 0)
    
    
    #4. 평가 예측
    y_prd = model.predict(x_tst)
    
    return r2_score(y_tst, y_prd)

Optimizer = BayesianOptimization(
    f = XGB,     
    pbounds = bayesian_params,  
    random_state = 333,
)

Optimizer.maximize(init_points = 15, 
                   n_iter = 35)   

print(Optimizer.max)

# {'target': 0.8423452760585384,
#  'params': {'colsample_bylevel': 1.0,
#             'colsample_bytree': 0.7154331497292492,
#             'gamma': 0.0,
#             'learning_rate': 0.1,
#             'max_bin': 269.1143512076401,
#             'max_depth': 10.0,
#             'min_child_samples': 11.364550340996145,
#             'min_child_weight': 24.403465909651104,
#             'n_estimator': 385.7503169016884,
#             'num_leaves': 24.0, 'reg_alpha': 10.0,
#             'reg_lambda': 29.92891703861696,
#             'subsample': 1.0}}              