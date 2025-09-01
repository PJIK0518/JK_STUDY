# m10_02.copy

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import numpy as np
import warnings
import random

warnings.filterwarnings('ignore')

RS = 44
np.random.seed(RS)
random.seed(RS)

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
    'reg_lambda'        : (0.0, 100),       
    'reg_alpha'         : (0.0, 10),        
}

#2. 모델
def XGB(learning_rate, max_depth, min_child_weight,
        subsample, colsample_bytree, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),        # 소수점 반올림하고 정수형으로 두번 처리
        'min_child_weight'  : int(round(min_child_weight)),
        'subsample' : max(0, min(subsample,1)),                    # 0~1 사이의 값만 가능
        'reg_lambda' : max(0, reg_lambda),
        'reg_alpha' : reg_alpha
    }
    model = XGBRegressor(**params, n_jods = -1,
                         early_stopping_rounds=20,
                         eval_metric='rmse')

    #3. 훈련
    model.fit(x_trn, y_trn,
              eval_set = [(x_tst, y_tst)],
              verbose = 0)
    
    
    #4. 평가 예측
    y_prd = model.predict(x_tst)
    result =r2_score(y_tst, y_prd)
    
    return result           # 만약 
import time

S = time.time()
# 최적화 : BayesianOptimization
Optimizer = BayesianOptimization(
    f = XGB,     
    pbounds = bayesian_params,  
    random_state = RS,
)
n_iter = 100
Optimizer.maximize(init_points = 10, 
                   n_iter = n_iter)   
print(Optimizer.max)              
print(time.time() - S)
{'target': 0.8410271153046742, 'params': {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 10.0, 'min_child_weight': 34.21520844327828, 'reg_alpha': 0.01, 'reg_lambda': 0.0, 'subsample': 0.5}}