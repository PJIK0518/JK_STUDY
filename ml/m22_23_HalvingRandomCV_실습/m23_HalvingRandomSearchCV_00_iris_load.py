from sklearn.datasets import load_iris

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
                                  # 채에 거르듯이 최적의 파라미터를 찾기 위한 녀석
                                
import numpy as np
import random
import time

RS = 44
np.random.seed(RS)
random.seed(RS)

#1. 데이터

x, y = load_iris(return_X_y=True)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    stratify=y,
    random_state=RS
)

import warnings
warnings.filterwarnings('ignore')
NS = 5

KF = StratifiedKFold(n_splits= NS,
                     shuffle=True,
                     random_state=RS)

PM = [
    {'n_estimators': [100,500], 'max_depth': [6,10, 12] ,'learning_rate' : [0.1, 0.01, 0.001]}, # 18
    {'max _depth': [6,8,10,12], 'learning rate' : [0.1, 0.01, 0.001]},                          # 12
    {'min_child weight':[2,3,5,10], 'learning_rate' : [0.1, 0.01,0.001]},                       # 12
]

#2. 모델
from xgboost import XGBClassifier
# model = GridSearchCV(XGBClassifier(),       
#                      PM,           
#                      cv = KF,     
#                      verbose=1, 
#                      refit=True, # best_estimator_ / _params_ / _score_ 를 쓰려면 True
#                      n_jobs=-1,   
                                 
# )
import joblib
save = '00'
path = './Study25/_save/m22_HRGS_save/'
model = joblib.load(path + f'm22_{save}_best_model.joblib')

#4. 평가 예측
# print('훈련 최고점수 :', model.best_score_)   # GSCV 로 진행할 때 쓰는 기능 : 저장은 GSCV 안에 들어간 모델 기준을 저장됨
print('최고 성능평가 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)
print('실제 모델성능 :', accuracy_score(y_tst, y_prd))

# y_prd_best = model.predict(x_tst)
# print('최고 모델성능 :', accuracy_score(y_tst, y_prd_best))

# print('훈련 소요시간 :', time.time() - S)
# saveNum = '00'
# joblib.dump(model.best_estimator_, path + f'm16_best_model_{saveNum}.joblib')

print(type(model))  # 모델의 종류 출력
print(model)        # 모델의 파라미터에 대한 명세 출력

# 최고 성능평가 : 0.9555555555555556
# 실제 모델성능 : 0.9555555555555556
# <class 'xgboost.sklearn.XGBClassifier'>
# XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=6,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=100,
#               n_jobs=None, num_parallel_tree=None, ...)