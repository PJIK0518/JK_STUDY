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
model = GridSearchCV(XGBClassifier(),       
                     PM,           
                     cv = KF,     
                     verbose=1, 
                     refit=True, # best_estimator_ / _params_ / _score_ 를 쓰려면 True
                     n_jobs=-1,   
                                 
)

#3. 훈련
S = time.time()
model.fit(x_trn, y_trn)
print('최적 매개변수 :', model.best_estimator_)
print('최적 파라미터 :', model.best_params_)

#4. 평가 예측
print('훈련 최고점수 :', model.best_score_)
print('최고 성능평가 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)
print('실제 모델성능 :', accuracy_score(y_tst, y_prd))

y_prd_best = model.best_estimator_.predict(x_tst)
print('최고 모델성능 :', accuracy_score(y_tst, y_prd_best))

saveNum = '00'
path = './Study25/_save/m16_GS_save/'
import joblib
print('훈련 소요시간 :', time.time() - S)
joblib.dump(model.best_estimator_, path + f'm16_best_model_{saveNum}.joblib')
print(saveNum, '저장완료')