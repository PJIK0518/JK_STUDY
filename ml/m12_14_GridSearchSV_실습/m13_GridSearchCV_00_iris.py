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

NS = 5

KF = StratifiedKFold(n_splits= NS,
                     shuffle=True,
                     random_state=RS)

PM = [
    {'C':[1,10,100,1000],
     'kernel':['linear', 'sigmoid'],
     'degree':[3,4,5]},
    # 4*2*3 = 24회
    {'C':[1,10,100],
     'kernel':['rbf'],
     'gamma':[0.001,0.0001]},
    # 3*1*2 = 6회
    {'C':[1,10,100,1000],
     'kernel':['sigmoid'],
     'gamma':[0.01,0.001,0.0001],
     'degree':[3,4]}
    # 4*1*3*2 = 24회
]   # {24}+{6}+{24} = 총 54번 돌린다

#2. 모델
model = GridSearchCV(SVC(),         # model
                     PM,            # 파라미터 54가지 경우의 수
                     cv = KF,       # Kfold 5가지 경우의 수
                     verbose=1, 
                     refit=True,    # 재교육, 전체교육에 대한 검증 : 1회
                     n_jobs=-1,     # CPU을 몇 스레드 쓸 건가 
                                    # 54*5 + 1 = 271회 
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

print('훈련 소요시간 :', time.time() - S)

# Fitting 5 folds for each of 54 candidates, totalling 270 fits
# 최적 매개변수 : SVC(C=1, kernel='linear')
# 최적 파라미터 : {'C': 1, 'degree': 3, 'kernel': 'linear'}
# 훈련 최고점수 : 0.980952380952381
# 최고 성능평가 : 0.9555555555555556
# 실제 모델성능 : 0.9555555555555556
# 훈련 소요시간 : 0.6060550212860107