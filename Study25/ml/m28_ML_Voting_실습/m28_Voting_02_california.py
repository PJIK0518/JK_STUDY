# Bagging / Boosting / Voting / Stacking : ML에서 활용되는 주요한 기능들!
# 이미 ML 안에서 사용되고 있는 Ensembling 기법 중 하나

# Decision Tree : Pin ball(?) 느낌, 나무 구조의 결정 체계 : 데이터를 넣었을 때 도착 지점을 정해두는 느낌
                # 이상치 결측지에 강하다 > 어찌됐든 특정 분기점으로 들어가니까
                # Max_depth : Tree model이 몇 층으로 분지 되는지 결정
                
# RandomForest : Decision Tree라는 알고리즘 하나를 여러번 사용 = Bagging
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

#2. 모델
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

XGB = XGBRegressor()
LGB = LGBMRegressor()
CAT = CatBoostRegressor()

# vote = 'hard'
vote = 'soft'
model = VotingRegressor(
    estimators=[('XGB', XGB),
                ('LGB', LGB),
                ('CAT', CAT)],
)

#3. 훈련
model.fit(x_trn, y_trn)

#4. 평가 예측
rslt = model.score(x_tst, y_tst)

print(f'{vote}:', rslt)

# soft: 0.8458800458229289

# DecisionTreeRegressor  점수: 0.5841537054131041

# BaggingRegressor + DTR 점수: 0.8091707673506867 >> 직접 DTR을 Bagging 시킨 모델 : 점수 또이또이
            # bootstrap = True
            
# BaggingRegressor + DTR 점수: 0.6137284726843362
            # bootstrap = False
            # Sample 데이터 중복 허용
            
# RandomForestRegressor  점수: 0.8099250335618718 >> DTR이 Bagging 되어있는 모델  : 점수 또이또이