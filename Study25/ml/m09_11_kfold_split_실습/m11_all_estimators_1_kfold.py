from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score

import numpy as np
import warnings
import pandas as pd

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

NS = 5

KF = KFold(n_splits=NS,
           shuffle=True,
           random_state=333)

#2. 모델
model_list = all_estimators(type_filter='regressor')

max_score = 0
max_model = 'default'

for name, model in model_list:
    
    try:
        #3. 훈련        
        model = model()
        model.fit(x_trn, y_trn)
        
        #4. 평가 예측        
        score = cross_val_score(model, x_tst, y_tst, cv = KF)
        
        y_prd = cross_val_predict(model, x_tst, y_tst, cv = KF)
        R2 = r2_score(y_tst, y_prd)        
        
        print(name, '의 정답률 :', R2)
        if R2 > max_score:                  
            max_score = R2
            max_model = name        
    except:
        print(name, ': ERROR')
        

print('최고모델 :', max_model, max_score) 

# 최고모델 : HistGradientBoostingRegressor 0.7793691750785984