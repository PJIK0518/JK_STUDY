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
    random_state=777,
    stratify=y
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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb

#2 모델구성
model = XGBClassifier(
    n_estimators = 200,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,                 # CHAT GPT에 각각 계산 방식 물어보기 #
  eval_metric = 'mlogloss',       # 다중 분류 : mlogloss, merror
                                    # 이진 분류 : logloss, error
                                    # 2.1.1 버전 이후로 fit 에서 모델로 위치 변경
    early_stopping_rounds=10,
    random_state=RS
    )

model.fit(x_trn, y_trn,
          eval_set = [(x_tst,y_tst)],
          verbose = 0)

gain = model.get_booster().get_score(importance_type = 'gain')

total = sum(gain.values())

# print(total)

                            # Nan 값이 있으면 0으로 채워라
gain_list = [i / total for i in gain.values()] 
# print(gain_list)

# print(len(gain_list))



thresholds = np.sort(gain_list)

feature_names = np.array(DS.feature_names)

from sklearn.feature_selection import SelectFromModel

BEST_col = []
BEST_drp = []
BEST_scr = 0
BEST_trn = x_trn

for i in thresholds:
    selection = SelectFromModel(model,
                                threshold=i,
                                prefit=False)
     
    select_x_trn = selection.transform(x_trn)
    select_x_tst = selection.transform(x_tst)
    
    select_model = XGBClassifier(
    n_estimators = 200,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,                 
    eval_metric = 'mlogloss',
    early_stopping_rounds=10,
    random_state=42)
    
    if select_x_trn.shape[1] == 0:
        print('DONE')
        continue
    
    select_model.fit(select_x_trn, y_trn,
                     eval_set = [(select_x_tst,y_tst)],
                     verbose = False)
                
    score = select_model.score(select_x_tst,y_tst)

    Columns = selection.get_support()       
    Droped = [not i for i in Columns]
    C_feature = feature_names[Columns]
    D_feature = feature_names[Droped]
    
    if BEST_scr <= score:
        BEST_scr = score
        BEST_trn = select_x_trn
        BEST_col = C_feature
        BEST_drp = D_feature
        
    print(f'Threshold = {i:.3f} / n = {select_x_trn.shape[1]:2d} / R2 = {score*100:.3f}%')
    print(C_feature)
    print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')

print('최적 컬럼 :', BEST_trn.shape[1],'개','\n',
      BEST_col)
print('삭제 컬럼 :',f'{x_trn.shape[1]-BEST_trn.shape[1]}','개','\n',
      BEST_drp)
print('최고 점수', f'{BEST_scr*100:.3f}%')
    
# 최적 컬럼 : 3 개 
#  ['color_intensity' 'od280/od315_of_diluted_wines' 'proline']
# 삭제 컬럼 : 10 개 
#  ['alcohol' 'malic_acid' 'ash' 'alcalinity_of_ash' 'magnesium'
#  'total_phenols' 'flavanoids' 'nonflavanoid_phenols' 'proanthocyanins'
#  'hue']
# 최고 점수 100.000%