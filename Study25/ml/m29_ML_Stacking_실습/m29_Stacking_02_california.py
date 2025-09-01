import numpy as np
import random
import warnings

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

RS = 35
np.random.seed(RS)
random.seed(RS)

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.8,
    random_state=RS,
    
)

#2_1. 모델
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

XGB = XGBRegressor()
LGB = LGBMRegressor(verbosity=-1)
CAT = CatBoostRegressor(verbose=0)  # iterators Default : 1000
RFR = RandomForestRegressor()

model = [CAT, XGB, RFR, LGB]
trn_L = []
tst_L = []

for model in model :
    model.fit(x_trn, y_trn)
    
    y_trn_prd = model.predict(x_trn) # Stacking 시켜서 새로운 
    y_tst_prd = model.predict(x_tst)
    
    trn_L.append(y_trn_prd)
    tst_L.append((y_tst_prd))
    
    score = r2_score(y_tst, y_tst_prd)
    
    class_name = model.__class__.__name__
    
    # print('{0} R2 : {1:.4f}'.format(class_name, score))
    print(f'{class_name} R2 : {score:.4f}')
# XGBRegressor R2 : 0.8301
# LGBMRegressor R2 : 0.8360
# CatBoostRegressor R2 : 0.8492
# RandomForestRegressor R2 : 0.8051
    
# trn_L : list 형태 > numpy
# list : 데이터 column과 N 꼬여서 Append >> 전치!!

x_trn_NEW = np.array(trn_L).T
x_tst_NEW = np.array(tst_L).T

# print(x_trn_NEW)
# print(x_trn_NEW.shape) (16512, 4)
# print(x_tst_NEW.shape) (4128, 4)

#2_2. 모델 Stacking
model_S = RandomForestRegressor()
model_S.fit(x_trn_NEW,y_trn)
y_prd_S = model_S.predict(x_tst_NEW)
score_S = r2_score(y_tst, y_prd_S)

print('Stacking score :', score_S)

# CAT Stacking score : 0.7983593852827535
# RFR Stacking score : 0.7921566098719263
# XGB Stacking score : 0.7943926086740463
# LBG Stacking score : 0.7969438579048882
# 3 + RFR Stacking score : 0.8194806386213433
# 3 + CAT Stacking score : 0.7999265874729351
# 3 + XGB Stacking score : 0.7852941453226098
# 3 + LBG Stacking score : 0.7921359473426813
# SQ Stacking score : 0.7969438579048882

