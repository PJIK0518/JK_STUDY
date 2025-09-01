import numpy as np
import random
import warnings

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')

RS = 35
np.random.seed(RS)
random.seed(RS)

#1. 데이터
DS = fetch_california_housing()
x = DS.data
y = DS.target

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.8,
    random_state=RS,  
)
# print(x.shape, y.shape) (20640, 8) (20640,)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb

#2 모델구성
model = XGBRegressor(
    n_estimators = 200,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,                 # CHAT GPT에 각각 계산 방식 물어보기 #
 #  eval_metric = 'mlogloss',       # 다중 분류 : mlogloss, merror
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

print(total)

                            # Nan 값이 있으면 0으로 채워라
gain_list = [gain.get(f'f{i}', 0) / total for i in range(x.shape[1])] 
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
    
    select_model = XGBRegressor(
    n_estimators = 200,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,                 
    eval_metric = 'logloss',
    early_stopping_rounds=10,
    random_state=RS)
    
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
        BEST_col = list(C_feature)
        BEST_drp = list(D_feature)
        
    print(f'Threshold = {i:.3f} / n = {select_x_trn.shape[1]:2d} / R2 = {score*100:.3f}%')
    print(C_feature)
    print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')

print('최적 컬럼 :', BEST_trn.shape[1],'개','\n',
      BEST_col)
print('삭제 컬럼 :',f'{x_trn.shape[1]-BEST_trn.shape[1]}','개','\n',
      BEST_drp)
print('최고 점수', f'{BEST_scr*100:.3f}%')

# 최적 컬럼 : 6 개 
#  ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']
# 삭제 컬럼 : 2 개 
#  ['AveBedrms', 'Population']
# 최고 점수 81.254%