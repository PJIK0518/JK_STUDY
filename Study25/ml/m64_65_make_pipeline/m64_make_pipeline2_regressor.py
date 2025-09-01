# 실습 02~05 을 한파일에 한번...! (fetch_cov 제외)

##############
#1. 데이터셋 (4개)
#2. 스케일러 (4개)
#3. 모델     (5개)

# [결과] : 6개의 데이터 셋에서 각...
#         어떤 모델, 어떤 스케일러에서 점수 어캐나오는지

# [실습준비]
from sklearn.datasets import fetch_california_housing, load_diabetes
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

x_02, y_02 = fetch_california_housing(return_X_y=True)
x_03, y_03 = load_diabetes(return_X_y=True)

path_ddr = './Study25/_data/dacon/00 따릉이/'
path_bik = './Study25/_data/kaggle/bike/'

trn_dbt = pd.read_csv(path_ddr + "train.csv", index_col=0)
trn_bik = pd.read_csv(path_bik + "train.csv", index_col=0)
trn_dbt = trn_dbt.dropna()
trn_bik = trn_bik.dropna()

x_04 = trn_dbt.drop(['count'], axis=1)
y_04 = trn_dbt['count']

x_05 = trn_bik.drop(['count'], axis=1)
y_05 = trn_bik['count']

best_scor_02 = 0
best_sclr_02 = ''
best_modl_02 = ''
best_nscl_02 = ''
best_nmdl_02 = ''

best_scor_03 = 0
best_sclr_03 = ''
best_modl_03 = ''
best_nscl_03 = ''
best_nmdl_03 = ''

best_scor_04 = 0
best_sclr_04 = ''
best_modl_04 = ''
best_nscl_04 = ''
best_nmdl_04 = ''

best_scor_05 = 0
best_sclr_05 = ''
best_modl_05 = ''
best_nscl_05 = ''
best_nmdl_05 = ''

list_data = [
     ('clfn', x_02, y_02),
     ('dbts', x_03, y_03),
     ('ddrg', x_04, y_04),
     ('bike', x_05, y_05)]

list_sclr = [
     ('MinMaxScaler', MinMaxScaler()),
     ('MaxAbsScaler', MaxAbsScaler()),
     ('RobustScaler', RobustScaler()),
     ('StandardScaler', StandardScaler())
]

list_modl = [
     ('XGBRegressor', XGBRegressor(verbose=0)),
     ('LGBMRegressor', LGBMRegressor(verbosity=-1)),
     ('CatBoostRegressor', CatBoostRegressor(verbose=0)),
     ('RandomForestRegressor', RandomForestRegressor(verbose=0)),
     ('GradientBoostingRegressor', GradientBoostingRegressor(verbose=0))
]

best_results = []

for data, x, y in list_data:
     print('data :', data)
     x_trn, x_tst, y_trn, y_tst = train_test_split(
          x, y,
          train_size=0.7,
          shuffle=True,
          random_state=777
     )
     
     best = {
     'dataset': data,
     'score': -1.0,
     'scaler_name': None,
     'model_name': None
    }
     
     for snme, sclr in list_sclr:
          for mnme, modl in list_modl:
               print('pipeline :', snme, mnme)
               model = make_pipeline(sclr, modl)
               
               model.fit(x_trn, y_trn)
               
               scr = model.score(x_tst, y_tst)
     
               if scr > best['score']:
                    best['score'] = scr
                    best['scaler_name'] = snme
                    best['model_name'] = mnme
                    
     best_results.append(best)
     
print("\n=== Best by dataset ===")
for br in best_results:
    print(f"{br['dataset']} : {br['score']:.4f} | {br['scaler_name']} | {br['model_name']}")
    
# === Best by dataset ===
# clfn : 0.8420 | RobustScaler | CatBoostRegressor
# dbts : 0.4103 | MinMaxScaler | GradientBoostingRegressor
# ddrg : 0.7714 | RobustScaler | CatBoostRegressor
# bike : 0.9997 | StandardScaler | RandomForestRegressor