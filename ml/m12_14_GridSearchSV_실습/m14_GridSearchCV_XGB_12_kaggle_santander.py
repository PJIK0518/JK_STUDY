# m10_12.copy

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

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


#1. 데이터
path = './Study25/_data/kaggle/santander/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

x = trn_csv.drop(['target'], axis=1)
y = trn_csv['target']

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)

#####################################
## Scaler
def Scaler(SC, a, b):
    SC.fit(a)
    return SC.transform(a), SC.transform(b)

x_trn, tst_csv = Scaler(RobustScaler(), x_trn, tst_csv)

def Scaler(SC, a, b):
    SC.fit(a)
    return SC.transform(a), SC.transform(b)

x_trn, tst_csv = Scaler(StandardScaler(), x_trn, tst_csv)

#####################################
## 증폭 : class_weight
classes = np.unique(y)
CW = compute_class_weight(class_weight='balanced', classes=classes, y=y)

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
model = GridSearchCV(XGBClassifier(),       
                     PM,           
                     cv = KF,     
                     verbose=2, 
                     refit=True,  
                     n_jobs=8,   
                                 
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

# 최적 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               feature_weights=None, gamma=None, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=6,
#               max_leaves=None, min_child_weight=None, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimators=500,
#               n_jobs=None, num_parallel_tree=None, ...)
# 최적 파라미터 : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
# 훈련 최고점수 : 0.9178888888888889
# 최고 성능평가 : 0.09965
# 실제 모델성능 : 0.09965
# 훈련 소요시간 : 1312.3164775371552