# m10_07.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#       >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#       >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#       >> 데이터의 소실 없이 훈련 가능

import warnings

warnings.filterwarnings('ignore')


from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
path = './Study25/_data/dacon/당뇨병/'

trn_csv = pd.read_csv(path + "train.csv", index_col=0)
'''print(trn_csv) [652 rows x 9 columns]
           Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
ID
TRAIN_000            4      103             60             33      192  24.0                     0.966   33        0
TRAIN_001           10      133             68              0        0  27.0                     0.245   36        0
TRAIN_002            4      112             78             40        0  39.4                     0.236   38        0
TRAIN_004            1      114             66             36      200  38.1                     0.289   21        0
TRAIN_647            1       91             64             24        0  29.2                     0.192   21        0
TRAIN_648           10      122             68              0        0  31.2                     0.258   41        0
TRAIN_649            8       84             74             31        0  38.3                     0.457   39        0
TRAIN_650            2       81             72             15       76  30.1                     0.547   25        0
TRAIN_651            1      107             68             19        0  26.5                     0.165   24        0
'''
'''print(trn_csv.info())
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Pregnancies               652 non-null    int64
 1   Glucose                   652 non-null    int64
 2   BloodPressure             652 non-null    int64
 3   SkinThickness             652 non-null    int64
 4   Insulin                   652 non-null    int64
 5   BMI                       652 non-null    float64
 6   DiabetesPedigreeFunction  652 non-null    float64
 7   Age                       652 non-null    int64
 8   Outcome                   652 non-null    int64
'''

tst_csv = pd.read_csv(path + "test.csv", index_col=0)
'''print(tst_csv) [116 rows x 8 columns]
           ID  Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age
0    TEST_000            5      112             66              0        0  37.8                     0.261   41
1    TEST_001            3      107             62             13       48  22.9                     0.678   23
2    TEST_002            3      113             44             13        0  22.4                     0.140   22
4    TEST_004            1      107             72             30       82  30.8                     0.821   24
111  TEST_111           10      111             70             27        0  27.5                     0.141   40
112  TEST_112            1      119             54             13       50  22.3                     0.205   24
113  TEST_113            3      187             70             22      200  36.4                     0.408   36
114  TEST_114            3      100             68             23       81  31.6                     0.949   28
115  TEST_115            2       84              0              0        0   0.0                     0.304   21
'''
'''print(tst_csv.info())
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   Pregnancies               116 non-null    int64
 1   Glucose                   116 non-null    int64
 2   BloodPressure             116 non-null    int64
 3   SkinThickness             116 non-null    int64
 4   Insulin                   116 non-null    int64
 5   BMI                       116 non-null    float64
 6   DiabetesPedigreeFunction  116 non-null    float64
 7   Age                       116 non-null    int64
'''

sub_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)
''' print(sub_csv) [116 rows x 1 columns]
          Outcome
ID
TEST_000        0
TEST_001        0
TEST_002        0
TEST_003        0
TEST_004        0
...           ...
TEST_111        0
TEST_112        0
TEST_113        0
TEST_114        0
TEST_115        0
'''

## nan_null / 0이 nan 값, 대체!

x = trn_csv.drop(['Outcome'], axis=1)
y = trn_csv['Outcome']

x = x.replace(0, np.nan)
x = x.fillna(x.mean())
# print(x.info())

tst_csv = tst_csv.replace(0, np.nan)
tst_csv = tst_csv.fillna(tst_csv.mean())

C = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)

MS = MinMaxScaler()

MS.fit(x_trn[C])
x_trn[C] = MS.transform(x_trn[C])
tst_csv[C] = MS.transform(tst_csv[C])

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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model = RandomizedSearchCV(XGBClassifier(),       
                     PM,           
n_iter=30,
                     cv = KF,     
                     verbose=1, 
                     refit=True,  
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

saveNum = '07'
import pandas as pd
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True))
# 모델을 돌려본 기준으로 점수가 높은 순서대로 정렬해서 dataframe 형태로 출력

print(pd.DataFrame(model.cv_results_).columns)

path = './Study25/_save/m18_RS_results/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True).to_csv(path + f'm18_{saveNum}_GSCV_results.csv')

path = './Study25/_save/m18_RS_save/'

import joblib
print('훈련 소요시간 :', time.time() - S)
joblib.dump(model.best_estimator_, path + f'm18_{saveNum}_best_model.joblib')
print(saveNum, '저장완료')