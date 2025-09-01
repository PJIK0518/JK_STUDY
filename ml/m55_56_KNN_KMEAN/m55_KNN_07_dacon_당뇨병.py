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
    random_state=777,
    stratify=y
)

RS = 777

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

#2 모델구성
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score
model = KNeighborsClassifier(n_neighbors=5)     # k_neighbors랑 같은 Parameter

model.fit(x_trn, y_trn)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ', model.__class__.__name__,'ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('acc :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

score1 = accuracy_score(y_tst, y_prd)
score2 = f1_score(y_tst, y_prd)

print('ACC :',score1)
print('F1S :',score2)
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ KNeighborsClassifier ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# acc : 0.5909090909090909
# ACC : 0.5909090909090909
# F1S : 0.4
# 최적 컬럼 : 7 개 
#  ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
# 삭제 컬럼 : 1 개 
#  ['Pregnancies']
# 최고 점수 72.727%