# m10_13.copy

# tst, val : 데이터가 아깝다...
# But. 데이터의 과적합을 막기 위해서 필요한 과정
#      But. tst, val에 중요도가 높은 애들이 있다면??
#           >> tst에 한 번 썼던 데이터를 trn에 넣고 기존 trn에서 다시 tst 선정
#           >> 데이터 1/n 로 나눠서 n 반복 : n_split (데이터가 많다면 n 수를 높여서 진행)
#           >> 데이터의 소실 없이 훈련 가능7
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
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

#1. 데이터
path = './Study25/_data/kaggle/otto/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

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
    a_scaled = SC.transform(a)
    b_scaled = SC.transform(b)
    return a_scaled, b_scaled

x_trn, x_tst = Scaler(MaxAbsScaler(), x_trn, x_tst)

#####################################
## 증폭 : class_weight
classes = np.unique(y)
CW = compute_class_weight(class_weight='balanced', classes=classes, y=y)

#####################################
## Label_y
LE = LabelEncoder()
LE.fit(y_trn)
y_trn = LE.transform(y_trn)
y_tst = LE.transform(y_tst)

from xgboost import XGBRegressor, XGBClassifier
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb

RS = 777

#2 모델구성
model = XGBClassifier(
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

from xgboost.plotting import plot_importance
import matplotlib.pyplot as plt
plot_importance(model)
plt.show()

# ORIG_SCR : 0.8099547511312217
# Quantile : 0.25
# DROP_SCR : 0.8989933970540703
# PLUS_SCR : 0.9048113773837558

# Threshold = 0.003 / n = 93 / R2 = 81.787%
# Threshold = 0.003 / n = 92 / R2 = 81.690%
# Threshold = 0.004 / n = 91 / R2 = 80.963%
# Threshold = 0.004 / n = 90 / R2 = 81.012%
# Threshold = 0.004 / n = 89 / R2 = 81.060%
# Threshold = 0.004 / n = 88 / R2 = 81.529%
# Threshold = 0.004 / n = 87 / R2 = 81.189%
# Threshold = 0.005 / n = 86 / R2 = 81.222%
# Threshold = 0.005 / n = 85 / R2 = 81.319%
# Threshold = 0.005 / n = 84 / R2 = 81.787%
# Threshold = 0.005 / n = 83 / R2 = 81.222%
# Threshold = 0.005 / n = 82 / R2 = 81.674%
# Threshold = 0.005 / n = 81 / R2 = 81.335%
# Threshold = 0.005 / n = 80 / R2 = 81.626%
# Threshold = 0.005 / n = 79 / R2 = 81.545%
# Threshold = 0.005 / n = 78 / R2 = 81.739%
# Threshold = 0.005 / n = 77 / R2 = 81.399%
# Threshold = 0.005 / n = 76 / R2 = 81.157%
# Threshold = 0.005 / n = 75 / R2 = 81.674%
# Threshold = 0.005 / n = 74 / R2 = 81.109%
# Threshold = 0.005 / n = 73 / R2 = 81.545%
# Threshold = 0.005 / n = 72 / R2 = 80.979%
# Threshold = 0.006 / n = 71 / R2 = 80.979%
# Threshold = 0.006 / n = 70 / R2 = 81.189%
# Threshold = 0.006 / n = 69 / R2 = 80.850%
# Threshold = 0.006 / n = 68 / R2 = 81.012%
# Threshold = 0.006 / n = 67 / R2 = 81.610%
# Threshold = 0.006 / n = 66 / R2 = 81.432%
# Threshold = 0.006 / n = 65 / R2 = 81.092%
# Threshold = 0.006 / n = 64 / R2 = 80.785%
# Threshold = 0.006 / n = 63 / R2 = 80.947%
# Threshold = 0.006 / n = 62 / R2 = 80.333%
# Threshold = 0.006 / n = 61 / R2 = 80.850%
# Threshold = 0.006 / n = 60 / R2 = 80.834%
# Threshold = 0.006 / n = 59 / R2 = 81.141%
# Threshold = 0.006 / n = 58 / R2 = 80.963%
# Threshold = 0.006 / n = 57 / R2 = 80.995%
# Threshold = 0.006 / n = 56 / R2 = 80.866%
# Threshold = 0.007 / n = 55 / R2 = 80.834%
# Threshold = 0.007 / n = 54 / R2 = 80.317%
# Threshold = 0.007 / n = 53 / R2 = 80.478%
# Threshold = 0.007 / n = 52 / R2 = 79.816%
# Threshold = 0.007 / n = 51 / R2 = 80.171%
# Threshold = 0.007 / n = 50 / R2 = 79.864%
# Threshold = 0.007 / n = 49 / R2 = 79.913%
# Threshold = 0.007 / n = 48 / R2 = 79.961%
# Threshold = 0.008 / n = 47 / R2 = 80.171%
# Threshold = 0.008 / n = 46 / R2 = 79.202%
# Threshold = 0.008 / n = 45 / R2 = 79.897%
# Threshold = 0.008 / n = 44 / R2 = 79.573%
# Threshold = 0.008 / n = 43 / R2 = 79.493%
# Threshold = 0.008 / n = 42 / R2 = 79.056%
# Threshold = 0.008 / n = 41 / R2 = 79.121%
# Threshold = 0.008 / n = 40 / R2 = 79.202%
# Threshold = 0.008 / n = 39 / R2 = 79.218%
# Threshold = 0.008 / n = 38 / R2 = 78.975%
# Threshold = 0.009 / n = 37 / R2 = 79.008%
# Threshold = 0.009 / n = 36 / R2 = 79.315%
# Threshold = 0.009 / n = 35 / R2 = 78.426%
# Threshold = 0.010 / n = 34 / R2 = 78.765%
# Threshold = 0.010 / n = 33 / R2 = 78.281%
# Threshold = 0.010 / n = 32 / R2 = 78.507%
# Threshold = 0.010 / n = 31 / R2 = 78.087%
# Threshold = 0.010 / n = 30 / R2 = 78.184%
# Threshold = 0.011 / n = 29 / R2 = 77.877%
# Threshold = 0.011 / n = 28 / R2 = 77.392%
# Threshold = 0.011 / n = 27 / R2 = 76.681%
# Threshold = 0.012 / n = 26 / R2 = 76.907%
# Threshold = 0.012 / n = 25 / R2 = 76.325%
# Threshold = 0.012 / n = 24 / R2 = 76.147%
# Threshold = 0.012 / n = 23 / R2 = 75.275%
# Threshold = 0.013 / n = 22 / R2 = 74.160%
# Threshold = 0.013 / n = 21 / R2 = 74.273%
# Threshold = 0.013 / n = 20 / R2 = 74.030%
# Threshold = 0.014 / n = 19 / R2 = 72.754%
# Threshold = 0.015 / n = 18 / R2 = 72.705%
# Threshold = 0.015 / n = 17 / R2 = 71.170%
# Threshold = 0.015 / n = 16 / R2 = 69.958%
# Threshold = 0.016 / n = 15 / R2 = 69.215%
# Threshold = 0.017 / n = 14 / R2 = 68.584%
# Threshold = 0.017 / n = 13 / R2 = 67.421%
# Threshold = 0.018 / n = 12 / R2 = 64.981%
# Threshold = 0.019 / n = 11 / R2 = 64.674%
# Threshold = 0.019 / n = 10 / R2 = 63.009%
# Threshold = 0.019 / n =  9 / R2 = 61.813%
# Threshold = 0.021 / n =  8 / R2 = 59.357%
# Threshold = 0.022 / n =  7 / R2 = 57.385%
# Threshold = 0.022 / n =  6 / R2 = 55.559%
# Threshold = 0.022 / n =  5 / R2 = 54.880%
# Threshold = 0.023 / n =  4 / R2 = 54.008%
# Threshold = 0.048 / n =  3 / R2 = 50.145%
# Threshold = 0.060 / n =  2 / R2 = 43.197%
# Threshold = 0.069 / n =  1 / R2 = 38.348%