# 08_kaggle_bank

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.utils import all_estimators
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

#1. 데이터
path = './Study25/_data/kaggle/bank/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)

LE_GEO = LabelEncoder()
LE_GEN = LabelEncoder()

# trn_csv['Surname']
trn_csv['Geography'] = LE_GEO.fit_transform(trn_csv['Geography'])
trn_csv['Gender'] = LE_GEN.fit_transform(trn_csv['Gender'])
tst_csv['Geography'] = LE_GEN.fit_transform(tst_csv['Geography'])
tst_csv['Gender'] = LE_GEN.fit_transform(tst_csv['Gender'])

trn_csv = trn_csv.drop(['CustomerId','Surname'], axis=1)
tst_csv = tst_csv.drop(['CustomerId','Surname'], axis=1)

x = trn_csv.drop(['Exited'], axis=1)
y = trn_csv['Exited']

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size=0.9,
    shuffle=True,
    random_state=777
)

NS = 5

KF = StratifiedKFold(n_splits=NS,
                     shuffle=True,
                     random_state=333)

#2. 모델
model_list = all_estimators(type_filter='classifier')

max_score = 0
max_model = 'default'

for name, model in model_list:
    
    try:
        #3. 훈련        
        model = model()
        score = cross_val_score(model, x_trn, y_trn, cv = KF)
        
        #4. 평가 예측        
        y_prd = cross_val_predict(model, x_tst, y_tst, cv = KF)
        ACC = accuracy_score(y_tst, y_prd)
        print(name, '의 정답률 :', ACC)
        if ACC > max_score:                  
            max_score = ACC
            max_model = name        
    except:
        print(name, ': ERROR')

print('최고모델 :', max_model, max_score)

# 최고모델 : GradientBoostingClassifier 0.855550169655841