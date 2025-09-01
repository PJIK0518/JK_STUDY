### https://www.kaggle.com/competitions/playground-series-s4e1/submissions
# 31-8.copy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

#1. 데이터
path = './Study25/_data/kaggle/bank/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
'''print(trn_csv) [165034 rows x 13 columns]
        CustomerId         Surname  CreditScore Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
id
0         15674932  Okwudilichukwu          668    France    Male  33.0       3       0.00              2        1.0             0.0        181449.97       0
1         15749177   Okwudiliolisa          627    France    Male  33.0       1       0.00              2        1.0             1.0         49503.50       0
2         15694510           Hsueh          678    France    Male  40.0      10       0.00              2        1.0             0.0        184866.69       0
3         15741417             Kao          581    France    Male  34.0       2  148882.54              1        1.0             1.0         84560.88       0
4         15766172       Chiemenam          716     Spain    Male  33.0       5       0.00              2        1.0             1.0         15068.83       0
...            ...             ...          ...       ...     ...   ...     ...        ...            ...        ...             ...              ...     ...
165029    15667085            Meng          667     Spain  Female  33.0       2       0.00              1        1.0             1.0        131834.75       0
165030    15665521       Okechukwu          792    France    Male  35.0       3       0.00              1        0.0             0.0        131834.45       0
165031    15664752            Hsia          565    France    Male  31.0       5       0.00              1        1.0             1.0        127429.56       0
165032    15689614          Hsiung          554     Spain  Female  30.0       7  161533.00              1        0.0             1.0         71173.03       0
165033    15732798         Ulyanov          850    France    Male  31.0       1       0.00              1        1.0             0.0         61581.79       1
'''
'''print(trn_csv) : index_col = 0 
        CustomerId         Surname  CreditScore Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
id
0         15674932  Okwudilichukwu          668    France    Male  33.0       3       0.00              2        1.0             0.0        181449.97       0
1         15749177   Okwudiliolisa          627    France    Male  33.0       1       0.00              2        1.0             1.0         49503.50       0
2         15694510           Hsueh          678    France    Male  40.0      10       0.00              2        1.0             0.0        184866.69       0
3         15741417             Kao          581    France    Male  34.0       2  148882.54              1        1.0             1.0         84560.88       0
4         15766172       Chiemenam          716     Spain    Male  33.0       5       0.00              2        1.0             1.0         15068.83       0
...            ...             ...          ...       ...     ...   ...     ...        ...            ...        ...             ...              ...     ...
165029    15667085            Meng          667     Spain  Female  33.0       2       0.00              1        1.0             1.0        131834.75       0
165030    15665521       Okechukwu          792    France    Male  35.0       3       0.00              1        0.0             0.0        131834.45       0
165031    15664752            Hsia          565    France    Male  31.0       5       0.00              1        1.0             1.0        127429.56       0
165032    15689614          Hsiung          554     Spain  Female  30.0       7  161533.00              1        0.0             1.0         71173.03       0
165033    15732798         Ulyanov          850    France    Male  31.0       1       0.00              1        1.0             0.0         61581.79       1
'''
'''print(trn_csv) : index_col = 1
                id         Surname  CreditScore Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
CustomerId
15674932         0  Okwudilichukwu          668    France    Male  33.0       3       0.00              2        1.0             0.0        181449.97       0
15749177         1   Okwudiliolisa          627    France    Male  33.0       1       0.00              2        1.0             1.0         49503.50       0
15694510         2           Hsueh          678    France    Male  40.0      10       0.00              2        1.0             0.0        184866.69       0
15741417         3             Kao          581    France    Male  34.0       2  148882.54              1        1.0             1.0         84560.88       0
15766172         4       Chiemenam          716     Spain    Male  33.0       5       0.00              2        1.0             1.0         15068.83       0
...            ...             ...          ...       ...     ...   ...     ...        ...            ...        ...             ...              ...     ...
15667085    165029            Meng          667     Spain  Female  33.0       2       0.00              1        1.0             1.0        131834.75       0
15665521    165030       Okechukwu          792    France    Male  35.0       3       0.00              1        0.0             0.0        131834.45       0
15664752    165031            Hsia          565    France    Male  31.0       5       0.00              1        1.0             1.0        127429.56       0
15689614    165032          Hsiung          554     Spain  Female  30.0       7  161533.00              1        0.0             1.0         71173.03       0
15732798    165033         Ulyanov          850    France    Male  31.0       1       0.00              1        1.0             0.0         61581.79       1
'''
'''print(trn_csv.head()) 앞에서부터 5개(Default)
            id         Surname  CreditScore Geography Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
CustomerId
15674932     0  Okwudilichukwu          668    France   Male  33.0       3       0.00              2        1.0             0.0        181449.97       0
15749177     1   Okwudiliolisa          627    France   Male  33.0       1       0.00              2        1.0             1.0         49503.50       0
15694510     2           Hsueh          678    France   Male  40.0      10       0.00              2        1.0             0.0        184866.69       0
15741417     3             Kao          581    France   Male  34.0       2  148882.54              1        1.0             1.0         84560.88       0
15766172     4       Chiemenam          716     Spain   Male  33.0       5       0.00              2        1.0             1.0         15068.83       0
'''
'''print(trn_csv.tail()) 뒤에서부터 5개(Default)
                id    Surname  CreditScore Geography  Gender   Age  Tenure   Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
CustomerId
15667085    165029       Meng          667     Spain  Female  33.0       2       0.0              1        1.0             1.0        131834.75       0
15665521    165030  Okechukwu          792    France    Male  35.0       3       0.0              1        0.0             0.0        131834.45       0
15664752    165031       Hsia          565    France    Male  31.0       5       0.0              1        1.0             1.0        127429.56       0
15689614    165032     Hsiung          554     Spain  Female  30.0       7  161533.0              1        0.0             1.0         71173.03       0
15732798    165033    Ulyanov          850    France    Male  31.0       1       0.0              1        1.0             0.0         61581.79       1
'''
'''print(trn_csv.isna().sum())
id                 0
Surname            0
CreditScore        0
Geography          0
Gender             0
Age                0
Tenure             0
Balance            0
NumOfProducts      0
HasCrCard          0
IsActiveMember     0
EstimatedSalary    0
Exited             0
'''



tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
'''print(tst_csv) [110023 rows x 12 columns]
        CustomerId    Surname  CreditScore Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary
id
165034    15773898   Lucchese          586    France  Female  23.0       2       0.00              2        0.0             1.0        160976.75
165035    15782418       Nott          683    France  Female  46.0       2       0.00              1        1.0             0.0         72549.27
165036    15807120         K?          656    France  Female  34.0       7       0.00              2        1.0             0.0        138882.09
165037    15808905  O'Donnell          681    France    Male  36.0       8       0.00              1        1.0             0.0        113931.57
165038    15607314    Higgins          752   Germany    Male  38.0      10  121263.62              1        1.0             0.0        139431.00
...            ...        ...          ...       ...     ...   ...     ...        ...            ...        ...             ...              ...
275052    15662091      P'eng          570     Spain    Male  29.0       7  116099.82              1        1.0             1.0        148087.62
275053    15774133        Cox          575    France  Female  36.0       4  178032.53              1        1.0             1.0         42181.68
275054    15728456      Ch'iu          712    France    Male  31.0       2       0.00              2        1.0             0.0         16287.38
275055    15687541   Yegorova          709    France  Female  32.0       3       0.00              1        1.0             1.0        158816.58
275056    15663942       Tuan          621    France  Female  37.0       7   87848.39              1        1.0             0.0         24210.56
'''
'''print(tst_csv.isna().sum())
CustomerId         0
Surname            0
CreditScore        0
Geography          0
Gender             0
Age                0
Tenure             0
Balance            0
NumOfProducts      0
HasCrCard          0
IsActiveMember     0
EstimatedSalary    0
'''

sub_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 문자 데이터의 수치화
#  CustomerId Surname  CreditScore Geography  Gender Age  Tenure  Balance  NumOfProducts  HasCrCard
#  IsActiveMember  EstimatedSalary  Exited

from sklearn.preprocessing import LabelEncoder

LE_GEO = LabelEncoder()         # class의 정의 : instance화 시킨다 >> 사용하기 쉽게 만들어 놓는다
LE_GEN = LabelEncoder()         # column의 값이 너무 크면, 함수 정의한 이후에 충돌할수도 
                                # >> 각각 

# trn_csv['Surname']
trn_csv['Geography'] = LE_GEO.fit_transform(trn_csv['Geography'])   # 특정 데이터 내 컬럼을 labeling 시켜 변환 및 적용시킨다
trn_csv['Gender'] = LE_GEN.fit_transform(trn_csv['Gender'])
tst_csv['Geography'] = LE_GEN.fit_transform(tst_csv['Geography'])
tst_csv['Gender'] = LE_GEN.fit_transform(tst_csv['Gender'])

# LE_GEN.fit(tst_csv['Gender'])
# LE_GEN.transform(tst_csv['Gender']) : 똑같은 기능인데 함수에 fit 시킨 후에 transform 진행

''' print(trn_csv['Geography'].value_counts())
0    94215
2    36213
1    34606
'''
''' print(trn_csv['Gender'].value_counts())
1    93150
0    71884
'''

trn_csv = trn_csv.drop(['CustomerId','Surname'], axis=1)
tst_csv = tst_csv.drop(['CustomerId','Surname'], axis=1)
''' print(trn_csv) [165034 rows x 11 columns]
        CreditScore  Geography  Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited
id
0               668          0       1  33.0       3       0.00              2        1.0             0.0        181449.97       0
1               627          0       1  33.0       1       0.00              2        1.0             1.0         49503.50       0
2               678          0       1  40.0      10       0.00              2        1.0             0.0        184866.69       0
3               581          0       1  34.0       2  148882.54              1        1.0             1.0         84560.88       0
4               716          2       1  33.0       5       0.00              2        1.0             1.0         15068.83       0
...             ...        ...     ...   ...     ...        ...            ...        ...             ...              ...     ...
165029          667          2       0  33.0       2       0.00              1        1.0             1.0        131834.75       0
165030          792          0       1  35.0       3       0.00              1        0.0             0.0        131834.45       0
165031          565          0       1  31.0       5       0.00              1        1.0             1.0        127429.56       0
165032          554          2       0  30.0       7  161533.00              1        0.0             1.0         71173.03       0
165033          850          0       1  31.0       1       0.00              1        1.0             0.0         61581.79       1
'''

x = trn_csv.drop(['Exited'], axis=1)
'''print(x) [165034 rows x 10 columns]
'''

y = trn_csv['Exited']
'''print(y)
id
0         0
1         0
2         0
3         0
4         0
         ..
165029    0
165030    0
165031    0
165032    0
165033    1
'''

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size=0.9,
                                              shuffle=True,
                                              random_state=55)


#2. 모델
def MODEL(drop = 0.5, optimizer = 'adam',
          activation1 = 'relu', activation2 = 'relu',
          activation3 = 'relu', activation4 = 'relu',
          activation5 = 'relu',
          node1 = 128, node2 = 64, node3 = 32,
          node4 = 16, node5 = 8, lr = 0.001):
    
    inputs = Input(shape=(10,), name = 'inputs')
    
    x = Dense(node1, activation=activation1, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    
    x = Dense(node2, activation=activation2, name = 'hidden2')(x)
    x = Dropout(drop)(x)
    
    x = Dense(node3, activation=activation3, name = 'hidden3')(x)
    x = Dropout(drop)(x)
    
    x = Dense(node4, activation=activation4, name = 'hidden4')(x)
    
    x = Dense(node5, activation=activation5, name = 'hidden5')(x)
    
    outputs = Dense(1, activation='linear', name = 'outputs')(x)
    
    model = Model(inputs= inputs, outputs = outputs)
    
    model.compile(optimizer= optimizer, loss = 'mse', metrics= ['mae'])
    
    return model

def creat_hyperparameter():
    batchs = [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activation1 = ['relu', 'elu', 'selu', 'linear']
    activation2 = ['relu', 'elu', 'selu', 'linear']
    activation3 = ['relu', 'elu', 'selu', 'linear']
    activation4 = ['relu', 'elu', 'selu', 'linear']
    activation5 = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16, 8]
    
    return {
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation1' : activation1,
        'activation2' : activation2,
        'activation3' : activation3,
        'activation4' : activation4,
        'activation5' : activation5,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,
    }

hyperparameters = creat_hyperparameter()
""" print(hyperparameters)
{'batch_size': [32, 16, 8, 1, 64], 'optimizer': ['adam', 'rmsprop', 'adadelta'], 'drop': [0.2, 0.3, 0.4, 0.5],
 'activation': ['relu', 'elu', 'selu', 'linear'], 'node1': [128, 64, 32, 16], 'node2': [128, 64, 32, 16],
 'node3': [128, 64, 32, 16], 'node4': [128, 64, 32, 16], 'node5': [128, 64, 32, 16, 8]} """

from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor

keras_model = KerasRegressor(
    model=MODEL,
    epochs=10,
    verbose=1,
    node1=128, node2=64, node3=32, node4=16, node5=8, drop=0.5,
    activation1 ='relu',
    activation2 ='relu',
    activation3 ='relu',
    activation4 ='relu',
    activation5 ='relu',
    optimizer='adam'
)

model = RandomizedSearchCV(
    keras_model, hyperparameters, cv =2,
    n_iter=5, verbose=1)
# n_iter : 최적 파라미터
# cv : 훈련 횟수
# 총 횟수 = (n_iter * cv + 1)

# sklearn.utils._param_validation.InvalidParameterError:
# The 'estimator' parameter of RandomizedSearchCV must be an object implementing 'fit'.
# Got <function MODEL at 0x773958287880> instead.

# : RandomizedSearchCV각 받아드리는 객체는 fit이 필요하다
# >>
import time

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

ES = EarlyStopping(monitor='loss',
                mode= 'min',
                patience= 50,
                restore_best_weights= True)

RLR = ReduceLROnPlateau(monitor = 'loss',
                        mode = 'auto',
                        patience = 10,
                        verbose = 1,
                        factor = 0.5)

#3. 훈련
S = time.time()
model.fit(x_trn, y_trn,
          epochs = 10,
          callbacks = [ES, RLR])

print('최적 매개변수 :', model.best_estimator_)
print('최적 파라미터 :', model.best_params_)

#4. 평가 예측
print('훈련 최고점수 :', model.best_score_)
print('최고 성능평가 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

y_prd_best = model.best_estimator_.predict(x_tst)

print('훈련 소요시간 :', time.time() - S)

# scr : 0.7909597673291323
# lss : 0.20904023945331573

# 최적 파라미터 : {'optimizer': 'rmsprop', 'node5': 16, 'node4': 32, 'node3': 64, 'node2': 16, 'node1': 32, 'drop': 0.2, 'batch_size': 32, 'activation5': 'relu', 'activation4': 'linear', 'activation3': 'elu', 'activation2': 'elu', 'activation1': 'selu'}
# 훈련 최고점수 : -0.0002025232303278779
# 516/516 [==============================] - 0s 699us/step
# 최고 성능평가 : -3.119793689765338e-05