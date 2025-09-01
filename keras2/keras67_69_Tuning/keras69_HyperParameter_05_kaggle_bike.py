# 31-5.copy

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#1. 데이터

path = './Study25/_data/kaggle/bike/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test_new_0527_1.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = trn_csv.drop(['count'], axis = 1)
y = trn_csv['count']

'''print(x) [10886 rows x 10 columns]    season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered
                     season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered
datetime
2011-01-01 00:00:00       1        0           0        1   9.84  14.395        81     0.0000       3          13
2011-01-01 01:00:00       1        0           0        1   9.02  13.635        80     0.0000       8          32
2011-01-01 02:00:00       1        0           0        1   9.02  13.635        80     0.0000       5          27
2011-01-01 03:00:00       1        0           0        1   9.84  14.395        75     0.0000       3          10
2011-01-01 04:00:00       1        0           0        1   9.84  14.395        75     0.0000       0           1
...                     ...      ...         ...      ...    ...     ...       ...        ...     ...         ...
2012-12-19 19:00:00       4        0           1        1  15.58  19.695        50    26.0027       7         329
2012-12-19 20:00:00       4        0           1        1  14.76  17.425        57    15.0013      10         231
2012-12-19 21:00:00       4        0           1        1  13.94  15.910        61    15.0013       4         164
2012-12-19 22:00:00       4        0           1        1  13.94  17.425        61     6.0032      12         117
2012-12-19 23:00:00       4        0           1        1  13.12  16.665        66     8.9981       4          84
'''
'''print(y) [10886 rows x 1 columns]     count
datetime
2011-01-01 00:00:00     16
2011-01-01 01:00:00     40
2011-01-01 02:00:00     32
2011-01-01 03:00:00     13
2011-01-01 04:00:00      1
                      ...
2012-12-19 19:00:00    336
2012-12-19 20:00:00    241
2012-12-19 21:00:00    168
2012-12-19 22:00:00    129
2012-12-19 23:00:00     88
'''

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size=0.2,
                                              random_state=42)

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


# scr : 0.9997813105583191
# lss : 7.218948841094971

# 최적 파라미터 : {'optimizer': 'adam', 'node5': 16, 'node4': 32, 'node3': 128, 'node2': 128, 'node1': 16, 'drop': 0.3, 'batch_size': 16, 'activation5': 'selu', 'activation4': 'selu', 'activation3': 'selu', 'activation2': 'relu', 'activation1': 'relu'}
# 훈련 최고점수 : 0.6564041011999283
# 137/137 [==============================] - 0s 688us/step
# 최고 성능평가 : 0.6280435924424077