# 최적의 Learning rate???
# 너무 작으면 갱신이 힘들고
# 너무 크거 일정 범위 안에서 핑퐁 칠 수 있음

# 초기에는 큰 값을 줬다가, 갱신이 안되면 수치를 줄여준다 (EarlyStopping이랑 비슷)

### 31-2.copy
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input

                                    # 
# import sklearn as sk
# print(sk.__version__)   # 1.1.3
# import tensorflow as tf
# print(tf.__version__)   # 2.9.3

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
# print(x)
# print(y)        # 데이터를 찍어보고 소수점(회귀) or 정수 몇개로만 구성(분류) AI 모델 종류 결정
# print(x.shape)  # (20640, 8)
# print(y.shape)  # (20640,)
# print(datasets.feature_names)
#                 # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size= 0.2,
                                              shuffle= True,
                                              random_state= 55)

### Scaling

MS = MinMaxScaler()

MS.fit(x_trn)

x_trn = MS.transform(x_trn)
x_tst = MS.transform(x_tst)


#2. 모델
def MODEL(drop = 0.5, optimizer = 'adam',
          activation1 = 'relu', activation2 = 'relu',
          activation3 = 'relu', activation4 = 'relu',
          activation5 = 'relu',
          node1 = 128, node2 = 64, node3 = 32,
          node4 = 16, node5 = 8, lr = 0.001):
    
    inputs = Input(shape=(8,), name = 'inputs')
    
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
    epochs=50,
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

# 훈련 소요시간 : 5530.565045118332
# 최적 파라미터 : {'optimizer': 'rmsprop',
#            'node5': 64, 'node4': 64, 'node3': 128, 'node2': 32, 'node1': 128,
#            'drop': 0.3, 'batch_size': 1,
#            'activation5': 'linear', 'activation4': 'linear', 'activation3': 'selu', 'activation2': 'relu', 'activation1': 'relu'}