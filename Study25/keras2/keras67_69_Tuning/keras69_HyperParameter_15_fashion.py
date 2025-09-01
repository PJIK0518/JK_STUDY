# 36-5.copy

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import fashion_mnist

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import time
##########################################################################
#1. 데이터
##########################################################################
(x_trn, y_trn), (x_tst, y_tst) = fashion_mnist.load_data()
# print(x_trn.shape) # (60000, 28, 28)
# print(x_tst.shape) # (10000, 28, 28)
# print(y_trn.shape) # (60000,)
# print(y_tst.shape) # (10000,)

#####################################
### x reshape 
x_trn = x_trn.reshape(60000,28,28,1)
x_tst = x_tst.reshape(x_tst.shape[0], x_tst.shape[1], x_tst.shape[2], 1)
                                            # 이렇게 입력해도 똑같이 인식함
# print(x_trn.shape, x_tst.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

#####################################
### y OneHot
y_trn = pd.get_dummies(y_trn)
y_tst = pd.get_dummies(y_tst)
# print(y_trn.shape, y_tst.shape) (60000, 10) (10000, 10)

#####################################
### Scaling
x_trn = (x_trn-127.5)/(510.0)
x_tst = (x_tst-127.5)/(510.0)

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
          epochs = 100,
          callbacks = [ES, RLR])

print('최적 매개변수 :', model.best_estimator_)
print('최적 파라미터 :', model.best_params_)

#4. 평가 예측
print('훈련 최고점수 :', model.best_score_)
print('최고 성능평가 :', model.score(x_tst, y_tst))

y_prd = model.predict(x_tst)

y_prd_best = model.best_estimator_.predict(x_tst)

print('훈련 소요시간 :', time.time() - S)

'''
loss : 0.22302168607711792
acc  : 0.9302999973297119
acc  : 0.9303
시간 : 1882.5303266048431

loss : 0.20780806243419647
acc  : 0.9316999912261963
acc  : 0.9317
시간 : 1904.9725253582


'''


