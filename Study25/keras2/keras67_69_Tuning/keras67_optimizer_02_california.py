### 31-2.copy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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


#2. 모델구성
model = Sequential()
model.add(Dense(128,input_dim=8, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

epochs = 100000

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop

optim = [Adam, Adagrad, SGD, RMSprop]
lrlst = [0.1, 0.01, 0.05, 0.001, 0.0001]

best_op = []
best_lr = 0
best_sc = -1000

for idx_1, op in enumerate(optim):
    for idx_2, lr in enumerate(lrlst):
        model.compile(loss = 'mse', optimizer = op(learning_rate=float(lr)))
        
        
        ES = EarlyStopping(monitor='val_loss',
                        mode= 'min',
                        patience= 100,
                        restore_best_weights= True)


        hist = model.fit(x_trn, y_trn, epochs = 100, batch_size = 32,
                verbose=0,
                validation_split=0.2)

        loss = model.evaluate(x_tst, y_tst)
        results = model.predict([x_tst])

        R2 = r2_score(y_tst, results)
        print(f'⏩⏩  {idx_1}/{len(optim)} | {idx_2}/{len(lrlst)}  ⏩⏩')
        print('optimizer    :', op)
        print('learing_rate :', lr)
        print('score        :',R2)
        if R2 >= best_sc:
            best_op = op
            best_lr = lr
            best_sc = R2

print('✅  DONE ✅')
print('최종옵티머 :', best_op)
print('최종학습률 :', best_lr)
print('최종모델능 :', best_sc)