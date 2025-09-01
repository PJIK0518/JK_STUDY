# 최적의 Learning rate???
# 너무 작으면 갱신이 힘들고
# 너무 크거 일정 범위 안에서 핑퐁 칠 수 있음

# 초기에는 큰 값을 줬다가, 갱신이 안되면 수치를 줄여준다 (EarlyStopping이랑 비슷)

### 31-2.copy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau


best_op = []
best_lr = 0
best_sc = -10000


model.compile(loss = 'mse', optimizer = Adam(learning_rate=0.01))


ES = EarlyStopping(monitor='val_loss',
                mode= 'min',
                patience= 50,
                restore_best_weights= True)

RLR = ReduceLROnPlateau(monitor = 'val_loss',
                        mode = 'auto',
                        patience = 10,
                        verbose = 1,
                        factor = 0.5)
                    # patience 만큼 갱신되지 않으면 해당 비율만큼 lr 하강(곱하기)


hist = model.fit(x_trn, y_trn, epochs = 10000, batch_size = 32,
        verbose=2,
        validation_split=0.2,
        callbacks = [ES, RLR]
        )

loss = model.evaluate(x_tst, y_tst)
results = model.predict([x_tst])

R2 = r2_score(y_tst, results)

print('score        :',R2)

# ES
# score        : 0.6947352053602394

# ES, RLR
# score        : 0.7220349255890997