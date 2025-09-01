# 63_3.copy
##########################################################################
#0. 준비
##########################################################################
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import time
import keras.metrics as metrics
print(dir(metrics))
exit()
##########################################################################
#1. 데이터
##########################################################################
x_datasets = np.array([range(100), range(301,401)]).T
# (100, 2)              삼성전자 종가, 하이닉스 종가

y1 = np.array(range(2001, 2101))
# (100,)           화성의 화씨 온도

y2 = np.array(range(13001, 13101))

# 이전 방식 : x1, x2를 합쳐서 모델 제작 후 > y 예측

x_trn, x_tst, y1_trn, y1_tst, y2_trn, y2_tst = train_test_split(
    x_datasets,  y1, y2,
    train_size=0.7,
    random_state=42)

##########################################################################
#2. 모델 제작
##########################################################################
### 모델 X
inputX = Input(shape=(2,))

upper1 = Dense(64, activation='relu')(inputX)
upper2 = Dense(64, activation='relu')(upper1)
upper3 = Dense(64, activation='relu')(upper2)
upper4 = Dense(64, activation='relu')(upper3)
upper5 = Dense(64, activation='relu')(upper4)
OutptX = Dense(64, activation='relu')(upper5)

#####################################
### 모델 Y1

lowr11 = Dense(32, activation='relu')(OutptX)
lowr21 = Dense(32, activation='relu')(lowr11)
lowr31 = Dense(32, activation='relu')(lowr21)
OtptY1 = Dense(1, activation='relu')(lowr31)

#####################################
### 모델 Y2

lowr12 = Dense(32, activation='relu')(OutptX)
lowr22 = Dense(32, activation='relu')(lowr21)
lowe32 = Dense(32, activation='relu')(lowr22)
OtptY2 = Dense(1, activation='relu')(lowe32)

M = Model(inputs = inputX, outputs = [OtptY1, OtptY2])

##########################################################################
#3. 컴파일 훈련
##########################################################################
M.compile(loss = 'mse',
          optimizer = 'adam')

M.fit(x_trn, [y1_trn, y2_trn],
      epochs = 500,
      verbose = 3,
      validation_split = 0.2)



##########################################################################
#4. 평가 훈련
##########################################################################

loss = M.evaluate(x_tst, [y1_tst, y2_tst])

x_prd = np.array([range(100,106), range(400,406)]).T

rslt = M.predict(x_prd)

print(loss)
print(rslt)
""" 
[566.368896484375, 553.2577514648438, 13.111173629760742]
[array([[2056.08  ],
       [2058.7546],
       [2061.4292],
       [2064.104 ],
       [2066.7788],
       [2069.4536]], dtype=float32), array([[13087.273],        
       [13104.297],
       [13121.319],
       [13138.344],
       [13155.366],
       [13172.389]], dtype=float32)] """