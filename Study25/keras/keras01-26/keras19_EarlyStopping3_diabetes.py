#18-3.copy

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) (442,)

# [실습] R2 0.62 이상
x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size = 0.85,
                                              shuffle = True,
                                              random_state=6974)

#2. 모델구성
model = Sequential()
model.add(Dense(400, input_dim = 10))
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(1))

epochs = 100000

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')

ES = EarlyStopping(monitor= 'val_loss',
                   mode= 'min',
                   patience= 500,
                   restore_best_weights=False)

hist = model.fit(x_trn, y_trn, epochs = epochs, batch_size=32,
          verbose=2,
          validation_split=0.2,
          callbacks= [ES])

# print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡhistㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
# print(hist)
# print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡhist.historyㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
# print(hist.history)
# print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡlossㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
# print(hist.history['loss'])
# print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡval_lossㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
# print(hist.history['val_loss'])

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows일 경우
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지

plt.figure(figsize=(9,6)) # 9*6 사이즈 (단위 확인 피룡) # 크기설정이 1번
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss') # y=epochs로 시간 따라서 자동으로
plt.title('당뇨 loss') # 표 제목 한글 깨짐해결법 피룡
plt.xlabel('epochs') # x 축 이름
plt.ylabel('loss') # y 축 이름
plt.legend(loc='upper right') # 우측상단 라벨표시 / defualt 는 빈자리
plt.grid() # 격자표시

#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
R2 = r2_score(y_tst,results)

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print("loss:", loss)
print("R2  :", R2)
plt.show() # 이건 반드시 제일 아래

'''
#[기존]
# train_size : 0.85 / random_state : 6974 / epochs : 500 / hidden_layer : 10 400 400 200 200 1 / batch_size : 3
# R2 : 0.6191175928790672
'''
'''
#[갱신]
# R2 : 0.6522464497045597
'''

''' [갱신_ES] 
train_size : 0.85 / random_state : 6974 / epochs : 100000 / hidden_layer : 10 400 400 200 200 1 / batch_size : 32
patient : 10 / restore : True
loss: 2469.559814453125
R2  : 0.6292140569563287

train_size : 0.85 / random_state : 6974 / epochs : 100000 / hidden_layer : 10 400 400 200 200 1 / batch_size : 32
patient : 100 / restore : True
loss: 2516.02978515625
R2  : 0.6222369790405421


train_size : 0.85 / random_state : 6974 / epochs : 100000 / hidden_layer : 10 400 400 200 200 1 / batch_size : 3
patient : 100 / restore : True
loss: 2381.70654296875
R2  : 0.642404610636015

train_size : 0.85 / random_state : 6974 / epochs : 100000 / hidden_layer : 10 400 400 200 200 1 / batch_size : 3
patient : 200 / restore : True
loss: 2349.482177734375
R2  : 0.647242802284121

train_size : 0.85 / random_state : 6974 / epochs : 100000 / hidden_layer : 10 400 400 200 200 1 / batch_size : 3
patient : 200 / restore : True
loss: 2328.2568359375
R2  : 0.6504296499740558

train_size : 0.85 / random_state : 6974 / epochs : 100000 / hidden_layer : 10 400 400 200 200 1 / batch_size : 3
patient : 500 / restore : True
loss: 2317.60888671875
R2  : 0.6520283880549829

train_size : 0.85 / random_state : 6974 / epochs : 100000 / hidden_layer : 10 400 400 200 200 1 / batch_size : 3
patient : 500 / restore : False
loss: 2334.05126953125
R2  : 0.6495596514551192
'''

