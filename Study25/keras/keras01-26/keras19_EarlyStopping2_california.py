# 18-2.copy

import sklearn as sk
print(sk.__version__)   # 1.1.3
import tensorflow as tf
print(tf.__version__)   # 2.9.3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x)
print(y)        # 데이터를 찍어보고 소수점(회귀) or 정수 몇개로만 구성(분류) AI 모델 종류 결정
print(x.shape)  # (20640, 8)
print(y.shape)  # (20640,)
print(datasets.feature_names)
                # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size= 0.2,
                                              shuffle= True,
                                              random_state= 55)

#2. 모델구성
model = Sequential()
model.add(Dense(8,input_dim=8))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

epochs = 100000

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

ES = EarlyStopping(monitor='val_loss',
                   mode= 'min',
                   patience= 100,
                   restore_best_weights= True)

hist = model.fit(x_trn, y_trn, epochs = epochs, batch_size = 32,
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
plt.title('캘리포니아 loss') # 표 제목 한글 깨짐해결법 피룡
plt.xlabel('epochs') # x 축 이름
plt.ylabel('loss') # y 축 이름
plt.legend(loc='upper right') # 우측상단 라벨표시 / defualt 는 빈자리
plt.grid() # 격자표시


#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict([x_tst])

from sklearn.metrics import r2_score, mean_squared_error
def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a,b))

rmse = RMSE(y_tst, results)
R2 = r2_score(y_tst, results)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', loss)
print('rmse :', rmse)
print('R2 :', R2)
plt.show()

'''##[기존]
# trn=0.85 / RS= 55 / e = 5000 / HL 8-64-128-128-64-64-1 / BS=50
# loss : 0.5563498139381409
# rmse : 0.7458885803295967
# R2 : 0.5843834853462511 ***********

##[갱신_validation]
# loss : 0.5707632899284363
# rmse : 0.7554887341834895
# R2 : 0.5736160338881686

#[갱신_ES]
# trn=0.85 / RS= 55 / e = 100000 / HL 8-64-128-128-64-64-1 / BS=50 patient : 10 / restore : True
loss : 0.6777143478393555
rmse : 0.8232339960620744
R2 : 0.4985546354799101

patient : 20 / restore : True
loss : 0.6726770997047424
rmse : 0.8201689225358081
R2 : 0.5022816576710789

patient : 100 / restore : True
loss : 0.6480043530464172
rmse : 0.8049872322613549
R2 : 0.5205370954302879
'''