#17_1.copy

import sklearn as sk
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

#1. 데이터
dataset = load_boston()    
x = dataset.data
y = dataset.target
# print(x)
# print(x.shape) # (506, 13)
# print(y)
# print(y.shape) # (506,)
x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size=0.15,
                                              shuffle=True,
                                              random_state=55)

#2. 모델구성
model = Sequential()
model.add(Dense(128,input_dim = 13, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

e = 200
b = 32
v = 0.2

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
hist = model.fit(x_trn, y_trn,
                 epochs = e,
                 batch_size = b,
                 verbose = 2,
                 validation_split = v)

'''print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡhistㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print(hist) # <keras.callbacks.History object at 0x0000011E9C490BE0>
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡhist.historyㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print(hist.history)
# [126.57898712158203, 85.05742645263672, 79.80768585205078, 73.9778823852539, 72.70500946044922, 68.3889389038086, 65.2877426147461, 70.47799682617188, 64.71198272705078, 62.79868698120117]
# [49.95176315307617, 68.548095703125, 50.90830612182617, 66.19173431396484, 56.54918670654297, 50.720420837402344, 85.67217254638672, 41.672603607177734, 41.31922149658203, 45.64241027832031]
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡlossㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print(hist.history['loss'])
# [122.76151275634766, 91.21681213378906, 75.81005096435547, 79.3133316040039, 69.39814758300781, 68.55708312988281, 66.84475708007812, 63.747955322265625, 60.12713623046875, 63.09190368652344]
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡval_lossㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print(hist.history['val_loss'])
# [51.9560546875, 51.53378677368164, 61.10300064086914, 51.60072326660156, 52.848106384277344, 46.44329071044922, 43.22845458984375, 39.889644622802734, 36.81156921386719, 36.20408248901367]
'''

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows일 경우
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지

plt.figure(figsize=(9,6)) # 9*6 사이즈 (단위 확인 피룡) # 크기설정이 1번
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='blue', label='val_loss') # y=epochs로 시간 따라서 자동으로
plt.title('보스턴 loss') # 표 제목 한글 깨짐해결법 피룡
plt.xlabel('epochs') # x 축 이름
plt.ylabel('loss') # y 축 이름
plt.legend(loc='upper right') # 우측상단 라벨표시 / defualt 는 빈자리
plt.grid() # 격자표시
plt.show() # 이건 반드시 제일 아래

#4. 평가,예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x)
r2 = r2_score(y, results)
rmse = np.sqrt(loss)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss)
print("rmse :", rmse)
print('r2 스코어 :', r2)

##[기존 최고치]
# train_size : 0.85 / epochs : 1200 / batch_size : 3 / hidden_layer : 13-64-128-128-64-1 / random_state : 42
# loss : 13.933738708496094
# r2 스코어 : 0.7252240190334657 ***********

''' validation 적용
train_size : 0.85 / epochs : 1200 / batch_size : 3 / hidden_layer : 13-64-128-128-64-1 / random_state : 42
loss : 26.947917938232422
r2 스코어 : 0.5879988937247986

loss : 24.593360900878906
r2 스코어 : 0.6985541922493144

train_size : 0.85 / epochs : 100 / batch_size : 3 / hidden_layer : 13-64-128-128-64-1 / random_state : 42 / relu
loss : 21.40628433227539
rmse : 4.626692591071444
r2 스코어 : 0.7670141293238002
'''