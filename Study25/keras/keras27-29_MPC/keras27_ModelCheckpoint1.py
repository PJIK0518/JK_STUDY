### 26-5.copy

import sklearn as sk
# print(sk.__version__)   1.1.3

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import numpy as np

#1. 데이터
DS = load_boston()

x = DS.data
y = DS.target

### print(x.shape, y.shape) (506, 13) (506,)

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size=0.8,
                                              shuffle=True,
                                              random_state=333)

### scaling
from sklearn.preprocessing import MinMaxScaler
MS = MinMaxScaler()     # 인스턴스 = 클래스를 변수화 시킨다
MS.fit(x_trn)
x_trn = MS.transform(x_trn)
x_tst = MS.transform(x_tst)

# print(np.min(x_trn), np.max(x_trn)) : 0.0, 1.0000000000000002
# print(np.min(x_tst), np.max(x_tst)) : -0.00557837618540494, 1.1478180091225068
    # a = 0.1
    # b = 0.2
    # print(a+b) 0.30000000000000004
    # : 이진법을 활용한 부동소수점 연산이기 때문에 약간의 오차가 발생

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim = 13))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(1))

# model.summary()

# path = './_save/keras26/'
# model.save(path + 'keras26_1_save.h5')
'''loss : 25.91206169128418
rmse : 5.090389149297348
r2 스코어 : 0.735804158651754'''

# model.save_weights(path + 'keras26_5_save1.h5')

# exit()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

ES = EarlyStopping(monitor= 'val_loss',               # 어떤 수치를 모니터링 할 것인가
                   mode= 'min',                       # 최대값 : max, 알아서 찾아라 : auto
                   patience= 10, verbose=1,           # 몇 번이나 참을것이냐, 통상적으로 커야지만 너무 크면 시간 낭비가 될 수도
                   restore_best_weights= True)       # 가장 최소 지점을 저장 한다 // Default : False, False가 성적이 더 좋을수도...

path = './_save/keras27_mcp/'
MCP = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      verbose=1,                           # val_loss의 개선 여부와 저장 시점을 출력, ES, MCP, Fit에 각각 verbose를 넣어서 원하는 수준으로 훈련 진행 상황을 볼 수 있다.
                      filepath= path + 'keras27_MCP1.hdf5' # 확장자의 경우 h5랑 같음
                                                           # patience 만큼 지나기전 최저 갱신 지점        
                      )


hist = model.fit(x_trn, y_trn,
                 epochs = 100,
                 batch_size = 32,
                 verbose = 2,
                 validation_split = 0.2,
                 callbacks = [ES, MCP])                    # 두 개 이상을 불러올수도 있다

'''
loss : 27.116012573242188
rmse : 5.2073037719382365
r2 스코어 : 0.7235287791378253'''

# path = './_save/keras26/'
# # model.save(path + 'keras26_3_save.h5')
# model.save_weights(path + 'keras26_5_save2.h5')


'''
loss : 23.79562759399414
rmse : 4.878076218551135
r2 스코어 : 0.75738297784978
'''

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡhistㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
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


#4. 평가,예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
r2 = r2_score(y_tst, results)
rmse = np.sqrt(loss)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss)
print("rmse :", rmse)
print('r2 스코어 :', r2)
