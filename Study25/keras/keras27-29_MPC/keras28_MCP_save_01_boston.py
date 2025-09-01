### 27-5.copy

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
MS = MinMaxScaler()
MS.fit(x_trn)
x_trn = MS.transform(x_trn)
x_tst = MS.transform(x_tst)

#2. 모델
model = Sequential()
model.add(Dense(10, input_dim = 13))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

ES = EarlyStopping(monitor= 'val_loss',               # 어떤 수치를 모니터링 할 것인가
                   mode= 'min',                       # 최대값 : max, 알아서 찾아라 : auto
                   patience= 50,                      # 몇 번이나 참을것이냐, 통상적으로 커야지만 너무 크면 시간 낭비가 될 수도
                   restore_best_weights= True)       # 가장 최소 지점을 저장 한다 // Default : False, False가 성적이 더 좋을수도...

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
import datetime

path_MCP = './_save/keras28_mcp/01_boston/'

date = datetime.datetime.now()
# print(date)            
# print(type(date))       
date = date.strftime('%m%d_%H%M')              

# print(date)             
# print(type(date))

filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path_MCP,'keras28_',date, '_', filename])

MCP = ModelCheckpoint(monitor='val_loss',
                      mode='auto',
                      save_best_only=True,
                      filepath= filepath # 확장자의 경우 h5랑 같음
                                         # patience 만큼 지나기전 최저 갱신 지점        
                      )

hist = model.fit(x_trn, y_trn,
                 epochs = 1000,
                 batch_size = 32,
                 verbose = 2,
                 validation_split = 0.2,
                 callbacks = [ES, MCP])                    # 두 개 이상을 불러올수도 있다

# path = './_save/keras27_mcp/'
# model.save(path + 'keras27_3_save.h5')
# model.save_weights(path + 'keras26_5_save2.h5')

'''
[local]
loss : 23.538257598876953
rmse : 4.851624222760554
r2 스코어 : 0.7600070979022162

[SAVE]

[MCP]
loss : 22.839855194091797
rmse : 4.7791061082687625
r2 스코어 : 0.7671279018671824
'''

#4. 평가,예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
r2 = r2_score(y_tst, results)
rmse = np.sqrt(loss)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss)
print("rmse :", rmse)
print('r2 스코어 :', r2)