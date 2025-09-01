### 20250531_practice_3_diabetes.copy

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(x)
# print(y)
# print(x.shape, y.shape) # (442, 10) (442,)

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              train_size = 0.85,
                                              shuffle = True,
                                              random_state=6974)

    ### scaling
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# MS = MinMaxScaler()
# MS.fit(x_trn)
# x_trn = MS.transform(x_trn)
# x_tst = MS.transform(x_tst)

# AS = MaxAbsScaler()
# AS.fit(x_trn)
# x_trn = AS.transform(x_trn)
# x_tst = AS.transform(x_tst)

# SS = StandardScaler()
# SS.fit(x_trn)
# x_trn = SS.transform(x_trn)
# x_tst = SS.transform(x_tst)

RS = RobustScaler()
RS.fit(x_trn)
x_trn = RS.transform(x_trn)
x_tst = RS.transform(x_tst)

'''
loss: 3773.294921875
R2  : 0.4334680488478657

[MS]
loss: 3956.329345703125
R2  : 0.40598680127748976

[AS]
loss: 3787.71533203125
R2  : 0.43130285999327844

[SS] **************
loss: 3619.743408203125
R2  : 0.45652263860005726

[RS]
loss: 3686.74462890625
R2  : 0.44646291644515224

'''

#2. 모델구성
model = Sequential()
model.add(Dense(400, input_dim = 10, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(1))

epochs = 100000

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')

ES = EarlyStopping(monitor= 'val_loss',
                   mode= 'min',
                   patience= 500,
                   restore_best_weights=True)

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
import datetime

path_MCP = './_save/keras28_mcp/03_diabetes/'

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

hist = model.fit(x_trn, y_trn, epochs = epochs, batch_size=32,
          verbose=2,
          validation_split=0.2,
          callbacks= [ES])

# path = './_save/practice/'

# model.save(path + '03_diabetes_20250531.h5')


# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows일 경우
# plt.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지

# plt.figure(figsize=(9,6)) # 9*6 사이즈 (단위 확인 피룡) # 크기설정이 1번
# plt.plot(hist.history['loss'], color='red', label='loss')
# plt.plot(hist.history['val_loss'], color='blue', label='val_loss') # y=epochs로 시간 따라서 자동으로
# plt.title('당뇨 loss') # 표 제목 한글 깨짐해결법 피룡
# plt.xlabel('epochs') # x 축 이름
# plt.ylabel('loss') # y 축 이름
# plt.legend(loc='upper right') # 우측상단 라벨표시 / defualt 는 빈자리
# plt.grid() # 격자표시

#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
R2 = r2_score(y_tst,results)

print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print("loss:", loss)
print("R2  :", R2)
# plt.show() # 이건 반드시 제일 아래

'''
#[갱신]
# R2 : 0.6522464497045597
'''

