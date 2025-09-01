### 41-3.copy

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

MS = MinMaxScaler()

MS.fit(x_trn)

x_trn = MS.transform(x_trn)
x_tst = MS.transform(x_tst)

from tensorflow.keras.layers import Conv2D, Flatten
### reshape

x_trn = x_trn.reshape(-1,5,2,1)
x_tst = x_tst.reshape(-1,5,2,1)
# model.add(Conv2D(10, 1, padding='same', input_shape = (13,1,1)))
# model.add(Conv2D(10, 1))
# model.add(Flatten())


#2. 모델구성
from tensorflow.python.keras.layers import Conv1D
model = Sequential()
model.add(Conv1D(10, 1, padding='same', input_shape = (5,2,1)))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Conv1D(10, 1))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(1))

epochs = 100000

''' loss
3963.572265625
[DO]
3465.424072265625
[CNN]
2477.224609375
[LSTM]
4012.264404296875
[Conv1D]
2314.77783203125
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')

# ES = EarlyStopping(monitor= 'val_loss',
#                    mode= 'min',
#                    patience= 500,
#                    restore_best_weights=True)

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
# import datetime

# path_MCP = './_save/keras28_mcp/03_diabetes/'

# date = datetime.datetime.now()
# # print(date)            
# # print(type(date))       
# date = date.strftime('%m%d_%H%M')              

# # print(date)             
# # print(type(date))

# filename = '{epoch:04d}-{val_loss:.4f}.h5'
# filepath = "".join([path_MCP,'keras28_',date, '_', filename])

# MCP = ModelCheckpoint(monitor='val_loss',
#                       mode='auto',
#                       save_best_only=True,
#                       filepath= filepath # 확장자의 경우 h5랑 같음
#                                          # patience 만큼 지나기전 최저 갱신 지점        
#                       )
import time
start = time.time()
hist = model.fit(x_trn, y_trn, epochs = 100, batch_size=32,
          verbose=2,
          validation_split=0.2,
        #   callbacks= [ES,MCP]
          )
end = time.time()
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

loss = model.evaluate(x_tst,y_tst)
print(loss)

# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)

# if gpus:
#     print('GPU 있다!!!')
# else:
#     print('GPU 없다...')

# time = end - start
# print("소요시간 :", time)

'''
GPU 있다!!!
소요시간 : 9.980796098709106

GPU 없다...
소요시간 : 5.148375988006592
'''

