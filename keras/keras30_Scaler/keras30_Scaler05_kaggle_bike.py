## keras28_MCP_save_05_kaggle_bike.copy

# keggle_bike2에서 제작한 new_test.csv로 count 예측

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#1. 데이터

path = 'c:/Study25/_data/kaggle/bike/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test_new_0527_1.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = trn_csv.drop(['count'], axis = 1)
y = trn_csv['count']

'''print(x) [10886 rows x 10 columns]    season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered
                     season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered
datetime
2011-01-01 00:00:00       1        0           0        1   9.84  14.395        81     0.0000       3          13
2011-01-01 01:00:00       1        0           0        1   9.02  13.635        80     0.0000       8          32
2011-01-01 02:00:00       1        0           0        1   9.02  13.635        80     0.0000       5          27
2011-01-01 03:00:00       1        0           0        1   9.84  14.395        75     0.0000       3          10
2011-01-01 04:00:00       1        0           0        1   9.84  14.395        75     0.0000       0           1
...                     ...      ...         ...      ...    ...     ...       ...        ...     ...         ...
2012-12-19 19:00:00       4        0           1        1  15.58  19.695        50    26.0027       7         329
2012-12-19 20:00:00       4        0           1        1  14.76  17.425        57    15.0013      10         231
2012-12-19 21:00:00       4        0           1        1  13.94  15.910        61    15.0013       4         164
2012-12-19 22:00:00       4        0           1        1  13.94  17.425        61     6.0032      12         117
2012-12-19 23:00:00       4        0           1        1  13.12  16.665        66     8.9981       4          84
'''
'''print(y) [10886 rows x 1 columns]     count
datetime
2011-01-01 00:00:00     16
2011-01-01 01:00:00     40
2011-01-01 02:00:00     32
2011-01-01 03:00:00     13
2011-01-01 04:00:00      1
                      ...
2012-12-19 19:00:00    336
2012-12-19 20:00:00    241
2012-12-19 21:00:00    168
2012-12-19 22:00:00    129
2012-12-19 23:00:00     88
'''

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size=0.2,
                                              random_state=42)

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
loss : 0.0005235401331447065
rmse : 0.022880999391300776
R2   : 1.0

[MS]
loss : 0.00547562912106514
rmse : 0.0739974940188189
R2   : 0.9999998211860657

[AS]
loss : 0.004681102465838194
rmse : 0.0684185827523356
R2   : 0.9999998807907104

[SS]
loss : 0.005199537146836519
rmse : 0.0721078161286037
R2   : 0.9999998211860657

[RS]
loss : 0.0007426317315548658
rmse : 0.027251270274151734
R2   : 1.0
'''


#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

epochs = 100000

ES = EarlyStopping(monitor= 'val_loss',
                   mode= 'min',
                   patience=100,
                   restore_best_weights=True)
################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
import datetime

path_MCP = './_save/keras28_mcp/05_kaggle_bike/'

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

# #3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
H = model.fit(x_trn, y_trn,
          epochs=epochs, batch_size=32,
          verbose = 2,
          validation_split= 0.2,
          callbacks = [ES])

# # plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows일 경우

# # plt.figure(figsize=(9,6)) # 9*6 사이즈 (단위 확인 피룡) # 크기설정이 1번
# # plt.plot(H.history['loss'], color='red', label='loss')
# # plt.plot(H.history['val_loss'], color='blue', label='val_loss') # y=epochs로 시간 따라서 자동으로
# # plt.title('kaggle_sub loss') # 표 제목 한글 깨짐해결법 피룡
# # plt.xlabel('epochs') # x 축 이름
# # plt.ylabel('loss') # y 축 이름
# # plt.legend(loc='upper right') # 우측상단 라벨표시 / defualt 는 빈자리
# # plt.grid() # 격자표시

#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
rmse = np.sqrt(loss)
R2 = r2_score(y_tst, results)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', loss)
print('rmse :', rmse)
print('R2   :', R2)
# plt.show()

## sampleSubmission 제작
# y_sub = model.predict(tst_csv)
# sub_csv['count'] = y_sub
# sub_csv.to_csv(path + 'samplesubmission_0527_2.csv', index=False)