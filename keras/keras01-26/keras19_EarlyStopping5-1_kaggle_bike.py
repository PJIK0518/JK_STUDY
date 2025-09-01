## https://www.kaggle.com/competitions/bike-sharing-demand/overview
## 지금은 시간은 data XX / 나중에 정제하면 연 월 일 시 등으로 사용가능
 # 1. train_csv에서 casual과 registered를 y로 잡는다
 # 2. 훈련해서, test_csv의 casual과 registered를 예측
 # 3. 예측한 casual과 registered를 test_csv에 컬럼으로 넣는다
 #    (n, 8) > (n, 10) test.csv 파일로 new_test.csv 파일 제작
 # Stacking 기법!!

#0. 준비
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#1. 데이터
path = 'c:/Study25/_data/kaggle/bike/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
'''print(trn_csv) : [10886 rows x 11 columns], season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered  count
                     season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered  count
datetime
2011-01-01 00:00:00       1        0           0        1   9.84  14.395        81     0.0000       3          13     16
2011-01-01 01:00:00       1        0           0        1   9.02  13.635        80     0.0000       8          32     40
2011-01-01 02:00:00       1        0           0        1   9.02  13.635        80     0.0000       5          27     32
2011-01-01 03:00:00       1        0           0        1   9.84  14.395        75     0.0000       3          10     13
2011-01-01 04:00:00       1        0           0        1   9.84  14.395        75     0.0000       0           1      1
...                     ...      ...         ...      ...    ...     ...       ...        ...     ...         ...    ...
2012-12-19 19:00:00       4        0           1        1  15.58  19.695        50    26.0027       7         329    336
2012-12-19 20:00:00       4        0           1        1  14.76  17.425        57    15.0013      10         231    241
2012-12-19 21:00:00       4        0           1        1  13.94  15.910        61    15.0013       4         164    168
2012-12-19 22:00:00       4        0           1        1  13.94  17.425        61     6.0032      12         117    129
2012-12-19 23:00:00       4        0           1        1  13.12  16.665        66     8.9981       4          84     88
'''

tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
'''print(tst_csv) : [6493 rows x 8 columns],   season  holiday  workingday  weather   temp   atemp  humidity  windspeed
                     season  holiday  workingday  weather   temp   atemp  humidity  windspeed
datetime
2011-01-20 00:00:00       1        0           1        1  10.66  11.365        56    26.0027
2011-01-20 01:00:00       1        0           1        1  10.66  13.635        56     0.0000
2011-01-20 02:00:00       1        0           1        1  10.66  13.635        56     0.0000
2011-01-20 03:00:00       1        0           1        1  10.66  12.880        56    11.0014
2011-01-20 04:00:00       1        0           1        1  10.66  12.880        56    11.0014
...                     ...      ...         ...      ...    ...     ...       ...        ...
2012-12-31 19:00:00       1        0           1        2  10.66  12.880        60    11.0014
2012-12-31 20:00:00       1        0           1        2  10.66  12.880        60    11.0014
2012-12-31 21:00:00       1        0           1        1  10.66  12.880        60    11.0014
2012-12-31 22:00:00       1        0           1        1  10.66  13.635        56     8.9981
2012-12-31 23:00:00       1        0           1        1  10.66  13.635        65     8.9981
'''

sub_csv = pd.read_csv(path + 'sampleSubmission.csv')
'''print(sub_csv) : [6493 rows x 2 columns],   datetime  count
                 datetime  count
0     2011-01-20 00:00:00      0
1     2011-01-20 01:00:00      0
2     2011-01-20 02:00:00      0
3     2011-01-20 03:00:00      0
4     2011-01-20 04:00:00      0
...                   ...    ...
6488  2012-12-31 19:00:00      0
6489  2012-12-31 20:00:00      0
6490  2012-12-31 21:00:00      0
6491  2012-12-31 22:00:00      0
6492  2012-12-31 23:00:00      0
'''

x = trn_csv.drop(['casual', 'registered', 'count'], axis = 1)
y = trn_csv[['casual', 'registered']]
'''print(x) : [10886 rows x 8 columns]
                     season  holiday  workingday  weather   temp   atemp  humidity  windspeed
datetime
2011-01-01 00:00:00       1        0           0        1   9.84  14.395        81     0.0000
2011-01-01 01:00:00       1        0           0        1   9.02  13.635        80     0.0000
2011-01-01 02:00:00       1        0           0        1   9.02  13.635        80     0.0000
2011-01-01 03:00:00       1        0           0        1   9.84  14.395        75     0.0000
2011-01-01 04:00:00       1        0           0        1   9.84  14.395        75     0.0000
...                     ...      ...         ...      ...    ...     ...       ...        ...
2012-12-19 19:00:00       4        0           1        1  15.58  19.695        50    26.0027
2012-12-19 20:00:00       4        0           1        1  14.76  17.425        57    15.0013
2012-12-19 21:00:00       4        0           1        1  13.94  15.910        61    15.0013
2012-12-19 22:00:00       4        0           1        1  13.94  17.425        61     6.0032
2012-12-19 23:00:00       4        0           1        1  13.12  16.665        66     8.9981
'''
'''print(y) : [10886 rows x 2 columns]
                     casual  registered
datetime
2011-01-01 00:00:00       3          13
2011-01-01 01:00:00       8          32
2011-01-01 02:00:00       5          27
2011-01-01 03:00:00       3          10
2011-01-01 04:00:00       0           1
...                     ...         ...
2012-12-19 19:00:00       7         329
2012-12-19 20:00:00      10         231
2012-12-19 21:00:00       4         164
2012-12-19 22:00:00      12         117
2012-12-19 23:00:00       4          84
'''

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size=0.2,
                                              shuffle= True,
                                              random_state=42)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=8, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(20, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(5, activation= 'relu'))
model.add(Dense(5, activation= 'relu'))
model.add(Dense(2, activation= 'linear'))

epochs = 100000

ES = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience= 250,
                   restore_best_weights= True)

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
H = model.fit(x_trn, y_trn,
              epochs=epochs, batch_size=16,
              verbose = 2,
              validation_split = 0.2,
              callbacks = [ES])

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows일 경우

plt.figure(figsize=(9,6)) # 9*6 사이즈 (단위 확인 피룡) # 크기설정이 1번
plt.plot(H.history['loss'], color='red', label='loss')
plt.plot(H.history['val_loss'], color='blue', label='val_loss') # y=epochs로 시간 따라서 자동으로
plt.title('kaggle_new loss') # 표 제목 한글 깨짐해결법 피룡
plt.xlabel('epochs') # x 축 이름
plt.ylabel('loss') # y 축 이름
plt.legend(loc='upper right') # 우측상단 라벨표시 / defualt 는 빈자리
plt.grid() # 격자표시

#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
rmse = np.sqrt(loss)
R2 = r2_score(y_tst, results)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', loss)
print('rmse :', rmse)
print('R2   :', R2)
plt.show()

## new_tst_csv 제작
y_tst_n = model.predict(tst_csv)
tst_csv[['casual', 'registered']] = y_tst_n
tst_csv.to_csv(path + 'test_new_0527_2.csv')

## Submission
'''
   Trial 0                      1                       2
new rmse 97.46390056868492      95.56682258045937       94.35544602889915
    R2   0.31526525639854086    0.40910390425278215     0.4261365447224514

   Trial 3
new rmse 94.4735257641658
    R2   0.4287868664462265
'''

### tuning
''' rmse / R2  94.4735257641658 / 0.4287868664462265
train_size : 0.8 / random_state : 42
hidden_layer : 16 32 16 16
epochs : 100000 / batch_size : 32
validation_split : 0.2 / Patience : 50
rmse : 94.35544602889915
R2   : 0.4261365447224514

train_size : 0.8 / random_state : 42
hidden_layer : 16 32 16 16
epochs : 100000 / batch_size : 32
validation_split : 0.2 / Patience : 100
rmse : 95.26358375908919
R2   : 0.40374546678672196 // overfit

train_size : 0.8 / random_state : 42
hidden_layer : 16 16 16 16
epochs : 100000 / batch_size : 32
validation_split : 0.2 / Patience : 100
rmse : 94.4735257641658     96.37130140438595
R2   : 0.4287868664462265   0.3877403717092927
'''
'''
train_size : 0.8 / random_state : 42
hidden_layer : 8 32 16 8
epochs : 100000 / batch_size : 32
validation_split : 0.2 / Patience : 100
rmse : 94.96546967023066
R2   : 0.4158844079980082

train_size : 0.8 / random_state : 42
hidden_layer : 8 32 16 8
epochs : 100000 / batch_size : 32
validation_split : 0.2 / Patience : 150
rmse : 95.30511754196361
R2   : 0.41262610852945886

train_size : 0.8 / random_state : 42
hidden_layer : 8 32 16 8
epochs : 100000 / batch_size : 16
validation_split : 0.2 / Patience : 150
rmse : 95.4379604605788
R2   : 0.40760883425271327

train_size : 0.8 / random_state : 42
hidden_layer : 8 32 16 8
epochs : 100000 / batch_size : 16
validation_split : 0.2 / Patience : 200
rmse : 94.89479166094944
R2   : 0.41014159600935746

train_size : 0.8 / random_state : 42
hidden_layer : 10 20 20 10
epochs : 100000 / batch_size : 16
validation_split : 0.2 / Patience : 200
rmse : 95.31625504898155
R2   : 0.39874994866401

train_size : 0.8 / random_state : 42
hidden_layer : 5 10 10 10 5 5
epochs : 100000 / batch_size : 16
validation_split : 0.2 / Patience : 200
rmse : 95.45377337519193
R2   : 0.4001096192315698

train_size : 0.8 / random_state : 42
hidden_layer : 5 30 20 10 5 5
epochs : 100000 / batch_size : 16
validation_split : 0.2 / Patience : 250
rmse : 94.84926363709947
R2   : 0.42111680924836603

train_size : 0.8 / random_state : 42
hidden_layer : 5 20 20 10 5 5
epochs : 100000 / batch_size : 16
validation_split : 0.2 / Patience : 250
'''