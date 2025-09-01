## https://www.kaggle.com/competitions/bike-sharing-demand/overview
## 지금은 시간은 data XX / 나중에 정제하면 연 월 일 시 등으로 사용가능
 # 1. train_csv에서 casual과 registered를 y로 잡는다
 # 2. 훈련해서, test_csv의 casual과 registered를 예측
 # 3. 예측한 casual과 registered를 test_csv에 컬럼으로 넣는다
 #    (n, 8) > (n, 10) test.csv 파일로 new_test.csv 파일 제작
 # Stacking 기법!!

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/kaggle/bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
samplesubmission_csv = pd.read_csv(path + 'samplesubmission.csv')

'''print(train_csv)
                   datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered  count
2011-01-01 00:00:00       1        0           0        1   9.84  14.395        81     0.0000       3          13     16
2011-01-01 01:00:00       1        0           0        1   9.02  13.635        80     0.0000       8          32     40
2011-01-01 02:00:00       1        0           0        1   9.02  13.635        80     0.0000       5          27     32
2011-01-01 03:00:00       1        0           0        1   9.84  14.395        75     0.0000       3          10     13
2011-01-01 04:00:00       1        0           0        1   9.84  14.395        75     0.0000       0           1      1
                ...     ...      ...         ...      ...    ...     ...       ...        ...     ...         ...    ...
2012-12-19 19:00:00       4        0           1        1  15.58  19.695        50    26.0027       7         329    336
2012-12-19 20:00:00       4        0           1        1  14.76  17.425        57    15.0013      10         231    241
2012-12-19 21:00:00       4        0           1        1  13.94  15.910        61    15.0013       4         164    168
2012-12-19 22:00:00       4        0           1        1  13.94  17.425        61     6.0032      12         117    129
2012-12-19 23:00:00       4        0           1        1  13.12  16.665        66     8.9981       4          84     88
'''
## print(train_csv.shape) (10886, 11)
## print(train_csv.columns) ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
'''print(train_csv.info())
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   season      10886 non-null  int64
 1   holiday     10886 non-null  int64
 2   workingday  10886 non-null  int64
 3   weather     10886 non-null  int64
 4   temp        10886 non-null  float64
 5   atemp       10886 non-null  float64
 6   humidity    10886 non-null  int64
 7   windspeed   10886 non-null  float64
 8   casual      10886 non-null  int64
 9   registered  10886 non-null  int64
 10  count       10886 non-null  int64
'''
## print(train_csv.isnull().sum())  없음
## print(train_csv.isna().sum())    없음
'''print(train_csv.describe())
             season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000
season : 봄 여름 가을 겨울 > 1, 2, 3, 4
holiday : 평일 휴일 > 0, 1 등등으로 분석...! > 나중에 이상한 데이터도 잡아내야함
'''

'''print(test_csv)
           datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed
2011-01-20 00:00:00       1        0           1        1  10.66  11.365        56    26.0027
2011-01-20 01:00:00       1        0           1        1  10.66  13.635        56     0.0000
2011-01-20 02:00:00       1        0           1        1  10.66  13.635        56     0.0000
2011-01-20 03:00:00       1        0           1        1  10.66  12.880        56    11.0014
2011-01-20 04:00:00       1        0           1        1  10.66  12.880        56    11.0014
                ...     ...      ...         ...      ...    ...     ...       ...        ...
2012-12-31 19:00:00       1        0           1        2  10.66  12.880        60    11.0014
2012-12-31 20:00:00       1        0           1        2  10.66  12.880        60    11.0014
2012-12-31 21:00:00       1        0           1        1  10.66  12.880        60    11.0014
2012-12-31 22:00:00       1        0           1        1  10.66  13.635        56     8.9981
2012-12-31 23:00:00       1        0           1        1  10.66  13.635        65     8.9981
'''
## print(test_csv.columns)  ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
## print(test_csv.shape)  (6493, 8)
'''print(test_csv.info()) =isna(), =isnull()
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   datetime    6493 non-null   object
 1   season      6493 non-null   int64
 2   holiday     6493 non-null   int64
 3   workingday  6493 non-null   int64
 4   weather     6493 non-null   int64
 5   temp        6493 non-null   float64
 6   atemp       6493 non-null   float64
 7   humidity    6493 non-null   int64
 8   windspeed   6493 non-null   float64
'''

'''print(samplesubmission_scv)
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
## print(samplesubmission_scv.shape)    (6493, 1)
## print(samplesubmission_scv.columns)  ['count']

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
'''print(x)
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
## print(x.shape) (10886, 8)

y = train_csv[['casual', 'registered']]
## print(y.shape) (10886, 2)

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                              test_size=0.2,
                                              random_state=3081)

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=8, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dense(2, activation= 'linear'))

epochs = 10

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_trn, y_trn, epochs = epochs, batch_size=32)

#4. 평가, 예측
results = model.predict(x_tst)
def RMSE(a,b):
    return np.sqrt(mean_squared_error(a,b))
rmse = RMSE(y_tst, results)
R2 = r2_score(y_tst, results)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('rmse :', rmse)
print('R2   :', R2)

## 파일 제출
y_tst_n = model.predict(test_csv)
print('test_csv 타입 :', type(test_csv)) # <class 'pandas.core.frame.DataFrame'>
print('y_tst_n 타입 :', type(y_tst_n)) # <class 'numpy.ndarray'> >> pandas를 통해 정리한 데이터는 index와 column명을 제외하고 인식
                                                               # >> test_csv : 헤더와 숫자를 모두 인식해서 data frame
                                                               # >> y_tst_n : 숫자들만 인식해서 numpy.ndarray
                                                               # 
exit()

tst_2_csv = test_csv     # .copy 를 뒤에 붙여줘야함 // 데이터 안전을 위해서
tst_2_csv[['casual', 'registered']] = y_tst_n
tst_2_csv.to_csv(path + 'test_new_T1.csv')