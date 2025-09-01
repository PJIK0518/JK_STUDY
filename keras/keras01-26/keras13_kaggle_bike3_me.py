# keggle_bike2에서 제작한 new_test.csv로 count 예측

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#1. 데이터

path = 'c:/Study25/_data/kaggle/bike/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test_new_0.csv', index_col=0)
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
                                              test_size=0.8,
                                              random_state=42)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
epochs = 100

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_trn, y_trn, epochs=epochs, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
rmse = np.sqrt(loss)
R2 = r2_score(y_tst, results)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', loss)
print('rmse :', rmse)
print('R2   :', R2)

## Submission
'''
   Trial 0                      1
new rmse 97.46390056868492      95.56682258045937
    loss 0.31526525639854086    0.40910390425278215
sub rmse 0.8493069165179817     0.6882331580205135
    loss 0.9999780925618209     0.9999856142160045
   score 1.30488             
'''

## tuning
'''
'''

## sampleSubmission 제작
y_sub = model.predict(tst_csv)
sub_csv['count'] = y_sub
sub_csv.to_csv(path + 'samplesubmission_0522_1.csv', index=False)