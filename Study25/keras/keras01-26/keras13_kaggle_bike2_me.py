## https://www.kaggle.com/competitions/bike-sharing-demand/overview
## 지금은 시간은 data XX / 나중에 정제하면 연 월 일 시 등으로 사용가능
 # 1. train_csv에서 casual과 registered를 y로 잡는다
 # 2. 훈련해서, test_csv의 casual과 registered를 예측
 # 3. 예측한 casual과 registered를 test_csv에 컬럼으로 넣는다
 #    (n, 8) > (n, 10) test.csv 파일로 new_test.csv 파일 제작
 # Stacking 기법!!

#0. 준비
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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
                                              random_state=42)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=8, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(2, activation= 'linear'))
epochs = 100

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_trn, y_trn, epochs=epochs, batch_size=16)

#4. 평가, 예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
rmse = np.sqrt(loss)
R2 = r2_score(y_tst, results)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print('loss :', loss)
print('rmse :', rmse)
print('R2   :', R2)

## tuning
''' Trial 0
train_size : 0.80 / epochs : 10 / batch_size : 32 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 9499.2119140625
rmse : 97.46390056868492
R2   : 0.31526525639854086
'''
''' train_size : 0.80 / epochs : 100 / batch_size : 16 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
train_size : 0.80 / epochs : 100 / batch_size : 32 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 9062.0947265625      9319.6103515625     9055.87109375       9107.5517578125     9115.4375
rmse : 95.19503519912422    96.53812900384231   95.16234073282351   95.43349389922021   95.47480034019448
R2   : 0.41708566662789065  0.3985893997680928  0.4174771623589619  0.4051878212490844  0.41379703300630655

train_size : 0.80 / epochs : 500 / batch_size : 32 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 10328.0556640625
rmse : 101.62704199209234
R2   : 0.35444061006718747

train_size : 0.80 / epochs : 300 / batch_size : 32 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 9794.9560546875      10306.6416015625
rmse : 98.96947031629249    101.52163120026441
R2   : 0.38800571051831917  0.3668042496236475

train_size : 0.80 / epochs : 200 / batch_size : 32 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 9389.984375          9393.8408203125     9327.5361328125
rmse : 96.90193174029092    96.92182839955352   96.57917028434495
R2   : 0.4094447746366211   0.40336173538035963 0.4164756327859969

train_size : 0.80 / epochs : 200 / batch_size : 32 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 9137.1875
rmse : 95.5886368769845
R2   : 0.4060790811203291

train_size : 0.80 / epochs : 50 / batch_size : 32 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 9116.96875           9655.513671875      9204.408203125
rmse : 95.48281913517216    98.2624733653443    95.93960706155201
R2   : 0.40370292734904384  0.37749032188432297 0.393035527656605

train_size : 0.80 / epochs : 100 / batch_size : 32 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 9132.6318359375
rmse : 95.56480437869111
R2   : 0.4080494951334057

train_size : 0.80 / epochs : 100 / batch_size : 16 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 9123.212890625       9092.396484375      9152.83984375       9071.037109375      9117.794921875
rmse : 95.51551125668018    95.35405856267997   95.67047529802494   95.24199236353154   95.48714532268205
R2   : 0.4196303320668722   0.40748379804476254 0.4092207454798974  0.4250911538767714  0.41468264571719016

train_size : 0.80 / epochs : 100 / batch_size : 8 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 9218.3603515625      9426.75             9122.6513671875     9119.513671875      9136.974609375
rmse : 96.01229271068627    97.09145173494936   95.51257177559141   95.49614480111227   95.58752329344557
R2   : 0.39981607747201625  0.3897594736189433  0.4160210985431707  0.4215157947565554  0.41259985295294527

train_size : 0.80 / epochs : 100 / batch_size : 5 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 9180.0966796875      9127.84375          9142.3154296875     9232.271484375      9370.3955078125
rmse : 95.81282106110591    95.539749580999     95.61545601882312   96.08470994062999   96.80080323950055
R2   : 0.4153588874256018   0.41262769762906193 0.3991718163876697  0.39976573854652836 0.39888125844911415
'''
'''
train_size : 0.80 / epochs : 100 / batch_size : 16 / hidden_layer : 64 64 64 64 64 64 / random_state : 42 / Activation : relu
loss : 9123.212890625       9092.396484375      9152.83984375       9071.037109375      9117.794921875
rmse : 95.51551125668018    95.35405856267997   95.67047529802494   95.24199236353154   95.48714532268205
R2   : 0.4196303320668722   0.40748379804476254 0.4092207454798974  0.4250911538767714  0.41468264571719016

'''
## new_tst_csv 제작
y_tst_n = model.predict(tst_csv)
tst_csv[['casual', 'registered']] = y_tst_n
tst_csv.to_csv(path + 'test_new_1.csv')