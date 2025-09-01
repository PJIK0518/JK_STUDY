# 31-5.copy

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#1. 데이터

path = './Study25/_data/kaggle/bike/'

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

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.1))
model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.1))
model.add(Dense(1, activation = 'linear'))

''' loss
0.16112305223941803

738.5867309570312
'''

# epochs = 100000
#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

best_op = []
best_lr = 0
best_sc = -10000


model.compile(loss = 'mse', optimizer = Adam(learning_rate=0.01))


ES = EarlyStopping(monitor='val_loss',
                mode= 'min',
                patience= 50,
                restore_best_weights= True)

RLR = ReduceLROnPlateau(monitor = 'val_loss',
                        mode = 'auto',
                        patience = 10,
                        verbose = 1,
                        factor = 0.5)
                    # patience 만큼 갱신되지 않으면 해당 비율만큼 lr 하강(곱하기)


hist = model.fit(x_trn, y_trn, epochs = 10000, batch_size = 32,
        verbose=2,
        validation_split=0.2,
        callbacks = [ES, RLR])

loss = model.evaluate(x_tst, y_tst)
results = model.predict([x_tst])

R2 = r2_score(y_tst, results)

print('scr :',R2)
print('lss :',loss)

# scr : 0.9997813105583191
# lss : 7.218948841094971