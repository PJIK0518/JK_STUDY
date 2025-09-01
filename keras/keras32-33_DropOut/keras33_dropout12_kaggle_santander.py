# https://www.kaggle.com/competitions/santander-customer-transaction-prediction
# 30_12.copy

##########################################################################
#0. 준비
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

RS = 42
##########################################################################
#1 데이터
##########################################################################
path = './_data/kaggle/santander/'

trn_csv = pd.read_csv(path + 'train.csv', index_col=0)
tst_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')
""" print(trn_csv) [200000 rows x 201 columns] 
              target    var_0   var_1    var_2   var_3    var_4    var_5   var_6    var_7   var_8   var_9  ...  var_189  var_190  var_191  var_192  var_193  var_194  var_195  var_196  var_197  var_198  var_199
ID_code                                                                                                    ...
train_0            0   8.9255 -6.7863  11.9081  5.0930  11.4607  -9.2834  5.1187  18.6266 -4.9200  5.7470  ...   0.5857   4.4354   3.9642   3.1364   1.6910  18.5227  -2.3978   7.8784   8.5635  12.7803  -1.0914  
train_1            0  11.5006 -4.1473  13.8588  5.3890  12.3622   7.0433  5.6208  16.5338  3.1468  8.0851  ...  -0.3566   7.6421   7.7214   2.5837  10.9516  15.4305   2.0339   8.1267   8.7889  18.3560   1.9518  
train_2            0   8.6093 -2.7457  12.0805  7.8928  10.5825  -9.0837  6.9427  14.6155 -4.9193  5.9525  ...  -0.8417   2.9057   9.7905   1.6704   1.6858  21.6042   3.1417  -6.5213   8.2675  14.7222   0.3965  
train_3            0  11.0604 -2.1518   8.9522  7.1957  12.5846  -1.8361  5.8428  14.9250 -5.8609  8.2450  ...   1.8489   4.4666   4.7433   0.7178   1.4214  23.0347  -1.2706  -2.9275  10.2922  17.9697  -8.9996  
train_4            0   9.8369 -1.4834  12.8746  6.6375  12.2772   2.4486  5.9405  19.2514  6.2654  7.6784  ...  -0.2829  -1.4905   9.5214  -0.1508   9.1942  13.2876  -1.5121   3.9267   9.5031  17.9974  -8.8104  
...              ...      ...     ...      ...     ...      ...      ...     ...      ...     ...     ...  ...      ...      ...      ...      ...      ...      ...      ...      ...      ...      ...      ...  
train_199995       0  11.4880 -0.4956   8.2622  3.5142  10.3404  11.6081  5.6709  15.1516 -0.6209  5.6669  ...  -0.4594   6.1415  13.2305   3.9901   0.9388  18.0249  -1.7939   2.1661   8.5326  16.6660 -17.8661  
train_199996       0   4.9149 -2.4484  16.7052  6.6345   8.3096 -10.5628  5.8802  21.5940 -3.6797  6.0019  ...  -0.1162   4.9611   4.6549   0.6998   1.8341  22.2717   1.7337  -2.1651   6.7419  15.9054   0.3388  
train_199997       0  11.2232 -5.0518  10.5127  5.6456   9.3410  -5.4086  4.5555  21.5571  0.1202  6.1629  ...   2.3425   4.0651   5.4414   3.1032   4.8793  23.5311  -1.5736   1.2832   8.7155  13.8329   4.1995  
train_199998       0   9.7148 -8.6098  13.6104  5.7930  12.5173   0.5339  6.0479  17.0152 -2.1926  8.7542  ...   0.3243   2.6840   8.6587   2.7337  11.1178  20.4158  -0.0786   6.7980  10.0342  15.5289 -13.9001  
train_199999       0  10.8762 -5.7105  12.1183  8.0328  11.5577   0.3488  5.2839  15.2058 -0.4541  9.3688  ...   1.7535   8.9842   1.6893   0.1276   0.3766  15.2101  -2.4907  -2.2342   8.1857  12.1284   0.1385  
"""
""" print(tst_csv) [200000 rows x 200 columns]
               var_0    var_1    var_2   var_3    var_4    var_5   var_6    var_7   var_8   var_9  var_10  ...  var_189  var_190  var_191  var_192  var_193  var_194  var_195  var_196  var_197  var_198  var_199
ID_code                                                                                                    ...
test_0       11.0656   7.7798  12.9536  9.4292  11.4327  -2.3805  5.8493  18.2675  2.1337  8.8100 -2.0248  ...   1.6591  -2.1556  11.8495  -1.4300   2.4508  13.7112   2.4669   4.3654  10.7200  15.4722  -8.7197  
test_1        8.5304   1.2543  11.3047  5.1858   9.1974  -4.0117  6.0196  18.6316 -4.4131  5.9739 -1.3809  ...   0.9812  10.6165   8.8349   0.9403  10.1282  15.5765   0.4773  -1.4852   9.8714  19.1293 -20.9760  
test_2        5.4827 -10.3581  10.1407  7.0479  10.2628   9.8052  4.8950  20.2537  1.5233  8.3442 -4.7057  ...   1.1821  -0.7484  10.9935   1.9803   2.1800  12.9813   2.1281  -7.1086   7.0618  19.8956 -23.1794  
test_3        8.5374  -1.3222  12.0220  6.5749   8.8458   3.1744  4.9397  20.5660  3.3755  7.4578  0.0095  ...   1.3104   9.5702   9.0766   1.6580   3.5813  15.1874   3.1656   3.9567   9.2295  13.0168  -4.2108  
test_4       11.7058  -0.1327  14.1295  7.7506   9.1035  -8.5848  6.8595  10.6048  2.9890  7.1437  5.1025  ...   1.6321   4.2259   9.1723   1.2835   3.3778  19.5542  -0.2860  -5.1612   7.2882  13.9260  -9.1846  
...              ...      ...      ...     ...      ...      ...     ...      ...     ...     ...     ...  ...      ...      ...      ...      ...      ...      ...      ...      ...      ...      ...      ...  
test_199995  13.1678   1.0136  10.4333  6.7997   8.5974  -4.1641  4.8579  14.7625 -2.7239  6.9937  2.6802  ...   1.0903   2.0544   9.6849   4.6734  -1.3660  12.8721   1.2013  -4.6195   9.1568  18.2102   4.8801  
test_199996   9.7171  -9.1462   7.3443  9.1421  12.8936   3.0191  5.6888  18.8862  5.0915  6.3545  3.2618  ...   1.7106   5.0071   6.6548   1.8197   2.4104  18.9037  -0.9337   2.9995   9.1112  18.1740 -20.7689  
test_199997  11.6360   2.2769  11.2074  7.7649  12.6796  11.3224  5.3883  18.3794  1.6603  5.7341  9.8596  ...  -1.0926   5.1536   2.6498   2.4937  -0.0637  20.0609  -1.1742  -4.1524   9.1933  11.7905 -22.2762  
test_199998  13.5745  -0.5134  13.6584  7.4855  11.2241 -11.3037  4.1959  16.8280  5.3208  8.9032  5.5000  ...   0.8885   3.4259   8.5012   2.2713   5.7621  17.0056   1.1763  -2.3761   8.1079   8.7735  -0.2122  
test_199999  10.4664   1.8070  10.2277  6.0654  10.0258   1.0789  4.8879  14.4892 -0.5902  7.8362  8.4796  ...   0.6155   0.1398   9.2828   1.3601   4.8985  20.0926  -1.3048  -2.5981  10.3378  14.3340  -7.7094  
"""

x = trn_csv.drop(['target'], axis=1)
y = trn_csv['target']

""" print(x.shape) (200000, 200) """
""" print(y.shape) (200000,) """
""" print(np.unique(y, return_counts=True)) (array([0, 1], dtype=int64), array([179902,  20098], dtype=int64)) """

x_trn, x_tst, y_trn, y_tst = train_test_split(x, y,
                                                train_size = 0.5,
                                                shuffle = True,
                                                random_state = RS,
                                                # stratify=y,
                                                )

x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn,
                                                train_size = 0.65,
                                                shuffle = True,
                                                random_state = RS,
                                                stratify=y_trn,
                                                )

#####################################
## Scaler
def Scaler(SC, a, b, c, d):
    SC.fit(a)
    return SC.transform(a), SC.transform(b), SC.transform(c), SC.transform(d)

x_trn, x_tst, x_val, tst_csv = Scaler(RobustScaler(), x_trn, x_tst, x_val, tst_csv)

def Scaler(SC, a, b, c, d):
    SC.fit(a)
    return SC.transform(a), SC.transform(b), SC.transform(c), SC.transform(d)

x_trn, x_tst, x_val, tst_csv = Scaler(StandardScaler(), x_trn, x_tst, x_val, tst_csv)

#####################################
## 증폭 : class_weight
classes = np.unique(y_trn)
CW = compute_class_weight(class_weight='balanced', classes=classes, y=y_trn)

##########################################################################
#2 모델구성
##########################################################################
## 불러오기
# path_L = 'C:/Study25/_data/kaggle/santander/MCP/'
# M = load_model(path_L + 'MCP_0608_2.h5')
#####################################
def LT(a,b,c,d,e,DO):
        model = Sequential()
        model.add(Dense(a, input_dim=200 , activation='relu'))
        model.add(Dropout(DO))
        
        model.add(BatchNormalization())
        model.add(Dense(b, activation='relu'))
        model.add(Dropout(DO))
        
        model.add(BatchNormalization())
        model.add(Dense(c, activation='relu'))
        model.add(Dropout(DO))

        model.add(BatchNormalization())
        model.add(Dense(d, activation='relu'))
        model.add(Dropout(DO))

        model.add(BatchNormalization())
        model.add(Dense(e, activation='relu'))
        model.add(Dropout(DO))
        
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        return model
    
M = LT(240,240,240,240,240, 0.1)
E, B, P = (100000, 100, 100)

''' loss
loss : 0.23793835937976837
loss : 0.23825055360794067
'''

#####################################
## 저장 정보
date = datetime.datetime.now()
date = date.strftime('%m%d')

saveNum = f'{date}_0_0'
fit_info = '{epochs:04d}_{val_acc:.4f}'

#####################################
## Callbacks
ES = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=P,
    restore_best_weights=True,
)

path_M = './_data/kaggle/santander/MCP/'

MCP = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath="".join([path_M, 'MCP_', saveNum, '.h5'])
)

##########################################################################
#3 컴파일, 훈련
##########################################################################
M.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

S = time.time()
H = M.fit(x_trn, y_trn,
          epochs=E, batch_size=B,
          verbose=2,
          validation_data = (x_val, y_val),
          callbacks = [ES, MCP])
E = time.time()
T = E - S

#####################################
## 모델 및 가중치 저장
path_S = './_data/kaggle/santander/save/'
path_W = './_data/kaggle/santander/weight/'

M.save(path_S + f'save_{saveNum}.h5')
M.save_weights(path_W + f'weight_{saveNum}.h5')

##########################################################################
#4. 평가 예측
L = M.evaluate(x_tst, y_tst)
R = M.predict(x_tst)
R = np.round(R)
F1 = f1_score(y_tst,R)

#####################################
## 그래프
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.title('SANTANDER')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(H.history['loss'], color = 'red', label = 'loss')
plt.plot(H.history['val_loss'], color = 'green', label = 'val_loss')
plt.legend(loc = 'upper right')
plt.grid()

print('loss :', L[0])
print('ACC  :', L[1])
print('F1sc :', F1)
print('time :', T)
plt.show()

#####################################
## 파일송출

y_sub = M.predict(tst_csv)
y_sub = np.round(y_sub)

path_Sb = './_data/kaggle/santander/sub/'
sub_csv['target'] = y_sub
sub_csv.to_csv(path_Sb + f'sample_submission_{saveNum}.csv', index=False)