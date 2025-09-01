### 40_4.copy

### [실습]
#1. 시간 : vs CNN, CPU vs GPU
#2. 성능 : 기존 모델 능가

from tensorflow.keras.datasets import cifar100

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

##########################################################################
#1. 데이터
##########################################################################
(x_trn, y_trn), (x_tst,y_tst) = cifar100.load_data()

#####################################
### Scaling
x_trn = (x_trn-127.5)/(510.0)
x_tst = (x_tst-127.5)/(510.0)

#####################################
### reshape
x_trn = x_trn.reshape(x_trn.shape[0],32, 16, 6)
x_tst = x_tst.reshape(x_tst.shape[0],32, 16, 6)

from sklearn.decomposition import PCA
import time

pca = PCA(n_components=x_trn.shape[1])

x_trn = pca.fit_transform(x_trn)
x_tst = pca.fit_transform(x_tst)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)

a = len(np.where(evr_cumsum >= 1.)[0]) + 1
b = len(np.where(evr_cumsum >= 0.999)[0]) + 1
c = len(np.where(evr_cumsum >= 0.99)[0]) + 1
d = len(np.where(evr_cumsum >= 0.95)[0]) + 1

num = [a,b,c,d]
acc = []

for p in num :
    pca = PCA(n_components=p)
    pca.fit(x_trn)
    x_trn_P = pca.transform(x_trn)
    x_tst_P = pca.transform(x_tst)
    
    S = time.time()

    #2. 모델
    from catboost import CatBoostRegressor, CatBoostClassifier

    model = CatBoostClassifier(verbose=0)
    
    #3. 컴파일 훈련
    model.fit(x_trn_P, y_trn)
    
    score = model.score(x_tst_P, y_tst)
    
    acc.append((p, score))

for p, ACC in acc:
    print(f"n_components={p:>3} | acc={float(ACC):.4f}")

'''
[CNN-GPU]
loss : 1.812417984008789
acc  : 0.5221999883651733
acc  : 0.5222
시간 : 5364.959238290787

[DNN-CPU]
loss : 3.081491708755493
acc  : 0.2833000123500824
acc  : 0.2833
시간 : 577.3938059806824

[DNN-GPU]
loss : 3.1438710689544678
acc  : 0.28769999742507935
acc  : 0.2877
시간 : 701.1971230506897

[LSTM]
loss : 4.1187052726745605
acc  : 0.07620000094175339
acc  : 0.0762
시간 : 127.99137330055237

[Conv1D]
loss : 4.568159580230713
acc  : 0.04600000008940697
acc  : 0.046
시간 : 67.59857845306396
'''



##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc']
              )

#####################################
### ES
ES = EarlyStopping(monitor = 'val_acc', mode = 'max',
                   patience= 50, verbose=1,
                   restore_best_weights=True,
)

#####################################
### 파일명
import datetime

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path = './_save/keras40/cifar100/'
filename = '{epoch:04d}-{val_loss:.4f}.h5'
filepath = "".join([path,'k40_',date, '_', filename])

#####################################
### MCP
MCP = ModelCheckpoint(monitor = 'val_acc',
                      mode = 'max',
                      save_best_only= True,
                      verbose=1,
                      filepath = filepath,
                      )

S = time.time()

hist = model.fit(x_trn, y_trn,
                 epochs = 100,
                 batch_size = 5000,
                 verbose = 1,
                 validation_split = 0.2,
                 callbacks = [ES, MCP],
                 )   

E = time.time()

T = E - S 

##########################################################################
#4. 평가,예측
##########################################################################
loss = model.evaluate(x_tst, y_tst,
                      verbose= 1)
results = model.predict(x_tst)

#####################################
### 결과값 처리
results = np.argmax(results, axis=1)
y_tst = np.argmax(y_tst, axis=1)
ACC = accuracy_score(results, y_tst)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss[0])
print("acc  :", loss[1])
print("acc  :", ACC)
print("시간 :", T)

