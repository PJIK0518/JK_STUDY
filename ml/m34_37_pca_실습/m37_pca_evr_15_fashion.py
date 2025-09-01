### 40_2.copy

### [실습]
#1. 시간 : vs CNN, CPU vs GPU
#2. 성능 : 기존 모델 능가

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import fashion_mnist

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import time

##########################################################################
#1. 데이터
##########################################################################
(x_trn, y_trn), (x_tst, y_tst) = fashion_mnist.load_data()

# print(x_trn.shape) # (60000, 28, 28)
# print(x_tst.shape) # (10000, 28, 28)
# print(y_trn.shape) # (60000,)
# print(y_tst.shape) # (10000,)

#####################################
### x reshape 
x_trn = x_trn.reshape(x_trn.shape[0], x_trn.shape[1]*x_trn.shape[2])
x_tst = x_tst.reshape(x_tst.shape[0], x_tst.shape[1]*x_tst.shape[2])

from sklearn.decomposition import PCA
import time

pca = PCA(n_components=x_trn.shape[1])

#####################################
### Scaling
x_trn = (x_trn-127.5)/(510.0)
x_tst = (x_tst-127.5)/(510.0)

x_trn = np.array(x_trn).reshape(-1,14*14*4)
x_tst = np.array(x_tst).reshape(-1,14*14*4)

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
[CNN_GPU]
loss : 0.20780806243419647
acc  : 0.9316999912261963
acc  : 0.9317
시간 : 1904.9725253582

[DNN_GPU]
loss : 0.3334116041660309
acc  : 0.8885999917984009
acc  : 0.8886
시간 : 1643.7009906768799

[DNN_CPU]
loss : 0.33551329374313354
acc  : 0.8920999765396118
acc  : 0.8921
시간 : 739.784743309021

[LSTM]
loss : 0.705035388469696
acc  : 0.7476000189781189
acc  : 0.7476
시간 : 813.0245838165283

[Conv1D]
loss : 0.2796419560909271
acc  : 0.8998000025749207
acc  : 0.8998
시간 : 582.1401100158691
'''


##########################################################################
#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc']
              )

ES = EarlyStopping(monitor = 'val_acc', mode = 'max',
                   patience= 50, verbose=1,
                   restore_best_weights=True,
    
)

################################# mpc 세이브 파일명 만들기 #################################
### 월일시 넣기
import datetime

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')              

### 파일 저장
filepath = "".join([path,'k40_',date, '.h5'])

MCP = ModelCheckpoint(monitor='val_acc',
                      mode='auto',
                      save_best_only=True,
                      verbose = 1,
                      filepath= filepath        
                      )

Start = time.time()
hist = model.fit(x_trn, y_trn,
                 epochs = 100,
                 batch_size = 50,  
                 verbose = 3,
                 validation_split = 0.2,
                 callbacks = [ES, MCP])       
End = time.time()

T = End - Start

#4. 평가,예측
loss = model.evaluate(x_tst, y_tst,
                      verbose= 1)
results = model.predict(x_tst)
results = np.argmax(results, axis=1)
y_tst = np.argmax(y_tst.values, axis=1)
ACC = accuracy_score(results, y_tst)

print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print("loss :", loss[0])
print("acc  :", loss[1])
print("acc  :", ACC)
print("시간 :", T)