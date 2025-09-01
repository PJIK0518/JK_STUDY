# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition
# 45.copy
##########################################################################
#0. 준비
##########################################################################
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import time

RS = 518
##########################################################################
#1. 데이터
##########################################################################
path_NP = './Study25/_data/kaggle/cat_dog/'

x = np.load(path_NP + '04_x_train.npy')
y = np.load(path_NP + '04_y_train.npy')

x_trn,x_tst,y_trn,y_tst = train_test_split(x, y,
                                           train_size=0.6,
                                           shuffle=True,
                                           random_state=50,
                                           stratify=y,
                                           )

#####################################
### pred 데이터 설정
path_PD = './Study25/_data/kaggle/cat_dog/'

from sklearn.decomposition import PCA
import time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

x_trn = x_trn.reshape(x_trn.shape[0],-1)
x_tst = x_tst.reshape(x_tst.shape[0],-1)

n = 2
lda = LDA(n_components=n-1)
lda.fit(x_trn,y_trn)
x_trn = lda.transform(x_trn)
x_tst = lda.transform(x_tst)

from tensorflow.keras.utils import to_categorical

y_trn = to_categorical(y_trn, num_classes=n)
y_tst = to_categorical(y_tst, num_classes=n)

from sklearn.model_selection import train_test_split


#####################################
### Scaling
x_trn = x_trn/255.0
x_tst = x_tst/255.0


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
import time

S = time.time()

model = Sequential()
model.add(Dense(64, input_shape=(n-1,)))
model.add(Dropout(0.2))

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(16, activation = 'softmax'))
model.add(Dropout(0.2))

model.add(Dense(2, activation = 'softmax'))


#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['acc'])

model.fit(x_trn, y_trn,
                epochs = 50,
                batch_size = 50,        
                verbose = 1)
#4. 평가,예측
loss = model.evaluate(x_tst, y_tst)
results = model.predict(x_tst)
results = np.argmax(results, axis=1)
y_tst_a = np.argmax(y_tst, axis=1)
ACC = accuracy_score(results, y_tst_a)

E = time.time()
T = S-E
# LDA : 0.596875
print('LDA :', ACC)

'''
[LSTM]
save : 0623_0_5
loss : 0.6918511986732483
acc  : 0.5257750153541565

[Conv1D]
save : 0623_0_5
loss : 0.5684945583343506
acc  : 0.7196000218391418
'''