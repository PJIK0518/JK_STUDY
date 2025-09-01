from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()

x_trn = x_trn.reshape(x_trn.shape[0], -1)
x_tst = x_tst.reshape(x_tst.shape[0], -1)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

n = 10
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

model.add(Dense(10, activation = 'softmax'))


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

print('LDA :', ACC)

# LDA : 0.897
    
# n_components : 784 훈련 소요 시간 67.80681443214417
# acc  : 0.9477142857142857

# n_components : 713 훈련 소요 시간 66.53353452682495
# acc  : 0.9508571428571428

# n_components : 486 훈련 소요 시간 82.91819834709167
# acc  : 0.9507142857142857

# n_components : 331 훈련 소요 시간 64.08586955070496
# acc  : 0.9511428571428572

# n_components : 154 훈련 소요 시간 63.98296499252319
# acc  : 0.9544285714285714