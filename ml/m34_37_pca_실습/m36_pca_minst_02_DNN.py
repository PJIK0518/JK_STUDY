from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np

(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()

# print(x_trn.shape, x_tst.shape, y_trn.shape, y_tst.shape)
# (60000, 28, 28) (10000, 28, 28) (60000,) (10000,)

x = np.concatenate([x_trn, x_tst], axis = 0)
y = np.concatenate([y_trn, y_tst], axis = 0)

# print(x.shape) (70000, 28, 28)

x = x.reshape(70000,28*28)
# print(x.shape) (70000, 784)

pca = PCA(n_components=784)

x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
# print(evr_cumsum)
##### [실습] #####
#1. 1.0 일 때 몇개?    : 72
#2. 0.999 이상은 몇개? : 299
#2. 0.99 이상은 몇개?  : 454
#2. 0.95 이상은 몇개?  : 631
# print(np.unique(evr_cumsum, return_counts=True))

# print(len(evr_cumsum))
# print(len(evr_cumsum)-len(np.where(evr_cumsum >= 1.)[0]))
# print(len(evr_cumsum)-len(np.where(evr_cumsum >= 0.999)[0]))
# print(len(evr_cumsum)-len(np.where(evr_cumsum >= 0.99)[0]))
# print(len(evr_cumsum)-len(np.where(evr_cumsum >= 0.95)[0]))
# n_conponent = 784
# n_conponent = 713
# n_conponent = 486
# n_conponent = 331
# n_conponent = 154
from sklearn.model_selection import train_test_split

x_trn, x_tst, y_trn, y_tst = train_test_split(
    x, y,
    train_size= 0.9,
    random_state=42
)

# print(x_trn.shape)
# print(x_tst.shape)
# print(y_trn.shape)
# print(y_tst.shape)
# (63000, 784)
# (7000, 784)
# (63000,)
# (7000,)

#####################################
### Scaling
x_trn = x_trn/255.0
x_tst = x_tst/255.0

#####################################
### OneHot
import pandas as pd
y_trn = pd.get_dummies(y_trn).values
y_tst = pd.get_dummies(y_tst).values

# print(x_trn.shape)
# print(x_tst.shape)
# print(y_trn.shape)
# print(y_tst.shape)

# (63000, 784)
# (7000, 784)
# (63000, 10)
# (7000, 10)
# exit()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense
import time

num = [784, 713, 486, 331, 154]
acc = []

for p in num :
    pca = PCA(n_components=p)
    pca.fit(x_trn)
    x_trn = pca.transform(x_trn)
    x_tst = pca.transform(x_tst)
    
    S = time.time()
    
    model = Sequential()
    model.add(Dense(64, input_shape=(p,)))
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
                    epochs = 1,
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
    
    acc.append((p, ACC, T))
    
print(acc)

for p, ACC, T in acc:
    print(f"n_components={p:>3} | acc={float(ACC):.4f} | time={T:.2f} sec")
    
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