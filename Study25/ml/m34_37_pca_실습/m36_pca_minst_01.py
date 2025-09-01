from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import numpy as np

(x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()

# print(x_trn.shape, x_tst.shape, y_trn.shape, y_tst.shape)
# (60000, 28, 28) (10000, 28, 28) (60000,) (10000,)

x = np.concatenate([x_trn, x_tst], axis = 0)
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

print(len(np.where(evr_cumsum >= 1.)[0])+1)
print(len(np.where(evr_cumsum >= 0.999)[0])+1)
print(len(np.where(evr_cumsum >= 0.99)[0])+1)
print(len(np.where(evr_cumsum >= 0.95)[0])+1)