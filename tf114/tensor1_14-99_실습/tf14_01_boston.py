import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
tf.compat.v1.random.set_random_seed(518)

#1. 데이터
(x_trn, y_trn), (x_tst, y_tst) = boston_housing.load_data()
print(x_trn.shape, y_trn.shape)
print(x_tst.shape, y_tst.shape)

