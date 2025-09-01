# aa04_ae2_그림.py - capy
# 그림 그리는 plt 쪽만 바꿔보자 학습하며 이미지 변환한 순서대로 한꺼본에 뽑아 낼 거다 

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape) # (60000, 784) (10000, 784)
# exit()
print(np.max(x_train), np.min(x_test))
print(np.max(x_train_noised), np.min(x_test_noised))
# exit()
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, 0, 1)

print(np.max(x_train_noised), np.min(x_test_noised)) # 1.0 0.0
# exit()

# 2. 모델
input_img = Input(shape=(28*28))

#### 파라미터(=매개변수) 지정 ####

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,))) 
    model.add(Dense(784, activation='sigmoid'))
    return model

hls_list = [1,8,32,64,157,331,486,713]
rlt_list = []
rlt_list.append(x_test)

for hls in hls_list:
     print(f'🔹🔹🔹🔹🔹🔹 hidden_layer_size = {hls} 🔹🔹🔹🔹🔹🔹')
     model = autoencoder(hidden_layer_size = hls)

     model.compile(optimizer='adam', loss='binary_crossentropy')
     model.fit(x_train_noised, x_train, epochs=20, batch_size=128, validation_split=0.2, verbose=0)

     decoded_imgs = model.predict(x_test_noised)
     
     rlt_list.append(decoded_imgs)
     
rlt_list = np.array(rlt_list)

import matplotlib.pyplot as plt
import random

fig, axes = plt.subplots(9, 5, figsize=(15,15))

random_images = random.sample(range(x_test.shape[0]), 5)
outputs = rlt_list

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()   