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

# hidden_size = 64
model_01 = autoencoder(hidden_layer_size=1)
model_08 = autoencoder(hidden_layer_size=8)
model_32 = autoencoder(hidden_layer_size=32)
model_64 = autoencoder(hidden_layer_size=64)
model_154 = autoencoder(hidden_layer_size=154)
model_331 = autoencoder(hidden_layer_size=331)
model_486 = autoencoder(hidden_layer_size=486)
model_713 = autoencoder(hidden_layer_size=713)

# 3. 컴파일, 훈련
print("==================== node 1개 시작 ====================")
model_01.compile(optimizer='adam', loss='binary_crossentropy')
model_01.fit(x_train_noised, x_train_noised, epochs=20, batch_size=128, validation_split=0.2, verbose=0) 

print("==================== node 8개 시작 ====================")
model_08.compile(optimizer='adam', loss='binary_crossentropy')
model_08.fit(x_train_noised, x_train_noised, epochs=20, batch_size=128, validation_split=0.2, verbose=0) 

print("==================== node 32개 시작 ====================")
model_32.compile(optimizer='adam', loss='binary_crossentropy')
model_32.fit(x_train_noised, x_train_noised, epochs=20, batch_size=128, validation_split=0.2, verbose=0)  

print("==================== node 64개 시작 ====================")
model_64.compile(optimizer='adam', loss='binary_crossentropy')
model_64.fit(x_train_noised, x_train_noised, epochs=20, batch_size=128, validation_split=0.2, verbose=0)  

print("==================== node 154개 시작 ====================")
model_154.compile(optimizer='adam', loss='binary_crossentropy')
model_154.fit(x_train_noised, x_train_noised, epochs=20, batch_size=128, validation_split=0.2, verbose=0)  

print("==================== node 331개 시작 ====================")
model_331.compile(optimizer='adam', loss='binary_crossentropy')
model_331.fit(x_train_noised, x_train_noised, epochs=20, batch_size=128, validation_split=0.2, verbose=0)  

print("==================== node 486개 시작 ====================")
model_486.compile(optimizer='adam', loss='binary_crossentropy')
model_486.fit(x_train_noised, x_train_noised, epochs=20, batch_size=128, validation_split=0.2, verbose=0)  

print("==================== node 713개 시작 ====================")
model_713.compile(optimizer='adam', loss='binary_crossentropy')
model_713.fit(x_train_noised, x_train_noised, epochs=20, batch_size=128, validation_split=0.2, verbose=0)  



# 4. 평가, 예측
decoded_imgs_01 = model_01.predict(x_test_noised)
decoded_imgs_08 = model_08.predict(x_test_noised)
decoded_imgs_32 = model_32.predict(x_test_noised)
decoded_imgs_64 = model_64.predict(x_test_noised)
decoded_imgs_154 = model_154.predict(x_test_noised)
decoded_imgs_331 = model_331.predict(x_test_noised)
decoded_imgs_486 = model_486.predict(x_test_noised)
decoded_imgs_713 = model_713.predict(x_test_noised) 

import matplotlib.pyplot as plt
import random

fig, axes = plt.subplots(9, 5, figsize=(15,15))

random_images = random.sample(range(decoded_imgs_01.shape[0]), 5)
outputs = [x_test, decoded_imgs_01, decoded_imgs_08, decoded_imgs_32, decoded_imgs_64, decoded_imgs_154, decoded_imgs_331, decoded_imgs_486, decoded_imgs_713]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()   