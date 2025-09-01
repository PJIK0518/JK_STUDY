# a02_ae_noised.py - capy

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
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,))) # (28*28,)처럼 써야 튜플이 된다
    model.add(Dense(784, activation='sigmoid'))
    return model

"""
[실습 - PCA 넣어보고 확인]
필요한 hidden_size 수 아래에 넣어보기 
0.95 이상: 
0.99 이상:
0.999 이상:
1.0 일때:

"""

hidden_size = 64 ## 아래 autoencoder 전부 model로 바꿔준다 < 수치조절. 이전 과정에 수치를 조정할 필요 없이 이번에는 hidden에서 자동으로 조정

# autoencoder = Model(input_img, decoded)
model = autoencoder(hidden_layer_size=hidden_size)

# 3. 컴파일, 훈련
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2) # noised 인덱스로 수정 

# 4. 평가, 예측
decoded_imgs = model.predict(x_test_noised) 
#### autoencoder 인덱스를  model 인덱스로 지정 완료 ####

import matplotlib.pyplot as plt

# 5. 시각화
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noised[i].reshape(28, 28))
    plt.gray()
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 복원
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
