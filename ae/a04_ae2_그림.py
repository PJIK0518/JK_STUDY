# aa03.py - capy
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

hidden_size = 64 # hidden_size 64로 학습하며 이미지 변환한 순서대로 그려볼거다
model = autoencoder(hidden_layer_size=hidden_size)

# 3. 컴파일, 훈련
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2) 

# 4. 평가, 예측
decoded_imgs = model.predict(x_test_noised) 

import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5),
      (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 7))

# 랜덤 5개 샘플 선택
random_images = random.sample(range(decoded_imgs.shape[0]), 5)

# 1행: 원본(noise 없는 원래 이미지)
for i, ax in enumerate((ax1, ax2, ax3, ax4, ax5)):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('ORIGINAL', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 2행: 노이즈 이미지
for i, ax in enumerate((ax6, ax7, ax8, ax9, ax10)):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('NOISED', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 3행: 오토인코더 복원 이미지
for i, ax in enumerate((ax11, ax12, ax13, ax14, ax15)):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel('DENOISED', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()