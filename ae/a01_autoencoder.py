# GAN은 "생성적 적대 신경망" (Generative Adversarial Network)의 약자로, 인공지능 모델 중 하나로
# 오토인코더(Autoencoder)는 인코더를 통해 입력을 신호로 변환한 다음 다시 디코더를 통해 레이블 따위를 만들어내는 비지도 학습 기법
# 그러나 Tensorflow - GAN 발표 이후 학습 성능에서 밀림. 컬럼, 피처 전처리에 사용에 적합하기도 

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255

# 2. 모델
input_img = Input(shape=(28*28))

### 인코더
# encoded = Dense(1, activation='relu')(input_img)
# encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(1024, activation='relu')(input_img) 
# encoded = Dense(64, activation='tanh')(input_img) 

### 디코더
# decoded = Dense(28*28, activation='linear')(encoded)
# decoded = Dense(28*28, activation='relu')(encoded) # 흐릿했던 liner 보다 깔끔하게 보이기 위해 
# decoded = Dense(28*28, activation='sigmoid')(encoded) # relu 보다 더
# decoded = Dense(28*28, activation='tanh')(encoded) # 적합x sigmoid가 화질이 뚜렷

## 실습[최상의 화질] ##

encoded = Dense(1, activation='relu')(input_img) 
# encoded = Dense(32, activation='relu')(input_img) 
# encoded = Dense(128, activation='relu')(input_img) 
# encoded = Dense(256, activation='relu')(input_img) 
# encoded = Dense(784, activation='relu')(input_img) 
# encoded = Dense(1024, activation='relu')(input_img)

# decoded = Dense(28*28, activation='relu')(encoded) 
decoded = Dense(28*28, activation='sigmoid')(encoded) 

######################

autoencoder = Model(input_img, decoded)

# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy') # binary_crossentropy 

autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

# 5. 시각화
n = 5
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
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

# 노이즈 작업으로 다음 파일 준비해보자