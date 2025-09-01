# a01_autoincoder.py - capy
# 현재도 생성형ai가 이미지 개선으로 최상의 가중치로 뽑아냄
# 조선시대 이미지 개선 or 사진관

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255
                                     # 평균 0
## # 노이즈 # noise오차값에 > image의 갯수 0 ~ 255 사이 출력되는 
## 각 오차값에서 넘어가는 랜덤 값을 주겠다 < 평균 0, 표면 0.1인 정규분포 형태의 랜덤값 - 노이즈 수치를 조절 # 해보니 0.1 이 
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

# print(x_train.shape, x_test.shape)
# print(x_train_noised.shape, x_test_noised.shape) # (60000, 784) (10000, 784)
# exit()
# print(np.max(x_train), np.min(x_test)) # 1.0 0.0 << 255로 정규화햔 값이기에,
# print(np.max(x_train_noised), np.min(x_test_noised)) # 1.4892949708881171 -0.5160416919563203 << 그렇게 많이 소실 시키지 않은걸로 이해, 
                                                                                                # (원본값에서 표준 정규화 이후에서 본값 언저리 부근에 noise +-오차  값)
                                                                                                # 극소수지만 1이상의 큰 값이 더 밝아지게 대비되는 효과 시각화 그려볼 예정  
# exit()
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, 0, 1)
print(np.max(x_train_noised), np.min(x_test_noised)) # 1.0 0.0 # 현재도 생성형ai가 이미지 개선으로 최상의 가중치로 뽑아냄,  
                                                     # loss 조절로 최상의 결과를 판단함, 결과값에 입렵될 fit에도 적용 해보자
# exit()

# 2. 모델
input_img = Input(shape=(28*28))

### 인코더
# encoded = Dense(1, activation='relu')(input_img)
# encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(1024, activation='relu')(input_img) 
# encoded = Dense(64, activation='tanh')(input_img) 
### 디코더
# decoded = Dense(28*28, activation='linear')(encoded)
# decoded = Dense(28*28, activation='relu')(encoded) 
# decoded = Dense(28*28, activation='sigmoid')(encoded) 
# decoded = Dense(28*28, activation='tanh')(encoded) 

    ## 실습[최상의 화질( 되도록 적은 노드면 good )] ##

# encoded = Dense(1, activation='relu')(input_img) 
# encoded = Dense(32, activation='relu')(input_img) 
encoded = Dense(64, activation='relu')(input_img) 
# encoded = Dense(256, activation='relu')(input_img) 
# encoded = Dense(784, activation='relu')(input_img) 
# encoded = Dense(1024, activation='relu')(input_img) 

# decoded = Dense(28*28, activation='relu')(encoded) 
decoded = Dense(28*28, activation='sigmoid')(encoded) 

    ######################

autoencoder = Model(input_img, decoded)

# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noised, x_train, epochs=50, batch_size=128, validation_split=0.2) # noised 인덱스로 수정 

# 4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised) # noised 인덱스로 수정

import matplotlib.pyplot as plt

# 5. 시각화
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noised[i].reshape(28, 28)) # 여기도 인덱스 수정
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
