# a03_ae1_pca.py - capy

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1-1. 데이터 로드 & 정규화
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255

# 1-2. 노이즈 추가 & 0~1 범위로 클리핑
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised = np.clip(x_test_noised, 0, 1)

# 2. PCA로 필요한 차원 수 계산
pca = PCA().fit(x_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
ratios = [0.95, 0.99, 0.999, 1.0]
dims = [np.argmax(cumsum >= r) + 1 for r in ratios]
print("PCA 기준별 필요한 차원 수:")
for r, d in zip(ratios, dims):
    print(f"  {r}: {d}")

# 3. 오토인코더 모델 함수
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

# 4. 각 PCA 기준별 학습 & 결과 저장
results = {}
for ratio, dim in zip(ratios, dims):
    print(f"\n=== PCA {ratio} 이상 (hidden={dim}) 학습 시작 ===")
    model = autoencoder(hidden_layer_size=dim)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(
        x_train_noised, x_train_noised,
        epochs=10,           # 시간이 길면 늘리세요
        batch_size=128,
        validation_split=0.2,
        verbose=0
    )
    decoded_imgs = model.predict(x_test_noised, verbose=0)
    results[ratio] = decoded_imgs

# 5. 시각화
n = 5  # 비교할 샘플 개수
plt.figure(figsize=(20, 8))

# (1) 노이즈 원본 표시
for i in range(n):
    ax = plt.subplot(len(ratios) + 1, n, i + 1)
    plt.imshow(x_test_noised[i].reshape(28, 28), cmap='gray')
    ax.set_title("Noised")
    ax.axis('off')

# (2) 각 PCA 기준별 복원 결과 표시
for row, ratio in enumerate(ratios):
    decoded_imgs = results[ratio]
    for i in range(n):
        ax = plt.subplot(len(ratios) + 1, n, (row + 1) * n + i + 1)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"PCA {ratio}")
        ax.axis('off')

plt.tight_layout()
plt.show()

"""
PCA 기준별 필요한 차원 수:
0.95 이상: 154
0.99 이상: 331
0.999 이상: 486
1.0 일때: 687
"""