import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=10.0, size=1000)
               # exponential : 지수함수
                     # scale : 지수분포도에서 평균을 10.0으로 지정
                     # size  : 생성하려는 데이터 수
                     # > 무작위적으로 지수분포 평균 10.0을 가지도록 1000개의 데이터 생성

# print(data)
print(data.shape)                   # (1000,)
print(np.max(data), np.min(data))   # 73.86990600595156 0.0073830833375246535

# 데이터를 로그형태로 변환
log_data = np.log1p(data) # 데이터 중에 0일 때를 대비해서 데이터에 1 더하기

plt.subplot(1,2,1)
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('DATA')

plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('LOG_DATA')

plt.show()

#            분포                              범위      
# DATA     : 축 방향으로 편향된 데이터          0~70     
# LOG_DATA : StandardScaler 느낌 >>            0~4      
#            평균 주변에 많이 분포하도록 변환 | y 값의 경우 예측하고 변환하면 오차가 커짐
                                            # 범위가 몇 십만까지 커지면 그 정도 오차를 감수하는게 성능이 더 좋을수도

re_data = np.expm1(log_data)

# print(re_data)
print(re_data.shape)
print(np.max(re_data), np.min(re_data))