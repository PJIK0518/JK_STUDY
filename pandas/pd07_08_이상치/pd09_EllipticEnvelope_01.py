# 데이터 사이의 관계를 확인하는 방법
# 평균 및 공분산으로 데이터를 타원형태의 군집화
# > Mahalanobis 거리를 측정하여 이상치 추적

# 컬럼 안에서 
import numpy as np

aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])
aaa = aaa.reshape(-1, 1)

from sklearn.covariance import EllipticEnvelope

outlier = EllipticEnvelope(contamination=.1)

outlier.fit(aaa)
results = outlier.predict(aaa)

print(results)