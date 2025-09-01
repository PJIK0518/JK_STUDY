import numpy as np

aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T

from sklearn.covariance import EllipticEnvelope
outlier = EllipticEnvelope(contamination=.1)

outlier.fit(aaa)
results = outlier.predict(aaa)
print(aaa)
print(results)

for i in range(len(aaa)):
    if results[i] == 1:
        continue
    if results[i] == -1:
        print(aaa[i])