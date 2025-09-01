import numpy as np

A = [10,20,30,40,50]

# print(np.percentile(A, 25)) 20.

A = [10,20,30,40]
# 

# print(np.percentile(A, 25)) 17.5
# np.percentile(A, Q) : X
# (X - Amin) / (Amax - Amin) = Q
# >>> 중요도 따라서 하위 순위 잘라내기 or 상위 순위 선택하기 가능!

''' index의 위치 찾기
A = [10,20,30,40]
(np.percentile(A, 25) : 2.0
rank = (n - 1) * ( Q / 100)
     = (4 - 1) * (25 / 100)
     = 3 * 0.25 = 0.75
'''

''' 보간법
작은 값 = Data의 0번째 = 10 (위치할 것으로 예상되는 뒷 값)
큰 값 = Data의 1번째 = 20   (위치할 것으로 예상되는 앞 값)

백분위 값 = 작은 값 + (큰 값 - 작은 값) * rank
          = 10 + (20 - 10)*0.75
          = 10 + 7.5 = 17.5
'''