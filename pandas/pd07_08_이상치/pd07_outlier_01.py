# 이상치 : 전체 데이터에서 정상적인 범주에서 벗어나는 데이터
#       > BUT. 정상적인 범주는 값은 같

# Tukey Fence : 데이터의 정상범위를 정의하는 방식
#   상한값 : Q3 + 1.5 * IQR
#   하한값 : Q1 - 1.5 * IQR

# 1.5 * IQR : 표준정규분포상으로 통상적으로 효과좋은 범위

import numpy as np

aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])

def outlier(data):
    Q1, Q2, Q3 = np.percentile(data, [25, 50, 75])
    print('제1사분위 :', Q1)
    print('제2사분위 :', Q2)
    print('제3사분위 :', Q3)
    
    IQR = Q3 - Q1
    print('IQR :', IQR)
    
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    return np.where((data > upper_bound) | (data < lower_bound)), \
        IQR, lower_bound, upper_bound
                                        # | : 또는, 두 가지 조건 중에 하나

outlier_loc, IQR, LOW, UP = outlier(aaa)

print(outlier_loc)  # array([ 0, 12])
print(IQR)          # 6.0

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.axhline(UP, color = 'violet', label = 'upper_bound')
plt.axhline(LOW, color = 'violet', label = 'lower_bound')
plt.legend()
plt.show()