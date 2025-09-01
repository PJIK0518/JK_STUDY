import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])
###### [print(data)] ######
#      0    1    2  3     4
# 0  2.0  NaN  6.0  8  10.0
# 1  2.0  4.0  NaN  8   NaN
# 2  2.0  4.0  6.0  8  10.0
# 3  NaN  4.0  NaN  8   NaN

data = data.transpose()
#[print(data)]
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 0. 결측치 확인 ##########################################
""" print(data.isnull()) 
       0      1      2      3
0  False  False  False   True
1   True  False  False  False
2  False   True  False   True
3  False  False  False  False
4  False   True  False   True """
""" print(data.isnull().sum())
0    1
1    2
2    0
3    3 """
""" print(data.info())
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   0       4 non-null      float64
 1   1       3 non-null      float64
 2   2       5 non-null      float64
 3   3       2 non-null      float64 """

# 1. 결측치 삭제
""" data = data.dropna() # 디폴트 : axis = 0 : 행 삭제
print(data)
     0    1    2    3
3  8.0  8.0  8.0  8.0 """
""" data = data.dropna(axis=1)
print(data)
      2
0   2.0
1   4.0
2   6.0
3   8.0
4  10.0 """

# 2_1. 특정값 : 평균
means = data.mean()
""" print(means)
0    6.500000
1    4.666667
2    6.000000
3    6.000000 """
""" data = data.fillna(means)
print(data)
      0         1     2    3
0   2.0  2.000000   2.0  6.0
1   6.5  4.000000   4.0  4.0
2   6.0  4.666667   6.0  6.0
3   8.0  8.000000   8.0  8.0
4  10.0  4.666667  10.0  6.0 """

# 2_2. 특정값 : 중위값
# 전체 데이터에 대한 중위값
mid = data.median()
""" print(mid)
0    7.0
1    4.0
2    6.0
3    6.0 """
""" data = data.fillna(mid)
print(data)
      0    1     2    3
0   2.0  2.0   2.0  6.0
1   7.0  4.0   4.0  4.0
2   6.0  4.0   6.0  6.0
3   8.0  8.0   8.0  8.0
4  10.0  4.0  10.0  6.0 """

# 2_3. 특정값 : 0
""" data = data.fillna(0)
print(data)
      0    1     2    3
0   2.0  2.0   2.0  0.0
1   0.0  4.0   4.0  4.0
2   6.0  0.0   6.0  0.0
3   8.0  8.0   8.0  8.0
4  10.0  0.0  10.0  0.0 """

# 2_4. 특정값 : 132
""" data = data.fillna(132)
print(data)
       0      1     2      3
0    2.0    2.0   2.0  132.0
1  132.0    4.0   4.0    4.0
2    6.0  132.0   6.0  132.0
3    8.0    8.0   8.0    8.0
4   10.0  132.0  10.0  132.0 """

# 2_5. 특정값 : ffill (통상적으로 마지막 값 및 시계열 데이터)
""" data = data.ffill() # 제일 첫 테이터는 처리 불가
print(data)
      0    1     2    3
0   2.0  2.0   2.0  NaN
1   2.0  4.0   4.0  4.0
2   6.0  4.0   6.0  4.0
3   8.0  8.0   8.0  8.0
4  10.0  8.0  10.0  8.0 """

# 2_6. 특정값 : bfill (통상적으로 첫 번째 값 및 시계열 데이터)
""" data = data.bfill() # 마지막 테이터는 처리 불가
print(data)
      0    1     2    3
0   2.0  2.0   2.0  4.0
1   6.0  4.0   4.0  4.0
2   6.0  8.0   6.0  8.0
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN """

########## [[ 컬럼 지정 및 컬럼별로 결측치 처리]] ##########
data.columns = ['x1', 'x2', 'x3', 'x4']
""" print(data)
     x1   x2    x3   x4
0   2.0  2.0   2.0  NaN
1   NaN  4.0   4.0  4.0
2   6.0  NaN   6.0  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN """

means = data['x1'].mean()
""" print(means) 6.5 """

mid = data['x4'].median()
""" print(mid) 6.0 """

data['x1'] = data['x1'].fillna(means)
data['x2'] = data['x2'].ffill()
data['x4'] = data['x4'].fillna(mid)
""" print(data)
     x1   x2    x3   x4
0   2.0  2.0   2.0  6.0
1   6.5  4.0   4.0  4.0
2   6.0  4.0   6.0  6.0
3   8.0  8.0   8.0  8.0
4  10.0  8.0  10.0  6.0 """
