# 파라미터 튜닝 기능 중 하나, 가장 기본이 되는 녀석 : BayesianOptimization

param_bouns = {'x1' : (-1, 5),  # x1 및 x2의 파라미터의 범위 설정하는 ditionary 형성
               'x2' : ( 0, 4)} 

def y_function(x1, x2):         # x1 및 x2를 변수로 가지는 y에 대한 함수 설정
    return -x1 **2 - (x2 - 2) **2 + 10
    # y = -x1^2-(x2-2)^2+10
    # y_max : x1 = 0, x2 = 2, 일때 y_max = 10

y_max = 0
y_min = 0

""" For, If로 혼자 심심하니까 해본거
for x1 in range(param_bouns['x1'][0], param_bouns['x1'][1]+1):
    for x2 in range(param_bouns['x2'][0], param_bouns['x2'][1]+1):
        y = y_function(x1, x2)
        if y > y_max :
            y_max = y
            x1_max = x1
            x2_max = x2
        elif y < y_min :
            y_min = y
            x1_min = x1
            x2_min = x2

print('max :', y_max,(x1_max, x2_max))
print('min :', y_min,(x1_min, x2_min)) """

from bayes_opt import BayesianOptimization

Optimizer = BayesianOptimization(
    f = y_function,                  # y에 대한 함수로
    pbounds = param_bouns,           # x1, x2의 범위에서 적용해라
    random_state = 333,
)

Optimizer.maximize(init_points = 15, # 초기 탐색
                   n_iter = 50)      # 최적화 횟수
                                     # Optimizer에 대해서 총 25회 최대값을 찾아라

print(Optimizer.max)                 # Optimizer에서 찾은 Max 값을 출력해라 (핑크색이 갱신치)
                                     # 보통 이런애들 Gaussian process이라는 것을 사용