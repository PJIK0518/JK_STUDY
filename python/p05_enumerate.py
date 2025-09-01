list = ['a', 'b', 'c', 'd', 5]

# print(list) ['a', 'b', 'c', 'd', 5]

for i in list:
    print(i)
# a
# b
# c
# d
# 5

for index, value in enumerate(list):
    print(index,value)
# 0 a
# 1 b
# 2 c
# 3 d
# 4 5

# list에는 뭐든 넣을 수 있음 : 여러개의 모델을 리스트에 넣어두고
#                            원하는 순번의 모델을 뽑아쓰기도 가능