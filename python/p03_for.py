### for : 반복문

#1. 리스트 읽기
aaa = [1, 2, 3, 4, 5]
for i in aaa:
    print(i)
    '''
    1
    2
    3
    4
    5
    '''

#2. 총합
add = 0
for i in range(1,11):
    add = add + i
print(add)     # 55

#3. 연결
results = []
for i in aaa:
    results.append(i+1)
print(results) # [2, 3, 4, 5, 6]

### 줄에 따라서 코드가 종속...?