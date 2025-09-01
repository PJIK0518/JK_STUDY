import pandas as pd

# print(pd.__version__) 1.5.3

data = [
    ['삼성', '1000', '2000'],
    ['현대', '1100', '3000'],
    ['LG', '2000', '500'],
    ['아모레', '3500', '6000'],
    ['네이버', '100', '1500'],
]

index = ['031', '059', '033', '045', '023']
columns = ['종목명', '시가', '종가']

DF = pd.DataFrame(data=data, index=index, columns=columns)
""" print(DF)
     종목명    시가    종가
031   삼성   1000  2000
059   현대   1100  3000
033   LG     2000   500
045  아모레   3500  6000
023  네이버   100  1500 """

### [실습_1] ###
AAA = []
for i in range(6):
    a = DF['시가'][i]
    if a >= 1100:
        aa = DF.loc[i]
        AAA.append(aa)
        continue
print(AAA)

### [해설_1] ###
BBB = DF['시가'] >=1100
print(DF[BBB])

### [해설_2] ###
print(DF[DF['시가'] >= 1100])
# : Dataframe 뒤에 [중괄호] 안에 조건문을 넣을 수 있다

### [해설_3] ###
print(DF.loc[DF['시가'] >= 1100])

# 시가가 1100 이상인 행의 종가
DF_a = DF[DF['시가'] >= 1100]['종가']


