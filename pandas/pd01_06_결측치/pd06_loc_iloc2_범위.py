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
031   삼성  1000  2000
059   현대  1100  3000
033   LG  2000   500
045  아모레  3500  6000
023  네이버   100  1500 """

print('========== 아모레와 네이버의 시가 ==========')
print(DF.iloc[3:]['시가'])
print(DF.iloc[3:,1])
print(DF['시가'][3:])
print(DF['시가']['045':])
print(DF.loc['045':]['시가'])
print(DF.loc['045':,'시가'])
# 045    3500
# 023     100
''' [[[ERROR]]] '''
# print(DF.iloc[3:][1])
# print(DF.iloc[3:]['시가'])