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
################# [[pandas 열행]] #################
# DF를 열(컬럼이 기준)을 불러오고 행을 불러올 수 있다
""" print(DF) :: 정상
     종목명    시가    종가
031   삼성    1000    2000
059   현대    1100    3000
033   LG      2000    500
045   아모레  3500    6000
023   네이버   100    1500 """
""" print(DF[0]) KeyError : 0 """
""" print(DF['031']) KeyError : 031 """
""" print(DF['종목명']) : 정상
031     삼성
059     현대
033     LG
045    아모레
023    네이버 """

############### [[아모레 출력]] ###############
""" print(DF[3,1]) KeyError : (3, 1) """
""" print(DF['045','종목명']) KeyError : ('045', '종목명') """
""" print(DF['045']['종목명']) KeyError : '045' """
""" print(DF['종목명']['045']) : 정상
아모레
"""

# loc  : 인덱스 기준으로 행 데이터 추출
# iloc : 행 번호를 기준으로 행 데이터 추출
       # int loc로 외워라 (사실 index loc)

print('========== 아모레 뽑기 ==========')
print(DF.loc['045'])
print(DF.iloc[3])
# 종목명     아모레
# 시가       3500
# 종가       6000
""" print(DF.iloc['045']) TypeError: Cannot index by location index with a non-integer key """
""" print(DF.loc[3]) KeyError: 3 """

print('========== 네이버 뽑기 ==========')
print(DF.loc['023'])
print(DF.iloc[4])
# 종목명     네이버
# 시가       100
# 종가       1500

print('========== 아모레 종가 뽑기 ==========')
print(DF.iloc[3][2])                    # 6000
print(DF.iloc[3]['종가'])               # 6000
print(DF.iloc[3, 2])                    # 6000
print(DF['종가'].iloc[3])               # 6000
print(DF.loc['045'][2])                 # 6000
print(DF.loc['045']['종가'])            # 6000
print(DF.loc['045','종가'])             # 6000
print(DF['종가'].loc['045'])            # 6000
print(DF.iloc[3].iloc[2])               # 6000
print(DF.iloc[3].loc['종가'])           # 6000
print(DF.loc['045'].loc['종가'])        # 6000
print(DF.loc['045'].iloc[2])            # 6000

""" print(DF.loc['045', 2]) KeyError: 2 """
""" print(DF.iloc[3,'종가']) ValueError: Location """

############### [DF 데이터 불러오기] ###############
# DF에서 바로 불러오기 : DF['column']['index']
                    # : DF['column'][n]
                    # pandas 열행
                       
# loc_iloc로 불러오기 : DF.loc['index']['column']
                    # : DF.loc['index'][m]
                    # : DF.iloc[n]['column']
                    # : DF.iloc[n][m]
    # loc  : 실제 행 이름으로 출력
    # iloc : 행 번호로 출력
        # loc_iloc 행열

        # pandas 열행, loc_iloc 행열 순서, 실제 이름 - 행 번호
        # : 어느정도 유연하게 적용 가능
        # 단, 서로 다른 형태의 정보를 한 대괄호 안에 섞으면 불가