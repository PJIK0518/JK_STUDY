# 함수   : 기능을 모아놓은 형식
# 클래스 : 여러가지 기능 및 변수에 의한 

class Father:
    def __init__(self, name):
        self.name = name
        print('Father __init__ 실행됨')
        print(self.name, '아빠')

# aaa = Father('재익')

class Son(Father):
    def __init__(self, K):
        print('Son __init__  시작')
        super().__init__(K)
        print('Son __init__  끝')

bbb = Son('흥민')