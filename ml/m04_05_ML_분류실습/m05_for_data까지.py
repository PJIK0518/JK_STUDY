""" [실습 목록]
06_cancer
07_dacon_당뇨병
08_kaggle_bank
09_wine
10_covtype

11_digits
12_kaggle_santander
13_kaggle_otto
"""
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

#1. 데이터
DT_list = [('iris', load_iris(return_X_y=True)),
           ('breast_cancer', load_breast_cancer(return_X_y=True)),
           ('digits',load_digits(return_X_y=True)),
           ('wine',load_wine(return_X_y=True))]

#2. 모델구성
ML_list = [('M_LSV',LinearSVC()),
           ('M_LGR',LogisticRegression()),
           ('M_DTF',RandomForestClassifier()),
           ('M_RFC',DecisionTreeClassifier())]

#3 컴파일, 훈련
for data,(X, y) in DT_list:
    print('"""', [[[f'{data}']]])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    for M_nm, model in ML_list:
        model.fit(X, y)
        score = model.score(X, y)
        print(f'{M_nm} :', score)
        
    print('"""')

""" [[['iris']]]
M_LSV : 0.9466666666666667
M_LGR : 0.9733333333333334
M_DTF : 1.0
M_RFC : 1.0
"""
""" [[['breast_cancer']]]
M_LSV : 0.9876977152899824
M_LGR : 0.9876977152899824
M_DTF : 1.0
M_RFC : 1.0
"""
""" [[['digits']]]
M_LSV : 0.9944351697273233
M_LGR : 0.9988870339454646
M_DTF : 1.0
M_RFC : 1.0
"""
""" [[['wine']]]
M_LSV : 1.0
M_LGR : 1.0
M_DTF : 1.0
M_RFC : 1.0
"""        
