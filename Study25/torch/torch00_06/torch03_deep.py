############################################################
#0. 준비
############################################################
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import random

RS = 555
torch.manual_seed(RS)
np.random.seed(RS)
random.seed(RS)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'CPU' )

print('torch :', torch.__version__, 'devise :', DEVICE)
# torch : 2.7.1+cu126 devise : cuda
# tensor에서는 가상환경에 GPU 용으로 깔면 바로 사용
# torch에서는 두 가지 설정을 맞춰야지 GPU 사용

############################################################
#1. 데이터
############################################################

x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

x = torch.FloatTensor(x)
""" print(x)        : tensor([1., 2., 3.]) """
""" print(x.shape)  : torch.Size([3]) """
""" print(x.size()) : torch.Size([3]) """

# torch는 최소 matrix 형태의 데이터를 처리 (x, y 모두)
x = x.unsqueeze(1).to(DEVICE)

                    # unsqueeze(n) = flatten 반대 같은 느낌, n 번째 차원을 추가
""" print(x)        : tensor([[1.],
                              [2.],
                              [3.]]) """
""" print(x.shape)  : torch.Size([3, 1]) """
""" print(x.size()) : torch.Size([3, 1]) """

y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
""" print(y.size()) : torch.Size([3, 1]) """
mean = torch.mean(x)
std = torch.std(x)
############### Scaler ###############
x = (x - mean) / std
######################################
print('스케일링 후 :', x)

# exit()
# 데이터를 GPU 용으로 설정

############################################################
#2. 모델
############################################################
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1)
).to(DEVICE)

                                  # nn : neural network
                                  # nn.Linear(input, output)
                                  # linear : y = wx + b / 사실 tensorflow는 y = xw + b
                                                        # 행렬 연산의 기본 : (앞의 행) * (뒤의 열)
                                                        # x의 컬럼(행)에 맞춰서 가중치의 열을 생성
''' in tensorflow...
    model = Sequential()
    model.add(Dense(1, input_dim = 1)) '''

############################################################
#3. 컴파일 훈련
############################################################ 
criterion = nn.MSELoss()        #  criterion : 표준
# optimizer = optim.Adam(model.parameters(), lr = 0.01)
optimizer = optim.SGD(model.parameters(), lr = 0.002)
                # SGD, Stochastic Gradient Descent :통계적인 경사 하강법

def train(model, criterion, optimizer, x, y):
    # model.train()                     # [훈련모드] Default : DROPOUT, BATCHNORMAILATION 적용

    optimizer.zero_grad()               # 기울기 초기화 (이전값들을 초기화하고 현 가중치 때의 기울기만 사용)
                                        # 각 배치마다 기울기 초기화
                                        # >>> 기울기 누적에 의한 문제 해결
                                        # 경사 하강법에서는 통상적으로 해야지 성능 안정적
                                        # >>> 
                                        # >>> 안하면 과도하게 값이 커짐
                                        
    hypothesis = model(x)               # 가설 설정 == 모델 설정 : y =xw + b
                                        # hepothesis = y_prd
                                        
    loss = criterion(y, hypothesis)     # loss = mse() /// 여기까지가 순전파
    
    loss.backward()                     # 역전파(backward)시작 /// 가중치 갱신을 위해서 진행하는 기울기값까지 계산
                                        # 가중치 값이 나오는건 아님
                                        
    optimizer.step()                    # 가중치 갱신 /// 1 epoch (batch를 설정했으면 1 batch)
    
    return loss.item()                  # torchtensor 형태의 수치를 numpy로 전환

    # 기울기 계산시 이해해야하는 개념 : 미분, 편미분, Chain-Rule

epochs = 2000

for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print(f'epoch : {epoch:>3d} | loss : {loss}')
                
''' in tensorflow...
    model.compile(loss = 'mse',
                  optimizer = 'adam') '''

print('🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹')
############################################################
#4. 평가 예측
############################################################
''' in tensorflow...
    loss = model.evaluate(x, y) '''

def evaluate(model, criterion, x, y):
    model.eval()                        # [평가모드] 평가에 대해서는 반드시 들어가야하는 값
                                        #            훈련에서 진행한 DROPOUT, BATCHNORMAILATION을 적용하지 않음
                                        
    with torch.no_grad():               # gradient 갱신을 하지 않겠다
        y_prd = model(x)                # x_tst 값으로 y_prd 예측
        F_lss = criterion(y, y_prd)     # 최종 loss 값 계산
    
    return F_lss.item()

n = 4
F_lss = evaluate(model, criterion, x, y)
x_prd = (torch.Tensor([[n]]).to(DEVICE) - mean) / std

reslt = model(x_prd)

print('Final_loss :', F_lss)
print(f'Predict{[n]} :', reslt.item())

#🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
# Final_loss : 2.98394808861957e-10
# Predict[4] : 3.9999630451202393