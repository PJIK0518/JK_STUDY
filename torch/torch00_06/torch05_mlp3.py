############################################################
#0. 준비
############################################################
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optm

### 랜덤고정
RS = 55
torch.manual_seed(RS)
np.random.seed(RS)
random.seed(RS)

### GPU 설정
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA
                             else 'CPU')
""" print(torch.__version__) 2.7.1+cu128 """
""" print(DEVICE)            cuda """

############################################################
#1. 데이터
############################################################
x = np.array([range(10), range(21, 31), range(201,211)]).T
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1]]).T

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)

############################################################
#2. 모델
############################################################
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2)).to(DEVICE)

EPOCHS = 1000

############################################################
#3. 컴파일 훈련
############################################################
loss = nn.MSELoss()
optimizer = optm.Adam(model.parameters(), lr = 0.001)

def TRAIN(MODL, LOSS, OPTM, XTRN, YTRN):
    MODL.train()
    OPTM.zero_grad()
    x_trn_prd = MODL(XTRN)
    trn_loss= LOSS(YTRN, x_trn_prd)
    trn_loss.backward()
    OPTM.step()
    return trn_loss.item()

for e in range(1, EPOCHS+1):
    trn_loss = TRAIN(model, loss, optimizer, x, y)
    print('epo :', e)
    print('mse :', trn_loss)
    
############################################################
#4. 평가 예측
############################################################
def EVALUATE(MODL, LOSS, XTST, YTST):
    MODL.eval()
    
    with torch.no_grad():
         x_tst_prd = MODL(XTST)
         tst_loss = LOSS(YTST, x_tst_prd)
    return tst_loss.item()

tst_loss = EVALUATE(model, loss, x, y)

x_prd = np.array([[10, 31, 211], [11, 32, 212]])
x_prd = torch.tensor(x_prd, dtype=torch.float32).to(DEVICE)
result = model(x_prd)

print('# ✅ 최종 성능 :', tst_loss)
#################### [ERROR] ####################
# print(f'# 🔍 예측 결과 : {result.item()}')
# print(f'# 🔍 예측 결과 : {result.item()}')
# RuntimeError: a Tensor with 4 elements cannot be converted to Scalar
# item의 경우에는 하나의 항목만 불러오는 녀석 > 결과값이 2차원이상이라서 불가

### [실습 : 해결-1] ###
print(f'# 🔍 예측 결과 : {np.round(result.cpu().detach().numpy()[0])}')
print(f'# 🔍 예측 결과 : {np.round(result.cpu().detach().numpy()[1])}')
# cpu()    : 데이터를 cpu로 넘김
# detach() : 2차원 이상의 결과에서 Grad 값을 떼어냄
# numpy()  : 데이터의 numpy 형식으로 변경

### [실습 : 해결-2] ###
print(f'# 🔍 예측 결과 : {torch.round(result[0][0])} | {torch.round(result[0][1])}')
print(f'# 🔍 예측 결과 : {torch.round(result[1][0])} | {torch.round(result[1][1])}')
# list에서 수치를 빼오는 형식

# ✅ 최종 성능 : 1.7315756606350874e-09
# 🔍 예측 결과 : [11.  0.]
# 🔍 예측 결과 : [12. -1.]