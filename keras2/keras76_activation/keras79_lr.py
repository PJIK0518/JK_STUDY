x = 10
y = 10
w = 0.001
lr = 0.0005
epochs = 12000

for e in range(1, epochs+1):
    hypothesis = x * w
    loss = (hypothesis - y) **2
    
    print('loss :', round(loss, 4), '\t predict :', round(hypothesis, 4))
    
    up_pred = x * (w+lr)
    up_loss = (y - up_pred) ** 2
    
    dw_pred = x * (w-lr)
    dw_loss = (y - dw_pred) ** 2
    
    if (up_loss > dw_loss):
        w = w - lr
    else:
        w = w + lr