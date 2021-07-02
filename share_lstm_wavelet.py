#基本lstm+wavelet优化
#对原始数据使用小波进行降噪处理

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts 
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pywt

##数据准备

ts.set_token('208cf0f7e03acf9024568071ab959d8cf3d1908385a7009906dc66d5') #需要在 tushare 官网申请一个账号，然后得到 token 后才能通过数据接口获取数据
pro = ts.pro_api()

time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
end_dt = time_temp.strftime('%Y%m%d')

#准备训练集数据

#这里是用 000001 平安银行为例，下载从 2015-1-1 到最近某一天的股价数据
df = ts.pro_bar(ts_code='603912.SH', start_date='20200731', end_date='20201228', freq='D')
df.head() #用 df.head() 可以查看一下下载下来的股票价格数据，显示数据如下：

#把数据按时间调转顺序，最新的放后面，从 tushare 下载的数据是最新的在前面，为了后面准备 X,y 数据方便
df = df.iloc[::-1]
df.reset_index(inplace=True)
# print(df)

#只用数据里面的收盘价字段的数据，也可以测试用更多价格字段作为预测输入数据
training_set = df.loc[:, ['close']]
#只取价格数据，不要表头等内容
training_set = training_set.values
# #对数据做规则化处理，都按比例转成 0 到 1 之间的数据，这是为了避免真实数据过大或过小影响模型判断
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

print(training_set_scaled.shape)

plt.figure()
plt.plot(training_set_scaled,"r-")

#小波去噪处理

training_set_scaled=training_set_scaled.reshape(-1)

w = pywt.Wavelet('db8')  # 选用Daubechies8小波
maxlev = pywt.dwt_max_level(len(training_set_scaled), w.dec_len)
threshold = 0.04  # Threshold for filtering

coeffs = pywt.wavedec(training_set_scaled, 'db8', level=maxlev)  # 将信号进行小波分解
print(coeffs[0].shape)
print(len(coeffs))
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波

training_set_scaled = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

plt.plot(training_set_scaled,"b--")

training_set_scaled=np.array(training_set_scaled)
training_set_scaled=training_set_scaled.reshape(-1,1)
print(training_set_scaled.shape)


# print(training_set_scaled)
#准备 X 和 y 数据，就类似前面解释的，先用最近一个交易日的收盘价作为第一个 y，然后这个交易日以前的 60 个交易日的收盘价作为 X。
#这样依次往前推，例如最近第二个收盘价是第二个 y，而最新第二个收盘价以前的 60 个交易日收盘价作为第二个 X，依次往前准备出大量的 X 和 y，用于后面的训练。
X_train = []
y_train = []
devia=10
for i in range(devia, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-devia:i])
    y_train.append(training_set_scaled[i, training_set_scaled.shape[1] - 1])
#     print("---:",training_set_scaled[i])
#     print(training_set_scaled[i+1, training_set_scaled.shape[1] - 1])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# X_train=X_train.reshape(-1,1,devia)
print(X_train.shape)
print(y_train.shape)


#准备测试集数据

dft = ts.pro_bar(ts_code='603912.SH', start_date='20210101', end_date='20210604', freq='D')
dft.head()

dft = dft.iloc[::-1]
dft.reset_index(inplace=True)
# print(dft)

testing_set = dft.loc[:, ['close']]
testing_set = testing_set.values
sct = MinMaxScaler(feature_range = (0, 1))
testing_set_scaled = sct.fit_transform(testing_set)
# print(training_set_scaled)
X_test = []
y_test = []
devia=10
for i in range(devia, len(testing_set_scaled)):
    X_test.append(testing_set_scaled[i-devia:i])
    y_test.append(testing_set_scaled[i, testing_set_scaled.shape[1] - 1])

X_test, y_test = np.array(X_test), np.array(y_test)
# X_test=X_test.reshape(-1,1,devia)
print("X_test:",X_test.shape)
print("y_test:",y_test.shape)


##需要确定的参数：batch_size=bs,epoch,初始learning_rate,learning_rate_decay,

##生成训练集
class SharesDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""

    def __init__(self, labels, shares, transform=None):
        """Method to initilaize variables.""" 
        self.transform=transform
        self.labels = labels
        self.shares = shares
    def __getitem__(self, index):
        label = self.labels[index]
        share = self.shares[index]

        if self.transform is not None:
            share = self.transform(share)

        return share, label

    def __len__(self):
        return len(self.labels) 

bs=1
train_set = SharesDataset(y_train, X_train, transform=None)
train_loader = DataLoader(train_set, batch_size=bs)
test_set = SharesDataset(y_test, X_test, transform=None)
test_loader = DataLoader(test_set, batch_size=bs)

##建立模型

class LSTMNET(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1,num_layers=8):
        super(LSTMNET,self).__init__()
        self.lstm_layer=nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True) 
        self.out_layer1 = nn.Linear(hidden_layer_size,output_size,bias=True)
        self.out_layer2 = nn.Linear(num_layers,output_size,bias=True)
        
    def forward(self,share):
        out,(h,c)=self.lstm_layer(share.to(torch.float32))
        out=h
#         print("out:",out.shape)
#         print("h:",h.shape)
        out=self.out_layer1(out)
#         print("out1:",out.shape)
        a,b,c=out.shape
        out=out.reshape(b,a)
        out=self.out_layer2(out)
#         print("out2:",out.shape)
#         a,b,c=out.shape
#         out=out.reshape(b,a)
        return out
    
print(LSTMNET)

model=LSTMNET()

epochs = 10

#初始化训练参数
learning_rate=0.1

#损失函数
error = nn.CrossEntropyLoss(reduction='mean')
loss_MSE = nn.MSELoss(reduction='mean')
loss_MAE = nn.L1Loss(reduction='mean')

#优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)
 
#预测结果
pred=[]
actl=[]
rmse=[]
mae=[]

count=0

for i in range(epochs):
    for shares, labels in train_loader:
        train = Variable(shares, requires_grad=True)
        labels = Variable(labels, requires_grad=True)
        outputs= model(train)
#         labels=labels.reshape(bs)
#         outputs=outputs.reshape(10)
#         print("input:",outputs.shape)
#         print("target:",labels.shape)
#         print("outputs:",outputs)
#         print("labels:",labels)
#         loss = error(outputs, labels.long())
        
#         outputs = torch.autograd.Variable(outputs)
        loss_mse = loss_MSE(outputs.float(), labels.float())
        loss_rmse = torch.sqrt(loss_mse)
        loss_mae = loss_MAE(outputs.float(), labels.float())
        
        optimizer.zero_grad()
        loss_rmse.backward()
        optimizer.step()
        
        count=count+1
        if count%100==0:
            rmse.append(loss_rmse.data)
            mae.append(loss_mae.data)
        
        if i==epochs*bs-1:
            a,_=outputs.shape
            for j in range(a):
                pred.append(outputs[j])
                actl.append(labels[j])

    print("epoch:{}    Train:  RMSE:{:.8f}  MAE:{:.8f} ".format(i, loss_rmse.data, loss_mae.data))
    
    
    for shares,labels in test_loader:
        test = Variable(shares, requires_grad=True)
        labels = Variable(labels, requires_grad=True)
        outputs= model(test)
#         loss = error(outputs, labels.long())
        outputs= torch.autograd.Variable(outputs)
        loss_mse = loss_MSE(outputs.float(), labels.float())
        loss_rmse = torch.sqrt(loss_mse)
        loss_mae = loss_MAE(outputs.float(), labels.float())
        
    print("            Test:  RMSE:{:.8f}  MAE:{:.8f}".format( loss_rmse.data, loss_mae.data))   
        
        
plt.figure()
plt.plot(pred,"r-")
plt.plot(actl,"b-")


plt.figure(figsize=(8, 4))
plt.plot(rmse,"r-")
plt.plot(mae,"b--")
