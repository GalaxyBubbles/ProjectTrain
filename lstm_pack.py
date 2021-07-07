from sys import argv
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
import datetime

tc = argv[1] #股票编号
ed = argv[2] #需要预测的日期（字符串） 如：20210305

class LSTMNET(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=10, output_size=1,num_layers=8):
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

# tc='603912.SH'
sd='20170101'
# ed='20201228'
model = torch.load('model_'+str(tc)+'.pkl')
model.eval()
# print(model)

t_str = ed[:4]+'-'+ed[5:6]+'-'+ed[-2:]
# print(t_str)
d = datetime.datetime.strptime(t_str, '%Y-%m-%d')
# print(d)
delta = datetime.timedelta(days=3)
dd = d + delta
# print(dd)
dd=dd.strftime('%Y-%m-%d %H:%M:%S')
ddd=dd[:4]+dd[6:7]+dd[9:10]
# print(ddd)
ed=ddd

ts.set_token('208cf0f7e03acf9024568071ab959d8cf3d1908385a7009906dc66d5') #需要在 tushare 官网申请一个账号，然后得到 token 后才能通过数据接口获取数据
pro = ts.pro_api()

time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
end_dt = time_temp.strftime('%Y%m%d')

#准备训练集数据

df = ts.pro_bar(ts_code=tc, start_date=sd, end_date=ed, freq='D')
df.head() #用 df.head() 可以查看一下下载下来的股票价格数据，显示数据如下：

#把数据按时间调转顺序，最新的放后面，从 tushare 下载的数据是最新的在前面，为了后面准备 X,y 数据方便
df = df.iloc[::-1]
df.reset_index(inplace=True)
# print(df)

dff=df.loc[:, ['trade_date','open','high','low','close','pre_close','change','pct_chg','vol','amount']]


#只用数据里面的收盘价字段的数据，也可以测试用更多价格字段作为预测输入数据
training_set = df.loc[:, ['close']]
#只取价格数据，不要表头等内容
training_set = training_set.values
training_set = training_set[-70:]
# #对数据做规则化处理，都按比例转成 0 到 1 之间的数据，这是为了避免真实数据过大或过小影响模型判断
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled1 = training_set_scaled
# print(training_set_scaled.shape)

# plt.figure()
# plt.plot(training_set_scaled,"r-")

#小波去噪处理

training_set_scaled=training_set_scaled.reshape(-1)

w = pywt.Wavelet('db8')  # 选用Daubechies8小波
maxlev = pywt.dwt_max_level(len(training_set_scaled), w.dec_len)
threshold = 0.05  # Threshold for filtering

coeffs = pywt.wavedec(training_set_scaled, 'db8', level=maxlev)  # 将信号进行小波分解
# print(coeffs[0].shape)
# print(len(coeffs))
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波

training_set_scaled = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构

# plt.plot(training_set_scaled,"b--")

training_set_scaled=np.array(training_set_scaled)
training_set_scaled=training_set_scaled.reshape(-1,1)
# print(training_set_scaled.shape)

X_train = []
y_train = []
devia=30
for i in range(devia, len(training_set_scaled)-1):
    X_train.append(training_set_scaled[i-devia:i])
    y_train.append(training_set_scaled[i+1, training_set_scaled.shape[1] - 1])
#     print("---:",training_set_scaled[i])
#     print(training_set_scaled[i+1, training_set_scaled.shape[1] - 1])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# X_train=X_train.reshape(-1,1,devia)
# print(X_train.shape)
# print(y_train.shape)

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
test_set = SharesDataset(y_train, X_train, transform=None)
test_loader = DataLoader(test_set, batch_size=bs)

for shares,labels in test_loader:
        test = Variable(shares, requires_grad=True)
        labels = Variable(labels, requires_grad=True)
        outputs= model(test)

print(outputs[0])