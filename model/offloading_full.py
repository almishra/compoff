import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler as mm_scaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from torchsummary import summary
import numpy as np
import copy
import glob
from collections import OrderedDict
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset class
class PrepareData(Dataset):
    def __init__(self, X, y, train=True):
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        if not torch.is_tensor(X):
            if train:
                X = self.sc1.fit_transform(X)
                self.X = torch.from_numpy(X)
            else:
                X = self.sc1.fit_transform(X)
                self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            y = y.to_numpy()
            if train:
                #y = self.sc2.fit_transform(np.reshape(y,(-1,1)))
                self.y = torch.from_numpy(y)
            else:
                #y = self.sc2.fit_transform(np.reshape(y,(-1,1)))
                self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_inverse(self,y):
        return self.sc2.inverse_transform(y)


class KernelRunModel(torch.nn.Module):
    def __init__(self, ip_features, num_hidden, op_features=1):
        super(KernelRunModel, self).__init__()

        self.hidden1 = torch.nn.Linear(ip_features, num_hidden)
        self.hidden2 = torch.nn.Linear(num_hidden, num_hidden*2)
        self.hidden3 = torch.nn.Linear(num_hidden*2, num_hidden)
        self.hidden4 = torch.nn.Linear(num_hidden, num_hidden)
        self.op_run = torch.nn.Linear(num_hidden, op_features)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        op = F.relu(self.hidden1(x))
        residual = op
        x = F.relu(self.hidden2(op))
        #x = self.dropout(x)
        x = F.relu(self.hidden3(x))
        x += residual
        x = F.relu(self.hidden4(x))
        x = self.dropout(x)
        x = F.relu(self.op_run(x))
        return x





dr_columns = ['kernel','Compiler','Cluster','gpu_name','div_int','div_double','assign_int',
              'thread_per_core', 'core_per_socket', 'num_sockets', 'cpu_clock', 'l1', 'l2', 'l3', 'connector_bandwidth', 'num_memory_bus',\
              ' memory_clock', 'memory_bandwidth', ' memory_total', ' sm_clock', 'num_cores', 'compute_capability', 'threads_per_wrap',\
              'max_wraps_per_sm', 'max_threads_per_sm', 'max_thread_blocks_per_sm', 'max_32-bit_registers_per_sm', 'max_registers_per_block',\
              'max_registers_per_thread', 'max_thread_block_size', 'fp32_cores_per_sm', 'sm_registers_per_fp32_cores', \
              'shared_memory_size_per_sm', 'div_double', 'div_int', 'assign_int', 'log_div_int', 'log_div_double', 'log_assign_int',
              'max_threads_per_sm','max_thread_blocks_per_sm','threads_per_wrap','max_wraps_per_sm','max_32-bit_registers_per_sm',\
              'max_registers_per_block','max_registers_per_thread','max_thread_block_size','l1','l2','l3',\
             'collapse', 'collapse_swap', 'combined', 'combined_swap', 'split', 'split_swap']

dataset_root=""
df = pd.read_csv(dataset_root+"matrix_multiplication.csv")


#single = pd.concat([df,df2], axis=0)

#single = single.loc[single['Cluster'] == 'Seawulf']
single = df.drop(columns=dr_columns)
#sys.exit("please check the dataset path and file names")

print(list(single.columns))
print(len(single))
print(len(list(single.columns)))
X = single.iloc[:, 0:-1]
y = single.iloc[:, -1]

train_eval_split=0.8
split_seed=43

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_eval_split, random_state=split_seed, shuffle=True)

#train_sets = CompData(X_train,y_train, scaler=True,train=True, task_num=3, num_sets=6, meta_train_batch=12, meta_test_batch=10)
train_sets = PrepareData(X_train, y_train, train=True)
test_sets = PrepareData(X_test, y_test, train=False)
train_loader = DataLoader(train_sets, batch_size=1, shuffle=True)



############# TRAIN SECTION

mod = KernelRunModel(31,50).to(device)
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

mod.apply(init_weights)

criterion = nn.MSELoss()
#criterion2 = nn.L1Loss()
opt = torch.optim.Adam(mod.parameters(), lr=1e-3)
for e in range(100):
    batch_losses = []

    for ix, (Xb, yb) in enumerate(train_loader):

        _X = Variable(Xb).float()
        _y = Variable(yb).float()

        #==========Forward pass===============
        _X = _X.to(device)
        _y = _y.to(device)
        preds = mod(_X)
        loss = criterion(preds, _y)/0.1
        #loss2 = criterion2(preds, _y)
        #total_loss = loss # + loss2
        #==========backward pass==============

        opt.zero_grad()
        loss.backward()
        opt.step()

        batch_losses.append(loss.item())
        #all_losses.append(loss.data[0])

    mbl = np.mean(np.sqrt(batch_losses)).round(3)

    if e % 1 == 0:
        print("Epoch [{}/{}], Batch loss: {}".format(e, 100, mbl))






############# EVALUATION SECTION

mod.eval()
with torch.no_grad():
    total_loss = 0
    gt_ = list()
    preds_ = list()
    for index, (xt, yt) in enumerate(test_sets):
        gt_.append(yt.cpu().data.numpy())
        
        _xt = Variable(xt).float()
        _yt = Variable(yt).float()
        
        _xt = _xt.to(device)
        _yt = _yt.to(device)
        
        predictions = mod(_xt)
        loss1 = criterion(predictions, _yt)
        preds_.append(predictions.cpu().data.numpy())
        print(predictions, _yt)
        total_loss += loss1
    
    mape = mean_absolute_percentage_error(gt_, preds_)
    rmse = np.sqrt(mean_squared_error(gt_, preds_))
    print('Test Loss: ', np.mean(np.sqrt(total_loss.item())))
    print('RMSE: ', rmse, ' MAPE:', mape)