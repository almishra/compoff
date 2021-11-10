import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
#import sys
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from torchsummary import summary
import numpy as np
import copy
import glob
from collections import OrderedDict
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from deepres import DeepRes, Block

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrepareData(Dataset):
    def __init__(self, X, y, train=True, scaler_obj=None):
        #self.sc1 = MinMaxScaler(feature_range=(0,5))
        self.sc1 = StandardScaler()
        #self.sc2 = MinMaxScaler(feature_range=(1,25))
        if not torch.is_tensor(X):
            if train:
                X = self.sc1.fit_transform(X)
                self.X = torch.from_numpy(X)
            else:
                if scaler_obj is not None:
                    X = scaler_obj.transform(X)
                    self.X = torch.from_numpy(X)
                else:
                    print('include scaler object from training')
                    #X = MinMaxScaler(feature_range=(x_min, x_max)).fit_transform(X)
                    #self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            y = y.to_numpy()
            y = np.reshape(y, (-1,1))
            if train:
                #y = self.sc2.fit_transform(y)
                self.y = torch.from_numpy(y)
            else:
                #y = MinMaxScaler(feature_range=(y_min, y_max)).fit_transform(y)
                self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def return_scaler_obj(self):
        return self.sc1


class KernelRunModel(torch.nn.Module):
    def __init__(self, ip_features, num_hidden, op_features=1):
        super(KernelRunModel, self).__init__()

        self.ip = torch.nn.Linear(ip_features, num_hidden)
        self.hidden2 = torch.nn.Linear(num_hidden, num_hidden*2)
        self.hidden3 = torch.nn.Linear(num_hidden*2, num_hidden)
        #self.hidden4 = torch.nn.Linear(80,num_hidden)
        #self.hidden5 = torch.nn.Linear(40, num_hidden)
        #self.bn1 = nn.BatchNorm1d(num_hidden)
        #self.bn2 = nn.BatchNorm1d(num_hidden*2)
        #self.bn3 = nn.BatchNorm1d(ip_features)
        self.hidden6 = torch.nn.Linear(num_hidden, num_hidden)
        self.hidden7 = torch.nn.Linear(num_hidden, ip_features)
        self.op_run = torch.nn.Linear(ip_features, op_features)
        self.dropout = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.1)
    
    def forward(self, x):
        op = F.relu(self.ip(x))
        x = F.relu(self.hidden2(op))
        op2 = F.relu(self.hidden3(x))
        
        x = self.dropout(op2)
        x = F.relu(self.hidden6(op2))
        
        x = F.relu(self.hidden7(x))
        
        x = F.relu(self.op_run(x))
        return x




dr_columns = ['kernel','Compiler','Cluster','gpu_name'] # using all hardware features except string variables


dataset_root=""
df = pd.read_csv(dataset_root+"matrix_multiplication_spec.csv")


#single = pd.concat([df,df2], axis=0)

single = df.drop(columns=dr_columns)


print(list(single.columns))
print(len(single))
print(len(list(single.columns)))
X = single.iloc[:, 0:-1]
y = single.iloc[:, -1]

train_eval_split=0.8
split_seed=42

#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_eval_split, random_state=split_seed, shuffle=True)

#train_sets = PrepareData(X_train, y_train, train=True)


total_sets = PrepareData(X, y, train=True)
m_scaler = total_sets.return_scaler_obj()
print(len(total_sets))
test_split = 0.2

dataset_size = len(total_sets)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))

train_indices, test_indices = indices[split:], indices[:split]


#val split
val_split=0.1 # 10% of remaining training data
vsplit = int(np.floor(val_split*(1-test_split)*dataset_size))
val_idx, train_idx = indices[split:split+vsplit], indices[split+vsplit:]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetRandomSampler(test_indices)
tr_loader = DataLoader(total_sets, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(total_sets, batch_size=1, sampler=val_sampler)
te_loader = DataLoader(total_sets, batch_size=1, sampler=test_sampler)
print(len(tr_loader), len(val_loader), len(te_loader))




from torch.autograd import Variable

#mod = KernelRunModel(69,138).to(device)
mod = DeepRes(4, Block, 69).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        #torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.00)

mod.apply(init_weights)

#print(mod)

criterion = nn.MSELoss(reduction='mean')
criterion2 = nn.L1Loss(reduction='sum')
num_epochs=20

#opt = torch.optim.Adam(mod.parameters(), lr=5e-3, weight_decay=1e-4)
#opt = torch.optim.Adagrad(mod.parameters(), lr=1e-2, weight_decay=1e-4)
opt = torch.optim.RMSprop(mod.parameters(), lr=5e-3, momentum=0.85, weight_decay=2e-4)
# opt = torch.optim.LBFGS(mod.parameters(), lr=1e-3) ## check closure error

#opt = torch.optim.Adam(mod.parameters(), lr=1e-3, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=0)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.25)
rmse_best=np.inf
mape_best=np.inf
l2l1_best=np.inf
best_model = None
for e in range(num_epochs):
    batch_losses = []
    mod.train()
    for ix, (Xb, yb) in enumerate(tr_loader):
        #opt.zero_grad()
        _X = Variable(Xb).float()
        _y = Variable(yb).float()

        #==========Forward pass===============
        _X = _X.to(device)
        _y = _y.to(device)
        preds = mod(_X)
        loss = criterion(preds, _y)
        loss2 = criterion2(preds, _y)
        total_loss = loss + loss2
        #==========backward pass==============

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        batch_losses.append(total_loss.item())
        #all_losses.append(loss.data[0])
    
    mbl = np.mean(np.sqrt(batch_losses)).round(3)
    
    print("Epoch [{}/{}], Batch loss: {}".format(e, num_epochs, mbl))
    
    mod.eval()
    mape=0.0
    rmse=0.0
    l2l1=0.0
    with torch.no_grad():
        gtr_ = list()
        pred_ = list()
        val_loss = list()
        for index, (xt, yt) in enumerate(val_loader):
            gtr_.append(yt.cpu().data.numpy()[0])

            _xt = Variable(xt).float()
            _yt = Variable(yt).float()

            _xt = _xt.to(device)
            _yt = _yt.to(device)

            predictions = mod(_xt)  ## no need to filter out negative op
            loss1 = criterion(predictions, _yt)
            loss2 = criterion2(predictions, _yt)
            val_loss.append(loss1.item()+loss2.item())

            pred_.append(predictions.cpu().data.numpy()[0])
            #print(predictions, _yt)

        mape = mean_absolute_percentage_error(gtr_, pred_)
        rmse = np.sqrt(mean_squared_error(gtr_, pred_))
        l2l1 = np.mean(np.sqrt(val_loss)).round(3)
        print('Epoch:', e, 'RMSE: ', rmse, ' MAPE:', mape, ' L2+L1 loss:', l2l1)
    
    if mape < mape_best:
        rmse_best=rmse
        mape_best=mape
        l2l1_best=l2l1
        best_model=copy.deepcopy(mod)
    
    lr_scheduler.step()



print('\n\nEvaluating Model.......')
print('Best Model - RMSE:', rmse_best, ' MAPE:', mape_best, ' L2+L1-', l2l1_best)
#Evaluate model on testing data
best_model.eval()
less_5_gt = list()
less_5_pred = list()

with torch.no_grad():
    total_loss = 0
    gt_ = list()
    preds_ = list()
    
    # custom prediction metric
    less_5_pr = 0
    less_5_gt = 0
    less_100_pr = 0
    less_100_gt = 0
    more_100_pr = 0
    more_100_gt = 0
    
    for index, (xt, yt) in enumerate(te_loader):
        gt_.append(yt.cpu().data.numpy()[0])
        gr_truth = yt.cpu().data.numpy()[0]


        _xt = Variable(xt).float()
        _yt = Variable(yt).float()
        
        _xt = _xt.to(device)
        _yt = _yt.to(device)
        
        predictions = F.relu(best_model(_xt))
        loss1 = criterion(predictions, _yt)
        preds_.append(predictions.cpu().data.numpy()[0])
        pr_val = predictions.cpu().data.numpy()[0]
        
        if gr_truth <= 5.0:
            less_5_gt += 1
            if abs(gr_truth - pr_val) <= 2.00:
                less_5_pr += 1
        elif gr_truth <= 100.00:
            less_100_gt += 1
            if abs(gr_truth - pr_val) <= 10.00:
                less_100_pr += 1
        else:
            more_100_gt += 1
            if abs(gr_truth-pr_val) <= 0.1*gr_truth:
                more_100_pr +=1

        
        print(predictions.cpu().data.numpy()[0][0],',', _yt.cpu().data.numpy()[0][0])
        total_loss += loss1
    
    mape = mean_absolute_percentage_error(gt_, preds_)
    rmse = np.sqrt(mean_squared_error(gt_, preds_))
    #print('Test Loss: ', np.mean(np.sqrt(total_loss.item())))
    print('RMSE: ', rmse, ' MAPE:', mape)
    print('5: ground truth total- ', less_5_gt, ' predicted total - ', less_5_pr)
    print('100: ground truth total- ', less_100_gt, ' predicted total - ', less_100_pr)
    print(' more 100: ground truth total - ', more_100_gt, ' predicted total - ', more_100_pr)



print('\n\n\nevaluating custom set to see predictions for all offloading variants of one MM variant')
df2 = pd.read_csv('mm_test_offload.csv')
single2=df2.drop(columns=dr_columns)
X2 = single2.iloc[:, 0:-1]
y2 = single2.iloc[:, -1]
total_sets2 = PrepareData(X2, y2, train=False, scaler_obj=m_scaler)
test_loader_2 = DataLoader(total_sets2, batch_size=1, shuffle=True)


with torch.no_grad():
    gt2_ = list()
    preds2_ = list()

    for index, (xt, yt) in enumerate(test_loader_2):
        gt2_.append(yt.cpu().data.numpy()[0])
        gr_truth = yt.cpu().data.numpy()[0]


        _xt = Variable(xt).float()
        _yt = Variable(yt).float()

        _xt = _xt.to(device)
        _yt = _yt.to(device)

        predictions = F.relu(best_model(_xt))
        loss1 = criterion(predictions, _yt)
        preds2_.append(predictions.cpu().data.numpy()[0])
        pr_val = predictions.cpu().data.numpy()[0]

        print(predictions.cpu().data.numpy()[0][0],',', _yt.cpu().data.numpy()[0][0])
        

    mape = mean_absolute_percentage_error(gt2_, preds2_)
    rmse = np.sqrt(mean_squared_error(gt2_, preds2_))
    print('RMSE: ', rmse, ' MAPE:', mape)

