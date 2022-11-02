import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import math
#import sys
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from torchsummary import summary
import numpy as np
import copy
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from deepres import DeepRes, Block
#from torchlars import LARS
#from pytorch_optimizer import LARS
import argparse
from pickle import load 


parser = argparse.ArgumentParser()

parser.add_argument('--bracket', default="A", choices=["A", "B", "C", "neither"], \
        help='choose from classes A, B, C or neither')

args = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class PrepareData(Dataset):
    def __init__(self, X, y, train=True, scaler_obj=None):
        #self.sc1 = MinMaxScaler(feature_range=(0,5))
        
        # figure out log scaling for some of the large valued columns
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
            # keep this line
            y = np.true_divide(y, 1e6)
            y = np.reshape(y, (-1,1))
            if train:
                #y = self.sc2.fit_transform(y)
                print(type(y.dtype), y.dtype)
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





dr_columns = ['kernel'] 


dataset_root=""


from torch.autograd import Variable

#mod = KernelRunModel(69,138).to(device)
mod = DeepRes(4, Block, 50).to(device)

# load weights to the model object
mod.load_state_dict(torch.load('trained_model.pt'))

#load scaler object
m_scaler = load(open('std_scaler.pkl', 'rb'))

## Upadte data block when alok uploads qcd data

print('\n\n\nevaluating data from wilson d-slash kernel')
#tdf = pd.read_csv(dataset_root+"wilson_exxact_gpu.csv")
#tdf2 = pd.read_csv(dataset_root+"wilson_ookami_gpu.csv")
#tdf3 = pd.read_csv(dataset_root+"wilson_seawulf_gpu.csv")
#tdf4 = pd.read_csv(dataset_root+"wilson_summit_gpu.csv")

#tdf = pd.read_csv(dataset_root+"intel.csv")
tdf2 = pd.read_csv(dataset_root+"k80.csv")
tdf3 = pd.read_csv(dataset_root+"rtx.csv")
#tdf4 = pd.read_csv(dataset_root+"summit_llvm.csv")
#df5 = pd.read_csv(dataset_root+"summit_gcc.csv")


single2 = pd.concat([tdf2,tdf3], axis=0)
single2 = single2.drop(columns=dr_columns)

single2["memTo"].replace({0:1}, inplace=True)
single2["memFrom"].replace({0:1}, inplace=True)
single2["memAlloc"].replace({0:1}, inplace=True)
single2["memDelete"].replace({0:1}, inplace=True)

#applying log function to 13 columns 

updated_single2 = single2.apply(lambda x: np.log10(x) if x.name == "varDecl" or x.name == "refExpr" or x.name == "intLit" or \
        x.name == "memTo" or x.name == "memFrom" or x.name == "memAlloc" or x.name == "memDelete" or x.name == "addSubInt" or \
        x.name == "addSubFloat" or x.name == "mulFloat" or x.name == "logicalInt" or x.name == "remInt" or x.name == "assFloat" else x)

#df2 = pd.read_csv('Wilson.csv')
#single2=df2.drop(columns=dr_columns)
X2 = updated_single2.iloc[:, 0:-1]
y2 = updated_single2.iloc[:, -1]
total_sets2 = PrepareData(X2, y2, train=False, scaler_obj=m_scaler)
test_loader_2 = DataLoader(total_sets2, batch_size=1, shuffle=True)

mod.eval()

criterion = nn.MSELoss(reduction='mean')
criterion2 = nn.L1Loss(reduction='mean')
criterion3 = nn.L1Loss(reduction='sum')

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

        predictions = F.relu(mod(_xt))
        loss1 = criterion(predictions, _yt)
        preds2_.append(predictions.cpu().data.numpy()[0])
        pr_val = predictions.cpu().data.numpy()[0]

        print(predictions.cpu().data.numpy()[0][0],',', _yt.cpu().data.numpy()[0][0])
        

    mape = mean_absolute_percentage_error(gt2_, preds2_)
    rmse = np.sqrt(mean_squared_error(gt2_, preds2_))
    print('RMSE: ', rmse, ' MAPE:', mape)

