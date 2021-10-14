import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler as std_scaler

class CompData(Dataset):
    def __init__(self, X, y, train=True, scaler=True, meta_train_batch=32, meta_test_batch=32, test_batch=32):
        
                
        self.scaler = scaler
        if train:
            self.meta_train_batch = meta_train_batch
            self.meta_test_batch = meta_test_batch
        else:
            self.test_batch = test_batch
        
        if scaler:
            self.X = std_scaler().fit_transform(X)
        
        ## Add followin to __getitem__
        #if not torch.is_tensor(train_x):
        #    self.train_x = torch.from_numpy(train_x)
        #if not torch.is_tensor(train_y):
        #    self.train_y = torch.from_numpy(train_y)




    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):
        ### create model and check input sizes and then create matrices with that dimension [meta_train_batch, shape of 1 data row]



