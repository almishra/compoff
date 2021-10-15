import torch
from sklearn.preprocessing import MinMaxScaler as mm_scaler
from torch.utils.data import Dataset
import numpy as np


class CompData(Dataset):
    def __init__(self, X, y, train=True, scaler=True, task_num=3, num_sets=100. meta_train_batch=10, meta_test_batch=10, test_batch=32):
        self.task_num = task_num       
        self.scaler = scaler
        if train:
            self.meta_train_batch = meta_train_batch
            self.meta_test_batch = meta_test_batch
        else:
            self.test_batch = test_batch
        
        if scaler:
            X = mm_scaler().fit_transform(X)
        
        ### create sets. support_x has 10000 sets of 5/ 25 images each. Total ~1000 sets (for 1000 iterations)
        ### create set with 32 rows (#meta_train_number) and 32 rows (#meta_test_number) and append to train and test lists
        #convert pandas dataframe to numpy array
        X = X.to_numpy()
        y = y.to_numpy()
        
        entire_data = np.hstack((X,y))
        
        #num_rows = len(X) # for index sampling
        #self.selected_sample_indices = np.random.choice(num_rows,\
        #                                           size=task_num*(meta_train_batch+meta_test_batch),\
        #                                           replace=False)
        
        total_rows_required = num_sets*(meta_train_batch+meta_test_batch)
        # we get shuffled data so directly pick total rows required
        total_train_rows = entire_data[:total_rows_required]
        
        train_rows = total_train_rows[:num_sets*meta_train_batch]
        test_rows = total_train_rows[num_threats*meta_train_batch:]
        
        train_rows = np.hsplit(train_rows, num_sets)
        test_rows = np.hsplit(test_rows, num_sets)
        
        num_features = len(train_rows[0])#### DEFINE
        
        ### final np arrays with data and runtimes for num_set rows 
        train_rows_data = train_rows[:, :num_features-1]
        train_rows_runtime = train_rows[:,num_features-1:]
        
        test_rows_data = test_rows[:,:num_features-1]
        test_rows_runtime = test_rows[:,num_features-1:]
        #create sets here:
        final_sets = [] ## list of list. each list row will have train_rows_data/runtime, test_data/runtime
        for i in range(num_sets):
            temp = [train_rows_data[i]]+[train_rows_runtime[i]]+[test_rows_data[i]]+[test_rows_runtime[i]]
            final_sets.append(temp)
        self.final_sets = final_sets

    def __len__(self):
        return len(self.final_sets)

    def __getitem__(self,index):
                
        #zip sample without replacement from X
        
        train_row_data = self.final_sets[index,0]
        train_row_runtime = self.final_sets[index,1]
        test_row_data = self.final_sets[index,2]
        test_row_runtime = self.final_set[index,3]
        
        if np.asarray(train_row_data) is train_row_data:
            train_row_data = np.asarray(train_row_data)
            train_row_runtime = np.asarray(train_row_runtime)
            test_row_data = np.asarray(test_row_data)
            test_row_runtime = np.asarray(test_row_runtime)
        
        ## convert numpy array to torch tensor
        #if not torch.is_tensor(X):
        train_row_data = torch.from_numpy(train_row_data)
        train_row_runtime = torch.from_numpy(train_row_runtime)
        test_row_data = torch.from_numpy(test_row_data)
        test_row_runtime = torch.from_numpy(test_row_runtime)
        #if not torch.is_tensor(y):
        #    self.train_y = torch.from_numpy(y)               
        
        
        return train_row_data, train_row_runtime, test_row_data, test_row_runtime
