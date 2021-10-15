import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler as mm_scaler
import numpy as np


## DataSet for Training: meta-train & meta-test
## create separate one for Testing
class CompData(Dataset):
    def __init__(self, X, y, train=True, scaler=True, task_num=3, num_sets=100, meta_train_batch=10, meta_test_batch=10):
        self.task_num = task_num
        self.scaler = scaler
        if train:
            self.meta_train_batch = meta_train_batch
            self.meta_test_batch = meta_test_batch
        else:
            self.test_batch = test_batch

        if scaler:
            X = mm_scaler().fit_transform(X) # will return numpy ndarray

        y = y.to_numpy() # y is pandas dataframe initially

        entire_data = np.column_stack((X,y))


        total_rows_required = num_sets*(meta_train_batch+meta_test_batch)
        # we get shuffled data so directly pick total rows required
        total_train_rows = entire_data[:total_rows_required,:]

        train_rows = total_train_rows[:num_sets*meta_train_batch,:]
        test_rows = total_train_rows[num_sets*meta_train_batch:,:]

        num_features = len(train_rows[0]) #### DEFINE

        ### final np arrays with data and runtimes for num_set rows
        train_rows_data = train_rows[:, :num_features-1]
        train_rows_runtime = train_rows[:,num_features-1:]

        test_rows_data = test_rows[:,:num_features-1]
        test_rows_runtime = test_rows[:,num_features-1:]

        train_rows_data = np.vsplit(train_rows_data, num_sets)
        train_rows_runtime = np.vsplit(train_rows_runtime, num_sets)
        test_rows_data = np.vsplit(test_rows_data, num_sets)
        test_rows_runtime = np.vsplit(test_rows_runtime, num_sets)

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
        temp_store = self.final_sets[index]
        train_row_data = temp_store[0]
        train_row_runtime = temp_store[1]
        test_row_data = temp_store[2]
        test_row_runtime = temp_store[3]

        if np.asarray(train_row_data) is train_row_data:
            train_row_data = np.asarray(train_row_data)
            train_row_runtime = np.asarray(train_row_runtime)
            test_row_data = np.asarray(test_row_data)
            test_row_runtime = np.asarray(test_row_runtime)

        ## convert numpy array to torch tensor     
        train_row_data = torch.from_numpy(train_row_data)
        train_row_runtime = torch.from_numpy(train_row_runtime)
        test_row_data = torch.from_numpy(test_row_data)
        test_row_runtime = torch.from_numpy(test_row_runtime)
        #if not torch.is_tensor(y):
        #    self.train_y = torch.from_numpy(y)


        return train_row_data, train_row_runtime, test_row_data, test_row_runtime
