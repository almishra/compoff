import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from dataload import CompData

parser = argparse.ArgumentParser()
parser.add_argument('--batch_tr', type=int, default=32,\
        help=" [option] meta train batch size for each task model")
parser.add_argument('--batch_te', type=int, default=32,\
        help=" [option] meta test batch size for each task model")
parser.add_argument('--train_eval_split', type=float, default=0.6,\
        help=" [option] split data into train and test sets")
parser.add_argument('--task_num', type=int, default=3,\
        help=" [option] number of task models")
parser.add_argument('--lr', type=float, dafault=1e-3,\
        help=" [option] outer learning rate")
parser.add_argument('--decay', type=float, default=0.0,\
        help=" [option] L2 penalty for outer optimization algorithm")
#### Add support for learning rate decay
#parser.add_argument('--lr_decay', type=float, default=0.95,\
#        help=" [option] learning rate decay multiplicative factor" )

####  Update this to application to test on (generalization setting)
parser.add_argument('--app', type=str, default="matrix_multiplication",\
        help=" [option] choose application to train & test")
parser.add_argument('--dataset_root', type=str, \
        help=" [option] path to dataset root directory")
parser.add_argument('--split_seed', type=int, default=43,\
        help=" [option] train-test split seed (can be used fpr reproducibility)")
parser.add_argument('--shuffle', action='store_true',\
        help=" [option] shuffle dataset")


def train():




def test():






def main():
    args = parser.parse_args()
    if not args.dataset_root:
        sys.exit("Enter path to dataset root directory")

    print('<script option> ',args)
    worker(args)


def worker():
    
    dr_columns = ['kernel','Compiler','Cluster','gpu_name','outer','inner','var_decl','ref_expr','int_literal','float_literal','mem_to',\
                'mem_from','add_sub_int','add_sub_double','mul_int','mul_double','div_int','div_double','assign_int','assign_double']
    
    if args.dataset_root[-1] != "/":
        args.dataset_root += "/"

    if args.app == "matrix_multiplication":
        try:
            df = pd.read_csv(args.dataset_root+"matrix_multiplication.csv")   
            df = df.drop(columns=dr_columns)
        except:
            print('An error occured')
        finally:
            sys.exit("please check the dataset path and file names")
    
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.train_test_split, random_state=args.split_seed, shuffle=args.shuffle)
    
    train_sets = CompData(X_train,y_train, train=True, meta_train_batch=args.batch_tr, meta_test_batch=args.batch_te)
    test_sets = CompData(X_test, y_test, train=False, test_batch=32)



if __name__ == '__main__':
    main()

