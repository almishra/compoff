import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from dataload import CompData
from kernel_run_model import OffloadModel

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
parser.add_argument('--inner_epochs', type=int, default=1,\
        help=" [option] number of epochs for inner loop optimization per task model")
parser.add_argument('--epochs', type=int, default=20,\
        help=" [options] number of epochs")
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

update_factor = 0.01 # arg parser argument
update_steps = 2 # arg parser argument

def train_loop(model, x_train, y_train, x_test, y_test):
    original_model_copy = copy.deepcopy(model)
    loss_tasks = 0
    losses_q = [0 for _ in range(update_steps)]
    for k in range(task_num):
        # print(k)
        temp_weights=[w.clone() for w in list(original_model_copy.parameters())]
        
        outputs = original_model_copy.var_forward(x_train[k], temp_weights)
        loss = criterion(outputs, y_train[k])
        
        grad = torch.autograd.grad(loss, temp_weights)
        # temporary update weights 
        fast_weights = [w - update_factor*g for w,g in zip(temp_weights, grad)]
        
        ## run updated weights on meta-test batch
        with torch.no_grad():
            new_outputs = original_model_copy.var_forward(x_test[k], fast_weights)
            new_loss = criterion(new_outputs, y_test[k])
            losses_q[0] += new_loss
            #print('First_update_loss:',new_loss)
        
        for j in range(update_steps):
            ops = original_model_copy.var_forward(x_train[k], fast_weights)
            loss = criterion(ops, y_train[k])
            #print(j, 'step loss for task', k, ' before update:', loss)
            grads = torch.autograd.grad(loss, fast_weights)
            
            fast_weights = [w - update_factor*g for w,g in zip(fast_weights, grads)]
            
            new_ops = original_model_copy.var_forward(x_test[k], fast_weights)
            new_loss = criterion(new_ops, y_test[k])
            #print(j,' step loss for task',k,' after update:', new_loss)
            losses_q[j] += new_loss
        
        
    final_loss = losses_q[-1] / task_num
    ## return loss where grad is calculated (cannot be wrapped by torch.no_grad() context manager)
    return final_loss


def train():
    outer_epochs = 10
    task_num = 3
    global_model = OffloadModel(53,106)
    meta_optim = torch.optim.Adam(global_model.parameters(), lr=1e-4)
    #### add some lr decay: cosine or step or lambda
    global_model = global_model.to(device)
    criterion = nn.MSELoss()

    for idx in range(outer_epochs):
        train_set_loader = DataLoader(train_sets, batch_size=task_num, drop_last=True)
        outer_loss = 0
        for i, (x_train, y_train, x_test, y_test) in enumerate(train_set_loader):
            # print((x_train[0]))
            task_num_, set_size, cols = x_train.shape #<--verify
            #print(task_num_, set_size, cols)
            x_train, y_train, x_test, y_test = x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)

            # train returns loss
            total_set_loss = train_loop(global_model, x_train, y_train, x_test, y_test) #<-- returns loss

            meta_optim.zero_grad()
            total_set_loss.backward()
            meta_optim.step()
            outer_loss += total_set_loss.item()
            #print('Set loss: ',total_set_loss.item())
            ### do some validation or call the actual test function? (diff data set & loader)
            
        print('Outer loss:', np.sqrt(outer_loss))


def test():






def main():
    args = parser.parse_args()
    if not args.dataset_root:
        sys.exit("Enter path to dataset root directory")

    print('<script option> ',args)
    worker(args)


def worker(args):
    
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
    
    train_sets = CompData(X_train,y_train, train=True, task_num=args.task_num, num_sets=1000, meta_train_batch=args.batch_tr, meta_test_batch=args.batch_te)
    train_loader = DataLoader(train_sets, batch_size=args.task_num, drop_last=True) # we want equal number of sets i.e., divisible by task_num so drop last if not exactly divisible
    

    model = OffloadModel(53,106)
    model = model.to(device)
    train()
    # 1. write methods for inner loop? or add it to the train function ----> separate + done

    #test_sets = CompData(X_test, y_test, train=False, test_batch=32)
    
    # 2. create dataset class for testing/validation


if __name__ == '__main__':
    main()

