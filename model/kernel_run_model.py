import torch
import torch.nn
import torch.nn.functional as F

class KernelRunModel(torch.nn.Module):
    def __init__(self, ip_features, num_hidden, op_features=1):
        super(KernelRunModel, self).__init__()

        self.hidden1 = torch.nn.Linear(ip_features, num_hidden)
        self.hidden2 = torch.nn.Linear(num_hidden, num_hidden*2)
        self.hidden3 = torch.nn.Linear(num_hidden*2, num_hidden)
        self.hidden4 = torch.nn.Linear(num_hidden, num_hidden)
        self.op_run = torch.nn.Linear(num_hidden, op_features)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        op = F.relu(self.hidden1(x))
        residual = op
        x = F.relu(self.hidden2(op))
        x = self.dropout(x)
        x = F.relu(self.hidden3(x))
        x += residual
        x = F.relu(self.hidden4(x))
        x = self.dropout(x)
        x = F.sigmoid(self.op_run(x))
        return x
