import torch
import torch.nn as nn
import torch.nn.functional as F

class OffloadModel(torch.nn.Module):
    def __init__(self, ip_features, num_hidden, op_features=1):
        super(OffloadModel, self).__init__()

        self.mod1 = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(ip_features, num_hidden)),
            ('relu1', nn.ReLU())
        ]))
        self.mod2 = nn.Sequential(OrderedDict([
            ('lin2', nn.Linear(num_hidden,num_hidden*2)),
            ('relu2', nn.ReLU()),
            ('drop1', nn.Dropout(p=0.25)),
            ('lin3', nn.Linear(num_hidden*2, num_hidden)),
            ('relu3', nn.ReLU())

        ]))
        self.mod3 = nn.Sequential(OrderedDict([
            ('lin4', nn.Linear(num_hidden, num_hidden)),
            ('relu4', nn.ReLU()),
            ('drop2', nn.Dropout(p=0.25)),
            ('lin5', nn.Linear(num_hidden,op_features)),
            ('sig1', nn.Sigmoid())
        ]))

    def forward(self, x):
        op = self.mod1(x)
        x = self.mod2(op)
        x += op
        x = self.mod3(x)
        return x

    def var_forward(self, x, weights):
        op = F.relu(F.linear(x, weights[0], weights[1]))
        x = F.relu(F.linear(op, weights[2], weights[3]))
        x = F.dropout(p=0.25)
        x = F.relu(F.linear(x, weights[4], weights[5]))
        x += op
        x = F.relu(F.linear(x, weights[6], weights[7]))
        x = F.dropout(p=0.25)
        x = F.sigmoid(F.linear(x, weights[8], weights[9]))
        return x
