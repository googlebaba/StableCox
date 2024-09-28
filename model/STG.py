import torch
import torch.nn as nn
import math
import torch.optim as optim
import numpy as np
from utils import pretty
from torch.autograd import grad

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.weight_init()

    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

# Feature selection part
class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.00 * torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma
        self.input_dim = input_dim

    def renew(self):
        self.mu = torch.nn.Parameter(0.00 * torch.randn(self.input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class STG:
    def __init__(self, input_dim, output_dim, sigma=1.0, lam=0.1, hard_sum = 1.0):
        self.backmodel = LinearRegression(input_dim, output_dim)
        self.loss = nn.MSELoss()
        self.featureSelector = FeatureSelector(input_dim, sigma)
        self.reg = self.featureSelector.regularizer
        self.lam = lam
        self.mu = self.featureSelector.mu
        self.sigma = self.featureSelector.sigma
        #self.alpha = alpha
        self.optimizer = optim.Adam([{'params': self.backmodel.parameters(), 'lr': 1e-3},
                                     {'params': self.mu, 'lr': 3e-4}])
        self.hard_sum = hard_sum
        self.input_dim = input_dim

    def renew(self):
        self.featureSelector.renew()
        self.mu = self.featureSelector.mu
        self.backmodel.weight_init()
        self.optimizer = optim.Adam([{'params': self.backmodel.parameters(), 'lr': 1e-3},
                                     {'params': self.mu, 'lr': 3e-4}])


    def pretrain(self, X, Y, pretrain_epoch=100):
        pre_optimizer = optim.Adam([{'params': self.backmodel.parameters(), 'lr': 1e-3}])
        for i in range(pretrain_epoch):
            self.optimizer.zero_grad()
            pred = self.backmodel(X)
            loss = self.loss(pred, Y.reshape(pred.shape))
            loss.backward()
            pre_optimizer.step()


    def get_gates(self):
        return self.mu+0.5

    def get_ratios(self):
        return self.reg((self.mu + 0.5) / self.sigma)

    def get_params(self):
        return self.backmodel.linear.weight

    def train(self, X, Y, W, epochs):
        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.float)
        W = torch.tensor(W, dtype=torch.float)
        self.renew()
        self.pretrain(X, Y, 3000)
        for epoch in range(1,epochs+1):
            self.optimizer.zero_grad()
            Y_pred = self.backmodel(self.featureSelector(X))
            loss_erm =  torch.matmul(W.T, (Y_pred-Y)**2)
            reg = torch.sum(self.reg((self.mu + 0.5) / self.sigma))
            loss = loss_erm + self.lam * reg**2
            loss.backward()
            self.optimizer.step()
            if epoch % 1000 ==0:
                print("Epoch %d | Loss = %.4f | Ratio =  %s | Theta = %s" %
                      (epoch, loss, pretty(self.get_ratios()), pretty(self.get_params())))
