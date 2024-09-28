import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.modules.loss import MSELoss

class MLP(nn.Module):
    def __init__(self, input_units=10, hidden_units=[100, 100], output_units=1, initialization="uniform", bias=False, mask=None, device=None):
        super().__init__()
        if mask is None:
            self.mask = torch.ones(input_units, device=device)
        else:
            self.mask = torch.tensor(mask, dtype=torch.float, device=device)
        self.device = device
        #assert len(hidden_units) > 0
        units = [input_units, ] + hidden_units
        layerdict = OrderedDict()
        for i in range(len(units)-1):
            layerdict["layer%d"%i] = nn.Linear(units[i], units[i+1], bias=bias)
            layerdict["relu%d"%i] = nn.ReLU()
        self.features = nn.Sequential(layerdict)
        self.classifier = nn.Linear(units[-1], output_units, bias=bias)
        if initialization == "uniform":        
            def weights_init(m):
                if isinstance(m, nn.Linear):
                    m.weight.data.uniform_(-1, 1)
                    if bias:
                        m.bias.data.uniform_(-1, 1)
            self.apply(weights_init)
            
    def set_mask(self, mask):
        assert not (mask is None)
        self.mask = torch.tensor(mask, dtype=torch.float, device=self.device)        

    def forward(self, X):
        X = X*self.mask
        X = self.features(X)
        return self.classifier(X)

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float, device=self.device)
        X = X*self.mask
        self.eval()
        Y_pred = self.forward(X)
        self.train()
        return Y_pred.cpu().detach().numpy()


def train(model, X, Y, W=None, lr=0.05, num_iters = 1000, tol=1e-13, logger=None, device=None):
    model.to(device)
    X = torch.tensor(X, dtype=torch.float, device=device)
    Y = torch.tensor(Y, dtype=torch.float, device=device)
    W = torch.tensor(W, dtype=torch.float, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    
    loss_prev = 0.0
    for ite in range(num_iters):
        optimizer.zero_grad()
        Y_pred = model(X)
        loss =  torch.matmul(W.T, (Y_pred-Y)**2)
        loss.backward()
        optimizer.step()
        if ite == 0 or (ite+1) % 1000 == 0:
            logger.debug("%d/%d: loss %.4f" % (ite, num_iters, loss.item()))
        if abs(loss-loss_prev) <= tol:
            break
        loss_prev = loss.data




if __name__ == "__main__":
    
    model = MLP(hidden_units=[])
    print(next(model.parameters()).device)
