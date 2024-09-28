import torch
from torch import optim
import numpy as np
from utils import weighted_cov_torch

def decorr_loss(X, weight, cov_mask=None, order=1):
    n = X.shape[0]
    p = X.shape[1]
    balance_loss = 0.0 
    for a in range(1, order+1):
        for b in range(a, order+1):
            if a != b:
                cov_mat = weighted_cov_torch(X**a, X**b, W=weight**2/n)
            else:
                cov_mat = weighted_cov_torch(X**a, W=weight**2/n)
            cov_mat = cov_mat**2
            cov_mat = cov_mat * cov_mask
            balance_loss += torch.sum(torch.sqrt(torch.sum(cov_mat, dim=1)-torch.diag(cov_mat) +  1e-10))

    loss_weight_sum = (torch.sum(weight * weight) - n) ** 2
    loss_weight_l2 = torch.sum((weight * weight) ** 2)
    loss = 2000.0 / p * balance_loss + 100 * loss_weight_sum + 0.0005 * loss_weight_l2 # hard coding
    return loss, balance_loss, loss_weight_sum, loss_weight_l2

def DWR(X, cov_mask=None, order=1, num_steps = 5000, lr = 0.01, tol=1e-8, loss_lb=0.001, iter_print=500, logger=None, device=None):
    X = torch.tensor(X, dtype=torch.float, device=device)
    n, p = X.shape    
    
    if cov_mask is None:
        cov_mask = torch.ones((p, p), device=device)
    else:
        cov_mask = torch.tensor(cov_mask, dtype=torch.float, device=device)

    weight = torch.ones(n, 1, device=device)
    weight = weight.to(device)
    weight.requires_grad = True
    optimizer = optim.Adam([weight,], lr = lr)
    
    loss_prev = 0.0
    for i in range(num_steps):
        optimizer.zero_grad()
        loss, balance_loss, loss_s, loss_2 = decorr_loss(X, weight, cov_mask, order=order)
        loss.backward()
        optimizer.step()
        if abs(loss-loss_prev) <= tol or balance_loss <= loss_lb:
            break
        if (i+1) % iter_print == 0:
            logger.debug('iter %d: decorrelate loss %.6f balance loss %.6f loss_s %.6f  loss_l2 %.6f' % (i+1, loss, balance_loss, loss_s, loss_2))
    weight = (weight**2).cpu().detach().numpy()
    weight /= np.sum(weight) # normalize: weights sum up to 1
    return weight
    
    
