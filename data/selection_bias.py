import numpy as np
from math import ceil
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import random
import seaborn as sns
import matplotlib.pyplot as plt
from model.MLP import MLP
import torch
import time
from utils import gen_Cov, get_beta_s

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def _gen_data(p=10, r=1.7, n_single=2000, V_ratio=0.5, Vb_ratio=0.1, gener_method='cox_exp', true_func="linear", mlp=None, mode="S_|_V", misspe="poly", mms_strength=1.0, corr_s=0.9, corr_v=0.1, spurious="nonlinear", noise_variance=0.3, device=None, **options):
    '''
    p: dim of input X
    r: bias rate
    n: number of samples
    '''
    # dim of S, V, V_b
    p_v = int(p*V_ratio)
    p_s = p-p_v
    p_b = int(p*Vb_ratio)
    
    # generate covariates    
    Z = np.random.randn(n_single, p)
    S = np.zeros((n_single, p_s))
    V = np.zeros((n_single, p_v))
    if mode == "S_|_V":
        V = np.random.randn(n_single, p_v)
        for i in range(p_s):
            S[:, i] = 0.8*Z[:, i] + 0.2*Z[:, i+1] # hard-coding
    elif mode == "S->V":
        for i in range(p_s):
            S[:, i] = 0.8*Z[:, i] + 0.2*Z[:, i+1]
        for j in range(p_v):
            V[:, j] = 0.8*S[:, j] + 0.2*S[:, (j+1)%(p_s)] + np.random.randn(n_single)
    elif mode == "V->S":
        V = np.random.randn(n_single, p_v)
        for j in range(p_s):
            S[:, j] = 0.2*V[:, j] + 0.8*V[:, (j+1)%(p_v)] + np.random.randn(n_single)
    elif mode == "collinearity":
        Sigma_s = gen_Cov(p_s, corr_s)
        Sigma_v = gen_Cov(p_v, corr_v)
        S = np.random.multivariate_normal([0]*p_s, Sigma_s, n_single)
        V = np.random.multivariate_normal([0]*p_v, Sigma_v, n_single)
    else:
        raise NotImplementedError

    # generate f(S)
    if true_func == "linear":
        beta_s = get_beta_s(p_s)
        beta_s = np.reshape(beta_s, (-1, 1))
        linear_term = np.matmul(S, beta_s)
        if misspe == "poly": # hard-coding: S1·S2·S3
            nonlinear_term = np.reshape(np.prod(S[:,:3], axis=1), (-1, 1))
        elif misspe == "exp": # hard-coding: exp(S1·S2·S3)
            nonlinear_term = np.reshape(np.exp(np.prod(S[:,:3], axis=1)), (-1, 1))
        elif misspe == "None":
            nonlinear_term = 0
        else:
            raise NotImplementedError
        fs = linear_term + mms_strength*nonlinear_term
    elif true_func == "MLP":
        fs = mlp(torch.tensor(S, dtype=torch.float, device=device)).detach().cpu().numpy()
    elif true_func == "poly":
        fs = np.reshape(np.prod(S, axis=1), (-1, 1))
    elif true_func == "exp":
        fs = np.reshape(np.exp(np.prod(S, axis=1)), (-1, 1))
    else:
        raise NotImplementedError

    # generate spurious correlation
    if spurious == "nonlinear":
        D = np.abs(fs-r/abs(r)*V[:,-p_b:]) # dim: (n, p_b), select the last p_b dim of V as V_b
    elif spurious == "linear":
        D = np.abs(linear_term-r/abs(r)*V[:,-p_b:])
    else:
        raise NotImplementedError
    Pr = np.power(abs(r), -5*np.sum(D, axis=1)) # probability of being selected for certain samples
    select = np.random.uniform(size=Pr.shape[0]) < Pr
    # select
    S = S[select, :]
    V = V[select, :]
    X = np.concatenate((S, V), axis=1)
    fs = fs[select, :]

    lambda_param = fs #+ np.random.randn(*fs.shape)*np.sqrt(noise_variance)
    lambda_param = lambda_param.reshape(lambda_param.shape[0])


    U = np.random.uniform(0, 1, size=lambda_param.shape[0])
    #noise = (np.random.rand(*lambda_param.shape)*np.sqrt(noise_variance)).reshape(-1)
    if gener_method == "cox_exp":
        tmp = np.exp(-lambda_param).reshape(-1) 
        noise = (np.random.rand(*lambda_param.shape)*np.sqrt(noise_variance)).reshape(-1)
        Y = 1/0.5*(-np.log(U)*tmp)# + noise
    elif gener_method == "cox_weibull":
        tmp = np.exp(-lambda_param).reshape(-1)        
        Y = np.power(-1/0.5*np.log(U)*tmp, 4)
    elif gener_method == "cox_Gompertz":
        Y = 1/2 * np.log(1-np.log(U)/np.exp(lambda_param))
    elif gener_method == "exp_T":
        noise = np.random.exponential(scale=1, size=lambda_param.shape[0])
        Y = np.exp(lambda_param) + noise
    elif gener_method == "log_T":
        noise = np.random.normal(0, 0.5, size=lambda_param.shape[0])
        Y = np.exp(lambda_param + noise)
    elif gener_method == "poly":
        #lambda_param = lambda_param + abs(min(lambda_param)) + 0.001
        select = lambda_param > 0
        lambda_param = lambda_param[select]
        U = np.random.uniform(0, 1, size=lambda_param.shape[0])
        S = S[select, :]
        V = V[select, :]
        X = np.concatenate((S, V), axis=1)
        Y = -np.log(U)/lambda_param
    return X, S, V, fs, Y

def gen_selection_bias_data(args):
    n_total = args["n"]
    n_cur = 0
    S_list = []
    V_list = []
    fs_list = []
    Y_list = []
    while n_cur < n_total:
        _, S, V, fs, Y = _gen_data(n_single=n_total, **args)
        S_list.append(S)
        V_list.append(V)
        fs_list.append(fs)
        Y_list.append(Y)
        n_cur += Y.shape[0]
    S = np.concatenate(S_list, axis=0)[:n_total]
    V = np.concatenate(V_list, axis=0)[:n_total]
    fs = np.concatenate(fs_list, axis=0)[:n_total]
    Y = np.concatenate(Y_list, axis=0)[:n_total]
    X = np.concatenate((S, V), axis=1)

    return X, S, V, fs, Y


def data_split(X, split_ratio=0.8):
    p_split = int(len(X)*split_ratio)
    return X[:p_split], X[p_split:]



if __name__ == "__main__":
    setup_seed(7)
    X, S, V, fs, Y = gen_selection_bias_data(p=10, r=1.7, n_total=1000, mode="S->V", misspe="poly")
    print(S.shape)
    print(V.shape)
    print(fs.shape)
    print(Y.shape)
    corr = np.corrcoef((np.concatenate((X, fs, Y), axis=1).T)) 
    ax = sns.heatmap(corr,  cmap="YlGnBu")
    plt.savefig("test.png")

    regr = linear_model.LinearRegression()
    X_train, X_test = data_split(X)
    Y_train, Y_test = data_split(Y)
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)
    print(Y_pred.shape)
    print("Coefficients: ", regr.coef_)
    print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
