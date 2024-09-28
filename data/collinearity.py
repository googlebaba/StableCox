import numpy as np
from math import ceil

def _gen_Cov(rho = 0.8, p = 10, s = 2):
    Sigma = np.zeros((p, p))
    for i in range(p//s):
        Sigma[i*s:(i+1)*s, i*s:(i+1)*s] = rho
    for i in range(p):
        Sigma[i, i] = 1
    return Sigma
    
def gen_collinearity_data(rho = 0.8, p = 10, s = 2, n = 2000): # s是每个group的变量个数
    beta_base = [1/5, -2/5, 3/5, -4/5, 1, -1/5, 2/5, -3/5, 4/5, -1,] # hard-coded coefficients
    beta = beta_base * (ceil(p/len(beta_base)))
    beta = np.reshape(beta[:p], (-1, 1))
    Sigma = _gen_Cov(rho, p, s)
    X = np.random.multivariate_normal([0]*p, Sigma, n)
    samplecov = np.cov(X.T)
    e_vals,e_vecs = np.linalg.eig(samplecov)
    v = e_vecs[:, np.argmin(e_vals)].reshape((p, 1))
    bx = np.dot(X, v)
    fs = np.dot(X, beta) + np.reshape(np.prod(X[:,:3], axis=1), (-1, 1))
    #fs = np.dot(X, beta) + bx 
    Y = fs + np.random.randn(n, 1)
    return X, fs, Y
    