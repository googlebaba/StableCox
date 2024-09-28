import numpy as np

def get_metric_class(metric_name):
    if metric_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(metric_name))
    return globals()[metric_name]

def L1_beta_error(beta_hat, beta):
    return np.sum(np.abs(beta_hat-beta))    

def L2_beta_error(beta_hat, beta):
    return np.sum((beta_hat-beta)**2)