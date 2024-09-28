from random import sample
from sklearn import linear_model
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines import LogLogisticAFTFitter, WeibullAFTFitter, LogNormalAFTFitter
def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

def OLS(X, Y, W, **options):
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, Y, sample_weight=W.reshape(-1))
    return model

def Lasso(X, Y, W, lam_backend=0.01, iters_train=1000, **options):
    model = linear_model.Lasso(alpha=lam_backend, fit_intercept=False, max_iter=iters_train)
    model.fit(X, Y, sample_weight=W.reshape(-1))
    return model

def Ridge(X, Y, W, lam_backend=0.01, iters_train=1000, **options):
    model = linear_model.Ridge(alpha=lam_backend, fit_intercept=False, max_iter=iters_train)
    model.fit(X, Y, sample_weight=W.reshape(-1))
    return model


def LogLogistic(X, duration_col, event_col, W, pen, **options):
    tmp = X[duration_col]
    tmp[tmp==0] = 0.0001
    llf = LogLogisticAFTFitter(penalizer=pen, fit_intercept=False).fit(X, duration_col=duration_col, event_col=event_col)
    return llf

def Weibull(X, duration_col, event_col, W, pen, **options):
    tmp = X[duration_col]
    tmp[tmp==0] = 0.0001
    waf = WeibullAFTFitter(penalizer=pen, fit_intercept=False).fit(X, duration_col=duration_col, event_col=event_col)
    return waf
def LogNormal(X, duration_col, event_col, W, pen, **options):
    tmp = X[duration_col]
    tmp[tmp==0] = 0.0001
    llf = LogNormalAFTFitter(penalizer=pen, fit_intercept=False).fit(X, duration_col=duration_col, event_col=event_col)
    return llf



def Weighted_cox(X, duration_col, event_col, W, pen, **options):
    columns = X.columns
    all_X = np.concatenate((X, W), axis=1)

    all_X = pd.DataFrame(all_X, columns=list(columns)+["Weights"])
    cph = CoxPHFitter(penalizer=pen)
    cph.fit(all_X, duration_col=duration_col, event_col=event_col, weights_col="Weights")

    return cph
