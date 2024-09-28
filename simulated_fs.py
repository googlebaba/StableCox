from data.selection_bias import gen_selection_bias_data
from algorithm.DWR import DWR
from algorithm.SRDO import SRDO
from model.linear import get_algorithm_class
from metrics import get_metric_class
from utils import setup_seed, get_beta_s, get_expname, calc_var, pretty, get_cov_mask, BV_analysis
from Logger import Logger
from model.STG import STG
from lifelines import LogLogisticAFTFitter, WeibullAFTFitter, LogNormalAFTFitter
from sklearn.metrics import mean_squared_error
import numpy as np
import argparse
import os
import torch
from collections import defaultdict as dd
import pandas as pd
from lifelines import CoxPHFitter
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from sksurv.util import Surv
from lifelines.utils import concordance_index

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch sample reweighting experiments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data generation
    parser.add_argument("--p", type=int, default=10, help="Input dim")
    parser.add_argument("--n", type=int, default=10000, help="Sample size")
    parser.add_argument("--V_ratio", type=float, default=0.5)
    parser.add_argument("--Vb_ratio", type=float, default=0.1)
    parser.add_argument("--true_func", choices=["linear",], default="linear")
    parser.add_argument("--mode", choices=["S_|_V", "S->V", "V->S", "collinearity"], default="S_|_V")
    parser.add_argument("--misspe", choices=["poly", "exp", "None"], default="poly")
    parser.add_argument("--corr_s", type=float, default=0.9)
    parser.add_argument("--corr_v", type=float, default=0.1)
    parser.add_argument("--mms_strength", type=float, default=1.0, help="model misspecifction strength")
    parser.add_argument("--spurious", choices=["nonlinear", "linear"], default="nonlinear")
    parser.add_argument("--r_train", type=float, default=2.5, help="Input dim")
    parser.add_argument("--r_list", type=float, nargs="+", default=[-3, -2, -1.7, -1.5, -1.3, 1.3, 1.5, 1.7, 2, 3])
    parser.add_argument("--noise_variance", type=float, default=0.1)

    # frontend reweighting 
    parser.add_argument("--reweighting", choices=["None", "DWR", "SRDO"], default="DWR")
    parser.add_argument("--decorrelation_type", choices=["global", "group"], default="global")
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--iters_balance", type=int, default=20000)

    parser.add_argument("--topN", type=int, default=5)
    # backend model 
    parser.add_argument("--backend", choices=["OLS", "Lasso", "Ridge", "Weighted_cox", "LogLogistic"], default="Weighted_cox")
    parser.add_argument("--paradigm", choices=["regr", "fs",], default="regr")
    parser.add_argument("--iters_train", type=int, default=1000)
    parser.add_argument("--lam_backend", type=float, default=0.01) # regularizer coefficient
    parser.add_argument("--fs_type", choices=["oracle", "None", "given", "STG"], default="STG")
    parser.add_argument("--mask_given", type=int, nargs="+", default=[1,1,1,1,1,0,0,0,0,0])
    parser.add_argument("--mask_threshold", type=float, default=0.2)
    parser.add_argument("--lam_STG", type=float, default=3)
    parser.add_argument("--sigma_STG", type=float, default=0.1)
    parser.add_argument("--metrics", nargs="+", default=["L1_beta_error", "L2_beta_error"])
    parser.add_argument("--bv_analysis", action="store_true")

    parser.add_argument("--gener_method", choices=["cox_exp", "cox_weibull", "poly", "cox_Gompertz", "exp_T", "log_T"], default="cox_exp")
    # others
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--times", type=int, default=10)
    parser.add_argument("--result_dir", default="results")

    return parser.parse_args()



def generate_indicator(Y, cencored_rate = 0.1):
    n = Y.shape[0]
    num_elements = int(n * cencored_rate)
    indices = np.random.choice(n, size=num_elements, replace=False)
    random_values = np.random.uniform(0, Y[indices])
    Y[indices] = random_values
    indicator = np.ones_like(Y)
    indicator[indices] = 0
    return Y, indicator



def main(args, round, logger):
    setup_seed(args.seed + round)
    p = args.p
    p_v = int(p*args.V_ratio)
    p_s = p-p_v
    n = args.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle_mask = [True,]*p_s + [False,]*p_v

    # generate train data
    X_train, S_train, V_train, fs_train, Y_train = gen_selection_bias_data({**vars(args),**{"r": args.r_train}})

    Y_train, Y_censored = generate_indicator(Y_train, cencored_rate = 0.1)
    print("Y_train", Y_train)
    X_train_pd = pd.DataFrame(np.concatenate((Y_train.reshape((-1, 1)), Y_censored.reshape((-1,1)), X_train), axis=1), columns=["Survival.months", "Survival.status"]+list(range(0, X_train.shape[1])))

    beta_s = get_beta_s(p_s)
    beta_v = np.zeros(p_v)
    beta = np.concatenate([beta_s, beta_v])
    
    linear_var, nonlinear_var, total_var = calc_var(beta_s, S_train, fs_train)
    logger.info("Linear term var: %.3f, Nonlinear term var: %.3f, total var: %.3f" % (linear_var, nonlinear_var, total_var))
    
    # generate test data
    test_data = dict()
    for r_test in args.r_list:
        X_test, S_test, V_test, fs_test, Y_test = gen_selection_bias_data({**vars(args),**{"r": r_test}})
        Y_test, Y_censored = generate_indicator(Y_test, cencored_rate = 0.1)
        X_test_pd = pd.DataFrame(np.concatenate((Y_test.reshape((-1, 1)), Y_censored.reshape((-1,1)), X_test), axis=1), columns=["Survival.months", "Survival.status"]+list(range(0, X_test.shape[1])))
        test_data[r_test] = (X_test, S_test, V_test, fs_test, Y_test, X_test_pd)
 
    p = X_train.shape[1]
    if args.reweighting == "DWR":
        if args.decorrelation_type == "global":
            cov_mask = get_cov_mask(np.zeros(p))
        elif args.decorrelation_type == "group":
            cov_mask = get_cov_mask(np.array(oracle_mask, np.float))
        else:
            raise NotImplementedError            
        W = DWR(X_train, cov_mask=cov_mask, order=args.order, num_steps=args.iters_balance, logger=logger, device=device)
    elif args.reweighting == "SRDO":
        W = SRDO(X_train, p_s, decorrelation_type=args.decorrelation_type, max_iter=args.iters_balance)
    else:
        W = np.ones((n, 1))
    
    #W = np.ones((n, 1))/n
    mean_value = np.mean(W)
    W = W * (1/mean_value)


    results = dict()
    if args.paradigm == "regr":
        mask = [True,]*p
        model_func = get_algorithm_class(args.backend)
        model = model_func(X_train_pd, "Survival.months", "Survival.status", W, 0.00001, **vars(args))
            
    elif args.paradigm == "fs":
        if args.fs_type == "STG":
            stg = STG(p, 1, sigma=args.sigma_STG, lam=args.lam_STG)
            stg.train(X_train, Y_train, W=W, epochs=5000)
            select_ratio = stg.get_ratios().detach().numpy()
            logger.info("Select ratio: " + pretty(select_ratio))
            mask = select_ratio > args.mask_threshold
        elif args.fs_type == "oracle":
            mask = oracle_mask
        elif args.fs_type == "None":
            mask = [True,]*p
        elif args.fs_type == "given":
            mask = np.array(args.mask_given, np.bool)
        else:
            raise NotImplementedError
        if np.array(mask, dtype=np.int64).sum() == 0:
            logger.info("All variables are discarded!")
            assert False
        logger.info("Hard selection: " + str(np.array(mask, dtype=np.int64)))
        model_func = get_algorithm_class(args.backend)
        model = model_func(X_train, Y_train, np.ones((n, 1))/n, **vars(args))
        model.fit(X_train[:, mask], Y_train)
    else:
        raise NotImplementedError
    
    # test 
    summary = model.summary
    fs_sorted_indices = summary['p'].sort_values().head(args.topN).index

 
    X_train_pd = X_train_pd[list(X_train_pd.columns[:2])+list(fs_sorted_indices)]
    #cph = LogLogisticAFTFitter(penalizer=0.00001, fit_intercept=False)
    cph = CoxPHFitter(penalizer=0.0001)
    cph.fit(X_train_pd, duration_col='Survival.months', event_col='Survival.status')

    summary = cph.summary
   
    coef = summary["coef"]


    train_score = cph.score(X_train_pd)
    optimal_p_value = 0

    c_index_dict = dict()

    for r_test, test in test_data.items():
        print("test ratio:", r_test)
        X_test, S_test, V_test, fs_test, Y_test, X_test_pd = test
        X_test_pd = X_test_pd[list(X_test_pd.columns[:2])+list(fs_sorted_indices)]
        c_index = cph.score(X_test_pd, scoring_method='concordance_index')
        c_index_dict[r_test] = c_index

    results["c_index"] = c_index_dict

    return results

if __name__ == "__main__":
    args = get_args()    
    setup_seed(args.seed)
    expname = get_expname(args)
    os.makedirs(os.path.join(args.result_dir, expname), exist_ok=True)
    logger = Logger(args)
    logger.log_args(args)

    p = args.p
    p_v = int(p*args.V_ratio)
    p_s = p-p_v
    beta_s = get_beta_s(p_s)
    beta_v = np.zeros(p_v)
    beta = np.concatenate([beta_s, beta_v])
        
    results_list = dd(list)
    for i in range(args.times):
        logger.info("Round %d" % i)
        results = main(args, i, logger)
        for k, v in results.items():
            results_list[k].append(v)
    

    logger.info("Final Result:")
    for k, v in results_list.items():        
        if k == "c_index":
            RMSE_dict = dict()
            for r_test in args.r_list:
                RMSE = [v[i][r_test] for i in range(args.times)]
                RMSE_dict[r_test] = sum(RMSE)/len(RMSE)
            logger.info("c_index average: %.3f" % (np.mean(list(RMSE_dict.values()))))
            logger.info("c_index std: %.3f" % ((np.std(list(RMSE_dict.values())))))
            logger.info("c_index max: %.3f" % ((np.max(list(RMSE_dict.values())))))
            logger.info("Detailed score:")
            str1 = ""
            for r_test in args.r_list:
                logger.info("%.1f: %.8f" % (r_test, RMSE_dict[r_test]))
                str1 += str(RMSE_dict[r_test]) + "\n"
            print(str1)
            
           
