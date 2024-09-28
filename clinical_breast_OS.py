from data.selection_bias import gen_selection_bias_data
from algorithm.DWR import DWR
from algorithm.SRDO import SRDO
from model.linear import get_algorithm_class
from metrics import get_metric_class
from utils import setup_seed, get_beta_s, get_expname, calc_var, pretty, get_cov_mask, BV_analysis
from Logger import Logger
from model.STG import STG
from sksurv.metrics import brier_score, cumulative_dynamic_auc
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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
duration_col = 'Survival.months'
event_col = 'Survival.status'
def get_args():
    parser = argparse.ArgumentParser(description="Script to launch sample reweighting experiments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # data generation
    parser.add_argument("--p", type=int, default=10, help="Input dim")
    parser.add_argument("--n", type=int, default=2000, help="Sample size")
    parser.add_argument("--V_ratio", type=float, default=0.5)
    parser.add_argument("--Vb_ratio", type=float, default=0.1)
    parser.add_argument("--true_func", choices=["linear",], default="linear")
    parser.add_argument("--mode", choices=["S_|_V", "S->V", "V->S", "collinearity"], default="collinearity")
    parser.add_argument("--misspe", choices=["poly", "exp", "None"], default="poly")
    parser.add_argument("--corr_s", type=float, default=0.9)
    parser.add_argument("--corr_v", type=float, default=0.1)
    parser.add_argument("--mms_strength", type=float, default=1.0, help="model misspecifction strength")
    parser.add_argument("--spurious", choices=["nonlinear", "linear"], default="nonlinear")
    parser.add_argument("--r_train", type=float, default=2.5, help="Input dim")
    parser.add_argument("--r_list", type=float, nargs="+", default=[-3, -2, -1.7, -1.5, -1.3, 1.3, 1.5, 1.7, 2, 3])
    parser.add_argument("--noise_variance", type=float, default=0.3)

    # frontend reweighting 
    parser.add_argument("--reweighting", choices=["None", "DWR", "SRDO"], default="DWR")
    parser.add_argument("--decorrelation_type", choices=["global", "group"], default="global")
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--iters_balance", type=int, default=2000)

    parser.add_argument("--topN", type=int, default=5)
    # backend model 
    parser.add_argument("--backend", choices=["OLS", "Lasso", "Ridge", "Weighted_cox", "LogLogistic", "Weibull", "LogNormal"], default="Weighted_cox")
    parser.add_argument("--paradigm", choices=["regr", "fs",], default="regr")
    parser.add_argument("--iters_train", type=int, default=5000)
    parser.add_argument("--lam_backend", type=float, default=0.01) # regularizer coefficient
    parser.add_argument("--fs_type", choices=["oracle", "None", "given", "STG"], default="STG")
    parser.add_argument("--mask_given", type=int, nargs="+", default=[1,1,1,1,1,0,0,0,0,0])
    parser.add_argument("--mask_threshold", type=float, default=0.2)
    parser.add_argument("--lam_STG", type=float, default=3)
    parser.add_argument("--sigma_STG", type=float, default=0.1)
    parser.add_argument("--metrics", nargs="+", default=["L1_beta_error", "L2_beta_error"])
    parser.add_argument("--bv_analysis", action="store_true")

    # others
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--result_dir", default="results")

    return parser.parse_args()



def main(args, round, logger):
    setup_seed(args.seed + round)
    p = args.p
    p_v = int(p*args.V_ratio)
    p_s = p-p_v
    n = args.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate train data
    
    training_pd_data = pd.read_csv('./clinical_data/breast_cancer/breast_train_survival.csv', index_col=0)
    test0_pd_data = pd.read_csv('./clinical_data/breast_cancer/breast_test0_survival.csv', index_col=0)
    test1_pd_data = pd.read_csv('./clinical_data/breast_cancer/breast_test1_survival.csv', index_col=0)

    test2_pd_data = pd.read_csv('./clinical_data/breast_cancer/breast_test2_survival.csv', index_col=0)

    training_pd_data = training_pd_data.drop(['Recurr.months', 'Recurr.status', 'Cohort'], axis=1)

    test0_pd_data = test0_pd_data.drop(['Recurr.months', 'Recurr.status', 'Cohort'], axis=1)
    test1_pd_data = test1_pd_data.drop(['Recurr.months', 'Recurr.status', 'Cohort'], axis=1)

    test2_pd_data = test2_pd_data.drop(['Recurr.months', 'Recurr.status', 'Cohort'], axis=1)



    X_train_np = np.array(training_pd_data.iloc[:, 2:])


    p = X_train_np.shape[1]
    n= X_train_np.shape[0]
    print("dim", p)
    if args.reweighting == "DWR":
        if args.decorrelation_type == "global":
            cov_mask = get_cov_mask(np.zeros(p))
        elif args.decorrelation_type == "group":
            cov_mask = get_cov_mask(np.array(oracle_mask, np.float))
        else:
            raise NotImplementedError            
        W = DWR(X_train_np, cov_mask=cov_mask, order=args.order, num_steps=args.iters_balance, logger=logger, device=device)
    elif args.reweighting == "SRDO":
        W = SRDO(X_train_np, p_s, hidden_layer_sizes = (69, 15), decorrelation_type=args.decorrelation_type, max_iter=args.iters_balance)
    else:
        W = np.ones((n, 1))/n
        
    mean_value = np.mean(W)
    W = W * (1/mean_value)
    
    W = np.clip(W, 0.02, 2)
    columns = training_pd_data.columns
    X_train_pd = training_pd_data
    all_X = np.concatenate((X_train_pd, W), axis=1)

    X_train_test = pd.DataFrame(all_X, columns=list(columns)+["Weights"])
    test0_W = np.ones((test0_pd_data.shape[0], 1))
    
    X_test0_pd = np.concatenate((test0_pd_data, test0_W), axis=1)

    X_test0_pd = pd.DataFrame(X_test0_pd, columns=list(columns)+["Weights"])

    test1_W = np.ones((test1_pd_data.shape[0], 1))
    
    X_test1_pd = np.concatenate((test1_pd_data, test1_W), axis=1)

    X_test1_pd = pd.DataFrame(test1_pd_data, columns=list(columns)+["Weights"])

    test2_W = np.ones((test2_pd_data.shape[0], 1))
    
    X_test2_pd = np.concatenate((test2_pd_data, test2_W), axis=1)

    X_test2_pd = pd.DataFrame(test2_pd_data, columns=list(columns)+["Weights"])



    results = dict()
    if args.paradigm == "regr":
        mask = [True,]*p
        model_func = get_algorithm_class(args.backend)
        model = model_func(X_train_pd, duration_col, event_col, W, 0.002, **vars(args))
        print("train score", model.score(X_train_test))
            
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
    summary.to_csv("./breast_results_stable.csv")
    print("summary", summary)

    coef = summary["coef"]


    
    c_index_dict = []

    
    c_index0 = model.score(X_test0_pd, scoring_method='concordance_index')
    print("test1")
    c_index1 = model.score(X_test1_pd, scoring_method='concordance_index')
    print(c_index1)
    c_index_dict.append(c_index1)

    print("test2")
    c_index2 = model.score(X_test2_pd, scoring_method='concordance_index')
    print(c_index2)
    c_index_dict.append(c_index2)



    print("c_index")
    mean_acc = np.mean(c_index_dict)
    std_acc = np.std(c_index_dict)
    worst_acc = min(c_index_dict)
    print(mean_acc)
    print(std_acc)
    print(worst_acc)



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
        if k == "RMSE":
            RMSE_dict = dict()
            for r_test in args.r_list:
                RMSE = [v[i][r_test] for i in range(args.times)]
                RMSE_dict[r_test] = sum(RMSE)/len(RMSE)
            logger.info("RMSE average: %.3f" % (np.mean(list(RMSE_dict.values()))))
            logger.info("RMSE std: %.3f" % ((np.std(list(RMSE_dict.values())))))
            logger.info("RMSE max: %.3f" % ((np.max(list(RMSE_dict.values())))))
            logger.info("Detailed RMSE:")
            for r_test in args.r_list:
                logger.info("%.1f: %.3f" % (r_test, RMSE_dict[r_test]))
        elif k == "beta_hat":
            beta_hat_array = np.array(v)
            beta_hat_mean = np.mean(beta_hat_array, axis=0)
            logger.info("%s: %s" % (k, beta_hat_mean))
            if args.bv_analysis:
                bv_dict = dict()
                bv_dict["s"] = BV_analysis(beta_hat_array[:,:p_s], beta[:p_s])
                bv_dict["v"] = BV_analysis(beta_hat_array[:,p_s:], beta[p_s:])
                bv_dict["all"] = BV_analysis(beta_hat_array, beta)
                for covariates in ["s", "v", "all"]:
                    logger.info("Bias for %s: %.4f, variance for %s: %.4f" % (covariates, bv_dict[covariates][0], covariates, bv_dict[covariates][1]))
        else:
            logger.info("%s: %.3f" % (k, sum(v)/len(v)))
            
           
