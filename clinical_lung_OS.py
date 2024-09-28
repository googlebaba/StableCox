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
duration_col='Survival.months'
event_col='Survival.status'
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
    parser.add_argument("--iters_balance", type=int, default=3000)

    parser.add_argument("--topN", type=int, default=5)
    # backend model 
    parser.add_argument("--backend", choices=["OLS", "Lasso", "Ridge", "Weighted_cox"], default="Weighted_cox")
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
    oracle_mask = [True,]*p_s + [False,]*p_v

    training_data = pd.read_csv('./clinical_data/lung_cancer/train_pd.csv', index_col=0)
    test0_data = pd.read_csv('./clinical_data/lung_cancer/test0_pd.csv', index_col=0)
    test1_data = pd.read_csv('./clinical_data/lung_cancer/test1_pd.csv', index_col=0)
    test2_data = pd.read_csv('./clinical_data/lung_cancer/test2_pd.csv', index_col=0)
    test3_data = pd.read_csv('./clinical_data/lung_cancer/test3_pd.csv', index_col=0)
    
    test4_data = pd.read_csv('./clinical_data/lung_cancer/test4_pd.csv', index_col=0)
    test5_data = pd.read_csv('./clinical_data/lung_cancer/test5_pd.csv', index_col=0)
    test6_data = pd.read_csv('./clinical_data/lung_cancer/test6_pd.csv', index_col=0)
    test7_data = pd.read_csv('./clinical_data/lung_cancer/test7_pd.csv', index_col=0)
       
    training_data.drop(['Reccur.status', 'Reccur.months'], axis=1, inplace=True)
    test0_data.drop(['Reccur.status', 'Reccur.months'], axis=1, inplace=True)
    test1_data.drop(['Reccur.status', 'Reccur.months'], axis=1, inplace=True)
    test2_data.drop(['Reccur.status', 'Reccur.months'], axis=1, inplace=True)
    test3_data.drop(['Reccur.status', 'Reccur.months'], axis=1, inplace=True)
    
    test4_data.drop(['Reccur.status', 'Reccur.months'], axis=1, inplace=True)
    test5_data.drop(['Reccur.status', 'Reccur.months'], axis=1, inplace=True)
    test6_data.drop(['Reccur.status', 'Reccur.months'], axis=1, inplace=True) 
    test7_data.drop(['Reccur.status', 'Reccur.months'],  axis=1, inplace=True)
    
    
    X_train_pd = training_data
    tmp = training_data[training_data.columns[:-2]]
    
    X_train_np = np.array(training_data[training_data.columns[:-2]])
    p = X_train_np.shape[1]
    n= X_train_np.shape[0]

    if args.reweighting == "DWR":
        if args.decorrelation_type == "global":
            cov_mask = get_cov_mask(np.zeros(p))
        elif args.decorrelation_type == "group":
            cov_mask = get_cov_mask(np.array(oracle_mask, np.float))
        else:
            raise NotImplementedError            
        W = DWR(X_train_np, cov_mask=cov_mask, order=args.order, num_steps=args.iters_balance, logger=logger, device=device)
    elif args.reweighting == "SRDO":
        W = SRDO(X_train_np, p_s, hidden_layer_sizes = (37, 4), decorrelation_type="global", max_iter=args.iters_balance)
    else:
        W = np.ones((n, 1))/n
        
    mean_value = np.mean(W)
    W = W * (1/mean_value)
    
    W = np.clip(W, 0.3, 3)

    columns = X_train_pd.columns
    all_X = np.concatenate((X_train_pd, W), axis=1)
    X_train_test = pd.DataFrame(all_X, columns=list(columns)+["Weights"])


    test_W = np.ones((test0_data.shape[0], 1))
    test0_data = np.concatenate((test0_data, test_W), axis=1)
    test0_pd = pd.DataFrame(test0_data, columns=list(columns)+["Weights"])


    test_W = np.ones((test1_data.shape[0], 1))
    test1_data = np.concatenate((test1_data, test_W), axis=1)
    test1_pd = pd.DataFrame(test1_data, columns=list(columns)+["Weights"])


    test_W = np.ones((test2_data.shape[0], 1))
    test2_data = np.concatenate((test2_data, test_W), axis=1)
    test2_pd = pd.DataFrame(test2_data, columns=list(columns)+["Weights"])


    test_W = np.ones((test3_data.shape[0], 1))
    test3_data = np.concatenate((test3_data, test_W), axis=1)
    test3_pd = pd.DataFrame(test3_data, columns=list(columns)+["Weights"])
    
    test_W = np.ones((test4_data.shape[0], 1))
    test4_data = np.concatenate((test4_data, test_W), axis=1)
    test4_pd = pd.DataFrame(test4_data, columns=list(columns)+["Weights"])

    test_W = np.ones((test5_data.shape[0], 1))
    test5_data = np.concatenate((test5_data, test_W), axis=1)
    test5_pd = pd.DataFrame(test5_data, columns=list(columns)+["Weights"])

    test_W = np.ones((test6_data.shape[0], 1))
    test6_data = np.concatenate((test6_data, test_W), axis=1)
    test6_pd = pd.DataFrame(test6_data, columns=list(columns)+["Weights"])

    test_W = np.ones((test7_data.shape[0], 1))
    test7_data = np.concatenate((test7_data, test_W), axis=1)
    test7_pd = pd.DataFrame(test7_data, columns=list(columns)+["Weights"])

    results = dict()
    if args.paradigm == "regr":
        mask = [True,]*p
        model_func = get_algorithm_class(args.backend)
        model = model_func(X_train_pd, duration_col, event_col, W, 0.002, **vars(args))
            
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
    coef = summary["coef"]
    
    c_index_dict = []


    c_index0 = model.score(test0_pd, scoring_method='concordance_index')
    c_index_dict.append(c_index0)

    
    c_index1 = model.score(test1_pd, scoring_method='concordance_index')
    c_index_dict.append(c_index1)


    c_index4 = model.score(test4_pd, scoring_method='concordance_index')
    c_index_dict.append(c_index4)

    c_index5 = model.score(test5_pd, scoring_method='concordance_index')
    c_index_dict.append(c_index5)

    c_index6 = model.score(test6_pd, scoring_method='concordance_index')
    c_index_dict.append(c_index6)
    
    c_index7 = model.score(test7_pd, scoring_method='concordance_index')
    c_index_dict.append(c_index7)


    print("c_index")
    print("\n".join([str(s) for s in c_index_dict]))
    mean_acc = np.mean(c_index_dict)
    std_acc = np.std(c_index_dict)
    worst_acc = min(c_index_dict)
    print("mean_acc", mean_acc)
    print("std_acc", std_acc)
    print("worst_acc", worst_acc)

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
            
           
