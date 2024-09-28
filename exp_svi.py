from data.selection_bias import gen_selection_bias_data
from algorithm.DWR import DWR
from algorithm.SRDO import SRDO
from model.linear import get_algorithm_class
from metrics import get_metric_class
from utils import setup_seed, get_beta_s, get_expname, calc_var, pretty, get_cov_mask, BV_analysis
from Logger import Logger
from model.STG import STG

from sklearn.metrics import mean_squared_error
import numpy as np
import argparse
import os
import torch
from collections import defaultdict as dd

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
    parser.add_argument("--iters_balance", type=int, default=20000)

    # backend model 
    parser.add_argument("--backend", choices=["OLS"], default="OLS")
    parser.add_argument("--paradigm", choices=["fs",], default="fs")
    parser.add_argument("--iters_train", type=int, default=1000)
    parser.add_argument("--lam_backend", type=float, default=0.01) # regularizer coefficient
    parser.add_argument("--fs_type", choices=["SVI"], default="SVI")
    parser.add_argument("--mask_given", type=int, nargs="+", default=[1,1,1,1,1,0,0,0,0,0])
    parser.add_argument("--mask_threshold", type=float, default=0.2)
    parser.add_argument("--lam_STG", type=float, default=3)
    parser.add_argument("--sigma_STG", type=float, default=0.1)
    parser.add_argument("--metrics", nargs="+", default=["L1_beta_error", "L2_beta_error"])
    parser.add_argument("--bv_analysis", action="store_true")
    # SVI
    parser.add_argument("--epoch_algorithm", type=int, default=10)
    parser.add_argument("--period_MA", type=int, default=3)


    # others
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--times", type=int, default=10)
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

    # generate train data
    X_train, S_train, V_train, fs_train, Y_train = gen_selection_bias_data({**vars(args),**{"r": args.r_train}})

    beta_s = get_beta_s(p_s)
    beta_v = np.zeros(p_v)
    beta = np.concatenate([beta_s, beta_v])
    
    linear_var, nonlinear_var, total_var = calc_var(beta_s, S_train, fs_train)
    logger.info("Linear term var: %.3f, Nonlinear term var: %.3f, total var: %.3f" % (linear_var, nonlinear_var, total_var))
    
    # generate test data
    test_data = dict()
    for r_test in args.r_list:
        X_test, S_test, V_test, fs_test, Y_test = gen_selection_bias_data({**vars(args),**{"r": r_test}})
        test_data[r_test] = (X_test, S_test, V_test, fs_test, Y_test)
    
    results = dict()
    
    cov_mask = get_cov_mask(np.zeros(p))
    select_ratio_MA = []
    for epoch in range(args.epoch_algorithm):
        logger.debug("Epoch %d" % epoch)
        # reweighting
        logger.debug("cov_mask:\n" + str(cov_mask))
        W = DWR(X_train, cov_mask=cov_mask, order=args.order, num_steps=args.iters_balance, logger=logger, device=device)
        # feature selection 
        stg = STG(p, 1, sigma=args.sigma_STG, lam=args.lam_STG)
        stg.train(X_train, Y_train, W=W, epochs=5000)
        select_ratio = stg.get_ratios().detach().numpy()
        if len(select_ratio_MA) >= args.period_MA:
            select_ratio_MA.pop(0)
        select_ratio_MA.append(select_ratio)
        select_ratio = sum(select_ratio_MA)/len(select_ratio_MA)
        logger.info("Select ratio: " + pretty(select_ratio))
        logger.info("Current hard selection: " + str(np.array(select_ratio > args.mask_threshold, dtype=np.int64)))
        cov_mask = get_cov_mask(select_ratio)
            
    mask = select_ratio > args.mask_threshold
    if np.array(mask, dtype=np.int64).sum() == 0:
        logger.info("All variables are discarded!")
        assert False    
    logger.info("Hard selection: " + str(np.array(mask, dtype=np.int64)))
    model_func = get_algorithm_class(args.backend)
    model = model_func(X_train, Y_train, np.ones((n, 1))/n, **vars(args))
    model.fit(X_train[:, mask], Y_train)    
    
    # test 
    RMSE_dict = dict()
    for r_test in args.r_list:
        X_test, S_test, V_test, fs_test, Y_test = test_data[r_test]
        RMSE_dict[r_test] = mean_squared_error(Y_test, model.predict(X_test[:,mask]))
    logger.info("Average RMSE: %.3f" % np.mean(list(RMSE_dict.values())))
    logger.info("Error STD: %.3f" % np.std(list(RMSE_dict.values())))
    logger.info("Error max: %.3f" % np.max(list(RMSE_dict.values())))
    results["RMSE"] = RMSE_dict
    
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
            
           