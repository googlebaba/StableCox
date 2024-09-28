import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
def column_wise_resampling(x, p_s, decorrelation_type="global", random_state = 0, category_groups= None,  **options):
    """
    Perform column-wise random resampling to break the joint distribution of p(x).
    In practice, we can perform resampling without replacement (a.k.a. permutation) to retain all the data points of feature x_j. 
    Moreover, if the practitioner has some priors on which features should be permuted,
    it can be passed through options by specifying 'sensitive_variables', by default it contains all the features
    """
    rng = np.random.RandomState(random_state)
    n, p = x.shape
    if 'sensitive_variables' in options:
        sensitive_variables = options['sensitive_variables']
    else:
        sensitive_variables = [i for i in range(p)] 
    x_decorrelation = np.zeros([n, p])
    if decorrelation_type == "global":
        for i in sensitive_variables:
            rand_idx = rng.permutation(n)
            x_decorrelation[:, i] = x[rand_idx, i]
    elif decorrelation_type == "group":
        rand_idx = rng.permutation(n)
        x_decorrelation[:, :p_s] = x[rand_idx, :p_s]
        for i in range(p_s, p):
            rand_idx = rng.permutation(n)
            x_decorrelation[:, i] = x[rand_idx, i]
    elif decorrelation_type == "category":
        print("len", len(category_groups))
        pre_j = 1
        for i, j in enumerate(category_groups):
            if j ==pre_j:
                x_decorrelation[:, i] = x[:, i]
                pre_j = j
                continue
            count = category_groups.count(j)
            if count == 1:
                rand_idx = rng.permutation(n)
                x_decorrelation[:, i] = x[rand_idx, i]
            else:
                print(i, count)
                rand_idx = rng.permutation(n)
                x_decorrelation[:, i:i+count] = x[rand_idx, i:i+count]
            pre_j = j 

    else:
        assert False
    return x_decorrelation

def SRDO(x, p_s, decorrelation_type="global", solver = 'adam', hidden_layer_sizes = (100, 5), category_groups = None, max_iter = 500, random_state = 3):
    """
    Calcualte new sample weights by density ratio estimation
           q(x)   P(x belongs to q(x) | x) 
    w(x) = ---- = ------------------------ 
           p(x)   P(x belongs to p(x) | x)
    """
    n, p = x.shape
    x_decorrelation = column_wise_resampling(x, p_s, decorrelation_type, category_groups = category_groups, random_state = random_state)
    P = pd.DataFrame(x)
    Q = pd.DataFrame(x_decorrelation)
    corr_matrix = np.corrcoef(x, rowvar=False)
    abs_corr_matrix = np.abs(corr_matrix)
    sum_abs_corr = np.sum(abs_corr_matrix) - np.sum(np.diag(abs_corr_matrix))
    P['src'] = 1 # 1 means source distribution
    Q['src'] = 0 # 0 means target distribution
    Z = pd.concat([P, Q], ignore_index=True, axis=0)
    labels = Z['src'].values
    Z = Z.drop('src', axis=1).values
    P, Q = P.values, Q.values
    # train a multi-layer perceptron to classify the source and target distribution
    #clf = LogisticRegression(random_state=0, C=0.1)
    clf = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    clf.fit(Z, labels)
    proba = clf.predict_proba(Z)[:len(P), 1]
    weights = (1./proba) - 1. # calculate sample weights by density ratio
    weights /= np.sum(weights) # normalize the weights to get average 1
    weights = np.reshape(weights, [n,1])
    return weights
