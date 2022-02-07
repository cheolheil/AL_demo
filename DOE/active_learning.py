import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist


"""
Active Learning Acquisition Functions (2022/02/07, C. Lee (cheolheil@vt.edu))
This script contains the following strategies and related functions:
1. Variance Reduction (IMSE)
2. PIMSE
3. Uncertainty Sampling
4. Variance-based QBC
5. Maximin Distance
*. Random Sampling (Passive Learning)

The acquisition functions have similar interfaces.
1. X_cand: candidate set of points that are to be evaluated
2. gp: Gaussian Process model
3. X_ref: reference set of points for some acquisition functions (generally, the entire design space)
4. weights: prior information for imposing importance on canddiate points (default: uniform)
5. global_search: whether to use global search or local search (default: True). For more information,
   please refer to the following paper: 
   Lee, Cheolhei, et al. "Partitioned Active Learning for Heterogeneous Systems.",
   arXiv preprint arXiv:2105.08547 (2021).
"""

# Variacne Reduction (IMSE: Integrated Mean Squared Error)
def imse(X_cand, gp, X_ref, p_ref=None):
    n = len(X_cand)
    m = len(X_ref)

    if p_ref is None:
        p_ref = np.ones(m) / m
    else:
        if len(p_ref) != m or not np.isclose(p_ref.sum(), 1.):
            raise Exception("Probability function of reference set is invalid!")
        else:
            pass

    imse_vals = np.zeros(n)
    k_ref = gp.kernel_(gp.X_train_, X_ref)
    L = gp.L_
    v = solve_triangular(L, k_ref, lower=True)
    for i in range(n):
        xi = X_cand[i][np.newaxis, :]
        k12 = gp.kernel_(gp.X_train_, xi)
        k22 = gp.kernel_(xi) + 1e-10
        k_ = gp.kernel_(xi, X_ref)
        L_ = chol_update(L, k12, k22)
        l12 = L_[-1, :-1]
        k_ -= l12.reshape(1, -1) @ v
        imse_vals[i] = np.inner(k_ * p_ref.reshape(k_.shape), k_) / m
    return imse_vals


# Partitioned IMSE (PIMSE)
def pimse(X_cand, gp, X_ref, global_search=True):
    C_cand = gp.region_classifier.predict(X_cand)
    C_ref = gp.region_classifier.predict(X_ref)
    cand_labels = np.unique(C_cand)
    ref_labels = np.unique(C_ref)
    
    sub_var_vals = np.zeros(len(gp.local_gp))
    imse_vals = np.full(len(X_cand), -np.inf)
    for c in ref_labels:
        if c in gp.unknown_classes:
            sub_var_vals[c] = np.inf
        else:
            sub_var_vals[c] = np.square(gp.local_gp[c].predict(X_ref[C_ref == c], return_std=True)[1]).mean()

    if global_search:
        # choose the most uncertain region`
        c_sel = np.argmax(sub_var_vals)

        if c_sel not in C_cand:
            raise Exception("No candidate is not in the most uncertain region")
        elif c_sel in gp.unknown_classes:
            return random_sampling(X_cand)
        else:
            valid_cand_id = np.where(C_cand == c_sel)[0]
            X_sel_cand = X_cand[valid_cand_id]
            X_sel_ref = X_ref[valid_cand_id]
            imse_vals[valid_cand_id] = imse(X_sel_cand, gp.local_gp[c_sel], X_sel_ref)
        return imse_vals

    else:
        
        for c in cand_labels:
            if c in gp.unknown_classes:
                imse_vals[C_cand == c] = -np.inf
            else:
                imse_vals[C_cand == c] = -imse(X_cand[C_cand == c], gp.local_gp[c], X_ref[C_ref == c]) \
                                         + sub_var_vals.sum()
        return imse_vals


# Uncertainty Sampling
def uncertainty_sampling(X_cand, gp, weights=None):
    # check weights has the same length of X_cand
    if weights is None:
        weights = np.ones(len(X_cand))
        weights /= np.sum(weights)
    else:
        assert len(weights) == len(X_cand), "weights must have the same length as X_cand"
    
    return gp.predict(X_cand, return_std=True)[1] * weights


# Variance-based QBC
def var_qbc(X_cand, gp, model_preference=None, weights=None):
    # check more than one GPs are given
    assert len(gp.estimators_) > 1, "Only one GP is given"

    if model_preference is None:
        model_preference = np.ones(len(gp.estimators_))
    else:
        assert len(model_preference) == len(gp.estimators_), "model_preference must have the same length as gp"
    
    if weights is None:
        weights = np.ones(len(X_cand))
    else:
        assert len(weights) == len(X_cand), "weights must have the same length as X_cand"
        weights /= np.sum(weights)
    
    pred_group = np.zeros((len(gp.estimators_), len(X_cand)))
    for i, gp in enumerate(gp.estimators_):
        pred_group[i] = gp.predict(X_cand) * model_preference[i]    
    return np.var(pred_group, axis=0) * weights


# Maximin Distance
def maximin_dist(X_cand, gp, metric='euclidean', weights=None):
    if weights is None:
        weights = np.ones(len(X_cand))
    else:
        assert len(weights) == len(X_cand), "weights must have the same length as X_cand"
        weights /= np.sum(weights)

    X_ = gp.X_train_
    # return averaged distance of each point in X_cand to X_
    return cdist(X_cand, X_, metric=metric).min(axis=1) * weights


# Random Sampling (Note: this is not active learning, but passivie learning)
def random_sampling(X_cand):
    score = np.zeros(len(X_cand))
    score[np.random.choice(len(X_cand), 1, replace=False)] = 1
    return score


"""
Below functions are not acquisition functions or may be deprecated
"""

# Rank-one Cholesky update for IMSE criteria
def chol_update(L, a12, a22, lower=True):
    # check if L is lower triangular
    if lower:
        if np.all(np.triu(L, k=1) == 0.):
            pass
        else:
            raise Exception("L is not a lower triangle matrix")
    # if L is upper triangular, transpose it
    else:
        if np.all(np.tril(L, k=1) == 0.):
            L = L.T
        else:
            raise Exception("L is not an upper triangle matrix")

    n = len(L)
    # check a12 is compatible with L
    if len(a12) != n:
        raise Exception("a12 length must be n")
    elif np.ndim(a12) != 2:
        a12 = a12[:, np.newaxis]
    else:
        pass

    l12 = solve_triangular(L, a12, lower=True)
    l22 = np.sqrt(a22 - np.square(l12).sum())
    L_sol = np.vstack((
        np.hstack((L, np.zeros((n, 1)))),
        np.hstack((l12.T, l22))
    ))
    return L_sol



# IMSE criterion using Sherman-Morrison-Woodbury formula
# This has the same performance as the IMSE with Rank-one Cholesky update (imse)
def imse_sherman(X_cand, gp, X_ref):
    if np.ndim(X_cand) == 1:
        X_cand = X_cand[np.newaxis, :]
    else:
        pass

    m = len(X_cand)
    n = len(X_ref)

    k_ref = np.trace(gp.predict(X_ref, return_cov=True)[1]) / n

    K = gp.kernel_(gp.X_train_) + np.eye(len(gp.X_train_)) * gp.alpha
    K_inv = np.linalg.inv(K)
    k_ = gp.kernel_(X_ref, gp.X_train_)
    imse_vals = np.full(m, k_ref)
    for i in range(m):
        xi = X_cand[i][np.newaxis, :]
        k12 = gp.kernel_(xi, gp.X_train_)
        k22 = gp.kernel_(xi) + gp.alpha
        v = k22 - k12 @ K_inv @ k12.T
        g = - (1 / v) * K_inv @ k12.T
        K_aug_inv = np.vstack((np.hstack((K_inv + g @ g.T * v, g)), np.hstack((g.T, (1 / v)))))
        k_aug = np.hstack((k_, gp.kernel_(X_ref, xi)))
        imse_vals[i] = k_ref - np.trace(k_aug @ K_aug_inv @ k_aug.T) / n
    return imse_vals