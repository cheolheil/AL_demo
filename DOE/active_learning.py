import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import approx_fprime
from scipy.linalg import solve_triangular, solve
from scipy.spatial.distance import cdist, pdist, squareform


# rank one Cholesky update
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


# this is the imse criteron using the Sherman-Morrison inverse formula
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


# this is the imse criteron using the rank one Cholesky update
def imse_chol(X_cand, gp, X_ref):
    if np.ndim(X_cand) == 1:
        X_cand = X_cand[np.newaxis, :]
    else:
        pass

    m = len(X_cand)
    n = len(X_ref)

    k_ref = np.trace(gp.predict(X_ref, return_cov=True)[1]) / n

    L = gp.L_
    k_ = gp.kernel_(X_ref, gp.X_train_)
    imse_vals = np.full(m, k_ref)
    for i in range(m):
        xi = X_cand[i][np.newaxis, :]
        k12 = gp.kernel_(gp.X_train_, xi)
        k22 = gp.kernel_(xi) + gp.alpha
        L_ = chol_update(L, k12, k22)
        k_aug = np.hstack((k_, gp.kernel_(X_ref, xi)))
        imse_vals[i] = k_ref - np.trace(k_aug @ solve_triangular(L_.T, solve_triangular(L_, k_aug.T, lower=True))) / n
    return imse_vals


def fast_pimse(X_cand, gp, X_ref, global_search=True):
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
            imse_vals[valid_cand_id] = fast_imse(X_sel_cand, gp.local_gp[c_sel], X_sel_ref)
        return imse_vals

    else:
        
        for c in cand_labels:
            if c in gp.unknown_classes:
                imse_vals[C_cand == c] = -np.inf
            else:
                imse_vals[C_cand == c] = -fast_imse(X_cand[C_cand == c], gp.local_gp[c], X_ref[C_ref == c]) \
                                         + sub_var_vals.sum()
        return imse_vals


def fast_imse(X_cand, gp, X_ref, p_ref=None):
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


def vfun_(h, xi, mu, sigma):
    return np.square(h - xi) * norm.pdf(h, mu, sigma)


def int_vfun_(mu, xi, sigma, alpha, truncate=True):
    vals = np.zeros_like(mu)
    ub = xi + alpha * sigma
    lb = xi - alpha * sigma
    for i in range(len(mu)):
        if truncate:
            vals[i] = quad(vfun_, lb[i], xi, args=(xi, mu[i], sigma[i]))[0]
        else:
            vals[i] = quad(vfun_, lb[i], ub[i], args=(xi, mu[i], sigma[i]))[0]
    return vals


def eif(X, gp, xi, alpha=2., truncate=True):
    mu, sigma = gp.predict(X, return_std=True)
    mu = mu.ravel()
    eps = alpha * sigma
    if truncate:
        eif_val = np.square(eps) * (norm.cdf(xi, mu, sigma) - norm.cdf(xi - eps, mu, sigma)) - int_vfun_(mu, xi, sigma,
                                                                                                         alpha,
                                                                                                         truncate)
    else:
        eif_val = (np.square(eps) - np.square(mu - xi)) * (
                norm.cdf(xi + eps, mu, sigma) - norm.cdf(xi - eps, mu, sigma)) \
                  + 2 * (mu - xi) * np.square(sigma) * (norm.pdf(xi + eps, mu, sigma) - norm.pdf(xi - eps, mu, sigma)) \
                  - int_vfun_(mu, xi, sigma, alpha, truncate)
    return eif_val


def eff(X, gp, xi, alpha=2.):
    mu, sigma = gp.predict(X, return_std=True)
    mu = mu.ravel()
    eps = alpha * sigma
    xi_ub = xi + eps
    xi_lb = xi - eps
    eff_val = (mu - xi) * (2 * norm.cdf(xi, mu, sigma) - norm.cdf(xi_lb, mu, sigma) - norm.cdf(xi_ub, mu, sigma)) \
              - sigma * (2 * norm.pdf(xi, mu, sigma) - norm.pdf(xi_lb, mu, sigma) - norm.pdf(xi_ub, mu, sigma)) \
              + eps * (norm.cdf(xi_ub, mu, sigma) - norm.cdf(xi_lb, mu, sigma))
    return eff_val


def schre(X, gpc, p=1e-1):
    # Note that gpc must indicate the safe as 1 (True)
    gpc = gpc.base_estimator_
    K_star = gpc.kernel_(gpc.X_train_, X)  # K_star =k(x_star)
    f_star = K_star.T.dot(gpc.y_train_ - gpc.pi_)  # Line 4
    v = solve(gpc.L_, gpc.W_sr_[:, np.newaxis] * K_star)  # Line 5
    
    var_f_star = gpc.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)
    nu = norm.ppf(1 - p)
    # Since gpc is a discriminative function, a positive returned value indicate the point is safe associated with p
    return f_star - nu * np.sqrt(var_f_star)


def my_acqfun(X_cand, f_gp, h_gp, xi, w=0.5, p_p=1e-3, p_o=1e-1, alpha=2.0, p=1, X_ref=None, prod=False):
    
    a_p = norm.ppf(1 - p_p)
    a_o = norm.ppf(1 - p_o)
    
    mu_cand, sigma_cand = h_gp.predict(X_cand, return_std=True)
    if X_ref is None:
        X_ref = X_cand.copy()
        mu_ref = mu_cand.copy()
        sigma_ref = sigma_cand.copy()
    else:
        mu_ref, sigma_ref = h_gp.predict(X_ref, return_std=True)
    pessim_ind = mu_cand + a_p * sigma_cand < xi
    if np.count_nonzero(pessim_ind) == 0:
        print("No safe region")
        S_p = np.atleast_2d(X_cand[np.argmin((xi + a_p * mu_cand) / sigma_cand)])
    else:
        S_p = np.atleast_2d(X_cand[pessim_ind])
    optim_ind = mu_ref + a_o * sigma_ref < xi
    if np.count_nonzero(optim_ind) == 0:
        S_o = X_ref
    else:
        S_o = np.atleast_2d(X_ref[np.argmin((xi + a_o * mu_ref) / sigma_ref)])
    
    J_f = fast_imse(S_p, f_gp, S_o)

    J_h = eif(S_p, h_gp, xi, alpha=alpha, truncate=True)

    J_vec = np.column_stack((J_f, J_h))

    J_vec_norm = np.zeros((len(X_cand), 2))
    J_vec_norm[pessim_ind] = (J_vec - J_vec.min(0)) / (J_vec.max(0) - J_vec.min(0))

    w_vec = np.array([1 - w, w]).reshape(2, 1)
    if not prod:
        J_sca = (J_vec_norm ** p @ w_vec).ravel() ** (1 / p)
    else:
        J_sca = np.prod(J_vec, axis=1)
    return J_sca, J_vec_norm


def uncertainty_sampling(X_cand, gp, weights=None):
    # check weights has the same length of X_cand
    if weights is None:
        weights = np.ones(len(X_cand))
        weights /= np.sum(weights)
    else:
        assert len(weights) == len(X_cand), "weights must have the same length as X_cand"
    
    return gp.predict(X_cand, return_std=True)[1] * weights


# this is an acquisition function of variance-based QBC
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


def maximin_dist(X_cand, gp, metric='euclidean', weights=None):
    if weights is None:
        weights = np.ones(len(X_cand))
    else:
        assert len(weights) == len(X_cand), "weights must have the same length as X_cand"
        weights /= np.sum(weights)

    X_ = gp.X_train_
    # return averaged distance of each point in X_cand to X_
    return cdist(X_cand, X_, metric=metric).min(axis=1) * weights


def random_sampling(X_cand):
    score = np.zeros(len(X_cand))
    score[np.random.choice(len(X_cand), 1, replace=False)] = 1
    return score

