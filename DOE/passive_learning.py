import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform


def lhs(n, m, criterion='random', X_init=None, T=10000, seed=None):
    if criterion not in ['random', 'maximin']:
        raise Exception('criterion must be random or maximin.')
        
    if seed is not None:
        np.random.seed(seed)
        
    if X_init is None:
        l = np.arange(-(n - 1) / 2, (n - 1) / 2 + 1)
        L = np.zeros((n, m))
        for i in range(m):
            L[:, i] = np.random.choice(l, n, replace=False)
        U = np.random.rand(n, m)
        X_old = (L + (n - 1) / 2 + U) / n
    else:
        X_old = X_init
        
    if criterion is 'random':
        return X_old
    elif criterion is 'maximin':
        X_new = X_old.copy()
        d_vec = pdist(X_old)
        d_mat = squareform(d_vec)
        md = d_vec[np.nonzero(d_vec)].min()

        for i in range(T):
            rows = np.argwhere(d_mat == md)[0]
            row = np.random.choice(rows, 1)
            col = np.random.choice(m)
            new_row = np.random.choice(np.delete(np.arange(n), row))
            rows = [row[0], new_row]
            X_new[rows, col] = X_new[rows[::-1], col]
            new_d = cdist(X_new[rows], X_new)
            mdprime = new_d[np.nonzero(new_d)].min()
            if mdprime > md:
                d_mat[rows, :] = new_d
                d_mat.T[rows, :] = new_d
                d_vec = squareform(d_mat)
                md = d_vec[np.nonzero(d_vec)].min()
            else:
                X_new[rows, col] = X_new[rows[::-1], col]
        return X_new


def rect_lhs(xref, n):
    lb = xref.min(axis=0)
    ub = xref.max(axis=0)
    x = lhs(n, xref.shape[1])
    x = (ub - lb) * x + lb
    return x