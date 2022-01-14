import numpy as np
from scipy.spatial.distance import cdist


def crude_grad(X, y, gamma=1.1):
    if np.ndim(X) == 1:
        X = X[np.newaxis, :]
    
    # define radius for closed ball with the maximum pairwise distance
    Xdist = cdist(X, X)
    np.fill_diagonal(Xdist, np.inf)
    
    row = np.argmin(Xdist, axis=0)
    col = np.argmax(np.min(Xdist, axis=0))
    r_max = gamma * Xdist[col, row[col]]

    grads = np.empty(len(X))
    
    for i in range(len(X)):
        Xdist_i = Xdist[i, np.where(Xdist[i] <= r_max)]
        ydist_i = abs(y[np.where(Xdist[i] <= r_max)] - y[i])
        
        grads[i] = (ydist_i / Xdist_i).mean()
    return grads, r_max