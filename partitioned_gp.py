import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


class PGP:
    def __init__(self, kernel, region_classifier, n_restarts=5):
        self.kernel = kernel
        self.region_classifier = region_classifier
        self.classes = region_classifier.classes_
        self.num_regions = len(self.classes)
        self.local_gp = []        
        self.X_group = []
        self.y_group = []        
        self.n_restarts = n_restarts
        for c in self.classes:
            gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.n_restarts)
            self.local_gp.append(gp)

    def fit(self, X, y):
        c_train = self.region_classifier.predict(X)
        self.unknown_classes = self.classes.copy()
        for c in self.classes:
            if c not in c_train:
                pass
            else:
                local_X = X[c_train == c]
                local_y = y[c_train == c]
                self.X_group.append(local_X)
                self.y_group.append(local_y)
                self.local_gp[c].fit(local_X, local_y)
                self.unknown_classes = np.delete(self.unknown_classes, np.where(self.unknown_classes==c))

    def predict(self, X_new, return_std=False):
        c_new = self.region_classifier.predict(X_new)
        mu = np.zeros(len(X_new))
        std = np.zeros(len(X_new))
        for c in np.unique(c_new):
            if c in self.unknown_classes:
                pass
            else:
                mu[c_new==c], std[c_new==c] = self.local_gp[c].predict(X_new[c_new==c], return_std=True)
        if return_std:
            return mu, std
        else:
            return mu