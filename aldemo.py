import numpy as np
import matplotlib.pyplot as plt
import inspect
import warnings
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVC
from sklearn.cluster import MeanShift
from partitioned_gp import PGP
from utils import crude_grad
from DOE.passive_learning import lhs
from DOE.active_learning import *


"""
Active Learning Demo (2022/02/07, C. Lee (cheolheil@vt.edu))
This script is the main script for active learning demo.
Note that the simulation will be different for each iteration due to not-fixed random seed.
"""

# don't show warnings
warnings.filterwarnings("ignore")

# simulation function to learn
def f(x, noise=False):
    if np.ndim(x) == 1:
        x = x[np.newaxis, :]
    if x.shape[1] != 2:
        raise Exception("Input must be 2-dim")

    x = 6 * (x - 0.5) + 1
    y = x[:, 0] * np.exp(-np.square(x[:, 0])-np.square(x[:, 1]))

    if noise:
        return y + np.random.normal(0, 1e-3, len(x))
    else:
        return y


print('=========== Welcome to Active Learning Demo! ===========')
user_choice = input('Choose a strategy from:\n1. Random\n2. Uncertainty Sampling\n3. Variance Reduction (IMSE)\n4. Maximin Distance\n5. Query by Committee\n6. Partitioned Active Learning\n--> Enter the number of the strategy (Press Ctrl+C to exit): ')

methods = {'1': random_sampling, '2': uncertainty_sampling, '3': imse, '4': maximin_dist, '5': var_qbc, '6': pimse}
strategies = {'1': 'Random', '2': 'Uncertainty Sampling', '3': 'Variance Reduction', '4': 'Maximin Distance', '5': 'Query by Committee', '6': 'Partitioned Active Learning'}
acq_fun = methods[user_choice]
strategy = strategies[user_choice]
arg_names = inspect.getfullargspec(acq_fun).args

m = 100     # resolution for ground truth plotting
n = 50      # resolution for candidate points

# generate the ground truth data
X1_ = np.linspace(0, 1, m)
X2_ = np.linspace(0, 1, m)
X1, X2 = np.meshgrid(X1_, X2_)
XX = np.array([X1.ravel(), X2.ravel()]).T
yy = f(XX)

# generate the initial data
X = lhs(20, 2, criterion='maximin', seed=3)
y = f(X)

# initialize model based on the user choice
if acq_fun == var_qbc:
    # model for QBC
    kernel1 = C() * RBF(length_scale_bounds=(1., 1e2))
    kernel2 = C() * RationalQuadratic()
    kernel3 = C() * RBF(length_scale_bounds=(1e-2, 1.))
    model = VotingRegressor(estimators=[('gp_lf', GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=5)),
                                            ('gp_dot', GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=5)),
                                            ('gp_hf', GaussianProcessRegressor(kernel=kernel3, n_restarts_optimizer=5))])
elif acq_fun == pimse:
    # model for PAL
    kernel = C() * RBF(1., (1e-2, 1e2))
    grad_init, r_max = crude_grad(X, y)
    cls = MeanShift()
    c_init = cls.fit_predict(grad_init.reshape(-1, 1))
    c_init[c_init!=0] = 1
    svc = SVC(probability=True)
    svc.fit(X, c_init)
    model = PGP(kernel=kernel, region_classifier=svc)
else:
    # model for else
    kernel = C() * RBF(1., (1e-2, 1e2))
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

# fit the model
model.fit(X, y)
y_pred = model.predict(XX)

# generate candidate points
C1_ = np.linspace(0, 1, n)
C2_ = np.linspace(0, 1, n)
C1, C2 = np.meshgrid(C1_, C2_)
X_cand = np.array([C1.ravel(), C2.ravel()]).T

# plot underyling truth and training data on the left plot
plt.ion()
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121)
ax1.set_title('Trained Samples with {}'.format(strategy))
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.contourf(X1, X2, yy.reshape(m, m), levels=50, cmap='PRGn')
ax1.scatter(X[:, 0], X[:, 1], edgecolors='k', facecolor='none', label='Initial Samples', clip_on=False)
ax1.scatter([], [], c='r', edgecolors='k', marker='D', label='Active Samples', clip_on=False)
ax1.legend(loc=2)

# i as the number of active learning iterations
i = 0

# plot the predicted function
ax2 = fig.add_subplot(122)
ax2.set_title('Predicted Function (# AL={})'.format(i))
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ctr = ax2.contourf(X1, X2, y_pred.reshape(m, m), levels=30, cmap='PRGn')
# print out the computation time for searching
tx = ax2.text(0.5, 0.95, 'Searching Time: 0.0s', fontsize=12)

# update the plot and model with active learning with user hitting enter
while True:
    input('Press Enter to continue (Press Ctrl+C to terminate)')
    kwargs = {'X_cand': X_cand, 'gp': model, 'X_ref': XX}
    required_args = inspect.getfullargspec(acq_fun).args
    pass_args = {}
    for arg in required_args:
        if arg in kwargs:
            pass_args[arg] = kwargs[arg]
    tic = time.time()
    x_new = X_cand[np.argmax(acq_fun(**pass_args))]
    tac = time.time()
    # update X, y and X_cand
    X = np.vstack((X, x_new))
    y = np.append(y, f(x_new))
    X_cand = np.delete(X_cand, np.argmax(acq_fun(**pass_args)), axis=0)
    # update the model
    model.fit(X, y)
    y_pred = model.predict(XX)
    i += 1
    # update the plot
    ax1.scatter(x_new[0], x_new[1], c='r', edgecolors='k', marker='D', label='Active Samples', clip_on=False)
    ax2.set_title('Predicted Function (# AL={})'.format(i+1))
    tx.set_text('Searching Time: {:.3f}s'.format(tac-tic))
    for coll in ctr.collections:
        plt.gca().collections.remove(coll)
    ctr = ax2.contourf(X1, X2, y_pred.reshape(m, m), levels=30, cmap='PRGn')
    fig.canvas.draw()
    fig.canvas.flush_events()
        