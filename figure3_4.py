from math import cos, sin, pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
import ot
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
import mmd
import theano.tensor as T
import theano
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.ioff()
np.random.seed(1976)
np.set_printoptions(precision=3)
scaler = MinMaxScaler()

def generateBiModGaussian(centers, sigma, n, npdf, A, b, inv=0):
    result = {}
    nbPoints = npdf
    xtmp = np.concatenate((centers[0] + sigma[0] * np.random.standard_normal((n / 2, 2)),
                           centers[1] + sigma[1] * np.random.standard_normal((n, 2)),
                           centers[2] + sigma[0] * np.random.standard_normal((n / 2, 2))))
    result['X'] = xtmp.dot(A) + b

    if inv == 0:
        result['y'] = np.concatenate((np.zeros(n / 2), np.ones(n), np.zeros(n / 2)))
        dist = norm(centers[1][0], sigma[1])
        result['labelfunc'] = dist.pdf(np.linspace(np.min(xtmp[:, 0]), np.max(xtmp[:, 0]), nbPoints))
    else:
        result['y'] = np.concatenate((np.ones(n / 2), np.zeros(n), np.ones(n / 2)))
        dist1 = norm(centers[0][0], sigma[0])
        dist2 = norm(centers[2][0], sigma[0])
        result['labelfunc'] = np.concatenate((dist1.pdf(np.linspace(np.min(xtmp[:n / 2, 0]), 0, nbPoints / 2)),
                                              dist2.pdf(np.linspace(0, np.max(xtmp[3 * n / 2:, 0]), nbPoints / 2))))
    return result


def make_trans_moons(num_src, ns, nt, degree):
    Xs = []
    ys = []

    Xt, yt = make_moons(nt, shuffle=False, noise=0.01)
    xMin = np.min(Xt[:, 0])
    xMax = np.max(Xt[:, 0])

    noise_int = np.linspace(0.01, 0.05, num_src)
    for i in noise_int:
        Xtmp, ytmp = make_moons(ns, shuffle=False, noise=i)
        Xs.append(Xtmp)
        ys.append(ytmp)
        if xMin < min([np.min(Xtmp[:, 0]), np.min(Xt[:, 0])]):
            xMin = min([np.min(Xtmp[:, 0]), np.min(Xt[:, 0])])
        if xMax > max([np.max(Xtmp[:, 0]), np.max(Xt[:, 0])]):
            xMax = max([np.max(Xtmp[:, 0]), np.max(Xt[:, 0])])

    transS = [-np.mean(a, axis=0) for a in Xs]
    transT = -np.mean(Xs[0], axis=0)
    Xs = 2 * (Xs + transS)
    Xt = 2 * (Xt + transT)

    theta = -degree * pi / 180
    theta_int = np.linspace(-30 * pi / 180, 30 * pi / 180, num_src)
    rotation = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    Xs = [np.dot(Xs[i], np.array([[cos(t), sin(t)], [-sin(t), cos(t)]])) for i, t in enumerate(theta_int)]
    Xt = np.dot(Xt, rotation.T)
    return Xs, ys, Xt, yt, xMin, xMax


plot_error = True # True if plotting errors is required
plot_moons = False # True if plotting data is required
theta_range = np.linspace(30, 360, 50) # the range of angles to be covered

# visualization parameters
cgray = "#FAFAFA"
cblack = "#000000"

plt.rcParams.update({'font.size': 32})
linewidth = 8
markerSize = 120

# Dataset generation parameters
ns = 300 # number of source points
nt = 20 # number of target points

nb_tr = 3 # number of trials for averaging

num_src = 5 # number of source domains

# variables to stock the results
lambdaWa = []
Wdista = []
MMDa = []
true_errora = []

# variables used in teano for mmd calculation
Xth, Yth = T.matrices('X', 'Y')
sigmath = T.scalar('sigma')
fn = theano.function([Xth, Yth, sigmath],
                     mmd.rbf_mmd2(Xth, Yth, sigma = sigmath))

a, b = np.ones((ns,)) / ns, np.ones((nt,)) / nt # empirical distributions for source and target domains
reg = 1e-1 # entropic regularization for \lambda computation
Mb = ot.utils.dist0(ns) # cost matrix on bins
Mb /= Mb.max() # normalization

if plot_moons: # to plot the data (avoid when len(theta_range)>5)
    fig, axes = plt.subplots(len(theta_range), num_src, figsize=(21, 16))

for j, it in enumerate(theta_range):
    lambdaW = []
    Wdist = []
    MMD = []
    true_error = []

    for tr in xrange(nb_tr):
        print 'Degree = ' + str(it)

        Xsrc, Ysrc, XT, YT, xMin, xMax = make_trans_moons(num_src, ns, nt, it) # generate moons as explained in the paper
        labels = np.unique(np.concatenate(Ysrc))

        l1 = labels[0]
        l2 = labels[1]

        probaSrc = []

        for i in range(len(Xsrc)):

            M = ot.dist(Xsrc[i], XT) # get cost matrix of ith source domain and the target one
            M /= M.max()

            Wdist.append(ot.emd2(a, b, M)) # calculate the Wasserstein distance

            mmd2,_ = fn(Xsrc[i], XT, sigma = mmd.kernelwidth(Xsrc[i],XT)) # calculatre the MMD distance
            MMD.append(mmd2)

            # use 1NN classifier to get the true error value
            neigh = KNeighborsClassifier(n_neighbors=3)
            neigh.fit(Xsrc[i], Ysrc[i])
            true_error.append(1 - neigh.score(XT, YT))

            # empirical estimation of the source labeling function with density estimation
            dist = norm(np.mean(Xsrc[i][Ysrc[i] == 1, 0]), np.std(Xsrc[i][Ysrc[i] == 1, 0]))
            probaSrc.append(dist.pdf(np.linspace(xMin, xMax, ns)))
            probaSrc[-1] = probaSrc[-1].astype(np.double) / probaSrc[-1].sum()

        # empirical estimation of the target labeling function with density estimation
        dist = norm(np.mean(XT[YT == 1, 0]), np.std(XT[YT == 1, 0]))
        probaTar = dist.pdf(np.linspace(np.min(XT[:, 0]), np.max(XT[:, 0]), ns))
        probaTar = probaTar.astype(np.double) / probaTar.sum()

        lab_funcs = np.vstack((np.vstack(probaSrc), probaTar)).T # stock all labeling functions in a single vector
        bary_wass, log = ot.bregman.barycenter(lab_funcs, Mb, reg, log=True) # calculate the barycenter of the labeling functions

        W = []
        reg_lambda = 1e-2
        for func in lab_funcs.T:
            W.append(ot.sinkhorn2(bary_wass, func, Mb, reg_lambda)) # distances between the barycenter and each labeling function

        lambdaW.append(np.asarray(W).sum())  # empirical lambda value

    Wdista.append(np.mean(np.reshape(np.asarray(Wdist), (nb_tr, num_src)), axis=0))
    MMDa.append(np.mean(np.reshape(np.asarray(MMD), (nb_tr, num_src)), axis=0))
    true_errora.append(np.mean(np.reshape(np.asarray(true_error), (nb_tr, num_src)), axis=0))
    lambdaWa.append(np.mean(lambdaW))

    if plot_moons:
        for k in range(len(Xsrc)):

            XS = Xsrc[k]
            YS = Ysrc[k]

            xMin = min([np.min(XS[:, 0]), np.min(XT[:, 0])]) - 1
            xMax = max([np.max(XS[:, 0]), np.max(XT[:, 0])]) + 1
            yMin = min([np.min(XS[:, 1]), np.min(XT[:, 1])]) - 1
            yMax = max([np.max(XS[:, 1]), np.max(XT[:, 1])]) + 1


            def drawPoints(ax, X, Y, b, r, m, z, label):
                ax.scatter(X[:, 0], X[:, 1], c=Y, label=label, edgecolor='black',
                           linewidth='1', marker=m, s=[markerSize] * len(X),
                           cmap=ListedColormap([b, r]), zorder=z)


            def finalizePlot(ax, xMin, xMax, yMin, yMax, flag_legend=False):
                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)

                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

                ax.set_xticks([])
                ax.set_yticks([])

                ax.set_xlim(xMin, xMax)
                ax.set_ylim(yMin, yMax)


            drawPoints(axes[j, k], XS, YS, cgray, cblack, "o", 1, label = "Source distribution")
            drawPoints(axes[j, k], XT, YT, cgray, cblack, "v", 2, label = "Target distribution")
            leg = False
            finalizePlot(axes[j, k], xMin, xMax, yMin, yMax, leg)

            if k==2:
                axes[j,k].set_title(str(it)+r'$^\circ$')


if plot_error:

    distA = np.mean(np.reshape(np.asarray(Wdista), (len(theta_range), num_src)), axis=1)
    distMMDA = np.mean(np.reshape(np.asarray(MMDa), (len(theta_range), num_src)), axis=1)
    errorA = np.mean(np.reshape(np.asarray(true_errora), (len(theta_range), num_src)), axis=1)

    # plot lambda vs true error
    plt.figure(2,figsize = (12,8))

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    plt.plot(theta_range, scaler.fit_transform(np.asarray(lambdaWa).reshape(-1, 1)), c='k', lw=5, label=r'$\hat{\lambda}$')
    plt.plot(theta_range, scaler.fit_transform(np.asarray(errorA).reshape(-1, 1)), c='gray', linestyle = '--', lw=5, label='1NN error')

    plt.ylim(0, 1.2)
    plt.xlim(30, 360)
    plt.xlabel(r'Rotation angle $\theta^\circ$', fontsize = 22)
    leg = plt.legend(loc = "upper center", ncol = 3, fontsize = 32, handletextpad=0.1, labelspacing=.1, markerscale=2., frameon = False, bbox_to_anchor=(0.5, 1.2))
    plt.show()

    # plot MMD vs Wasserstein vs true error
    plt.figure(3,figsize = (12,8))

    plt.plot(theta_range, scaler.fit_transform(np.asarray(distA).reshape(-1, 1)), c='black', linestyle = '--', lw=5, label='Wasserstein')
    plt.plot(theta_range, scaler.fit_transform(np.asarray(distMMDA).reshape(-1, 1)), c='black', linestyle = '-', lw=5, label='MMD')
    plt.plot(theta_range, scaler.fit_transform(np.asarray(errorA).reshape(-1, 1)), c='gray', linestyle = '--', lw=5, label='1NN error')

    plt.ylim(0, 1.2)
    plt.xlim(30, 360)
    plt.xlabel(r'Rotation angle $\theta^\circ$', fontsize = 22)
    leg = plt.legend(loc = "upper center", ncol = 3, fontsize = 32, handletextpad=0.1, labelspacing=.1, markerscale=2., frameon = False, bbox_to_anchor=(0.5, 1.2))
    plt.show()
