import numpy as np
import matplotlib.pylab as plt
import ot
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
import json

n = 500 # bumber of samples
k = 2 # number of classes, binary classification
n_iter = 10 # number of random draws
vis = True # for visualization, set to False if only storing in a file is needed

# setting up the classifier and the cross-validation
C_range = np.logspace(-2, 3, 5)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
param_grid = dict(C=C_range)
clf = GridSearchCV(SVC(kernel="linear",probability = True), param_grid=param_grid, cv=cv)

# bin positions
xfro = np.arange(k, dtype=np.float64) # k support points of Frogner's loss
xm = np.linspace(1,(n*10), n) # n support points for the empirical distribution

# loss matrix
Mfro = ot.dist(xfro.reshape((k, 1)), xfro.reshape((k, 1))) # kxk loss matrix of Frogner et al.
Mfro /= Mfro.max() # normalization

steps = 3 # increment for the sample size in the loop below

froW = np.zeros((n_iter,steps))
mW = np.zeros((n_iter,steps))
scr = np.zeros((n_iter,steps))

for size in xrange(steps):
    for it in xrange(n_iter):
        print 'Noise = '+str(0.1*it)
        X, y = make_classification(n_samples=n+size*1000, random_state=1, n_clusters_per_class=1, flip_y=0.1*it) # generate data
        M = ot.dist(X, X) # nxn loss matrix
        M /= M.max()

        clf.fit(X, y)
        scr[it,size] = 1-clf.score(X,y) # accuracy of the classifier

        probas = clf.predict_proba(X) # predict probabilities used a labeling function

        mW[it,size] = ot.emd2(probas[:,1]/np.sum(probas[:,1]), y/float(np.sum(y)), M) # calculate our global loss

        for i in range(n):
            lYi = [0, 0]
            lYi[y[i]]=1
            hYi = probas[i,:]
            froW[it,size] = froW[it,size] + ot.emd2(lYi, hYi, Mfro) # calculate the loss of Frogner for every point

results = {}
results['loss_zo'] = scr.tolist()
results['loss_fro'] = froW.tolist()
results['loss_our'] = mW.tolist()

# saving the results
with open('results_losses.txt', 'w+') as outfile:
    json.dump(results, outfile)

# plotting the results
if vis == True:
    max_m = 100

    with open('results_losses.txt') as data_file:
        results = json.load(data_file)

    mW = np.asarray(results['loss_our'])
    froW = np.asarray(results['loss_fro'])
    scr = np.asarray(results['loss_zo'])

    data = [mW,froW,scr]
    title = ['Our error function','Error function with Wasserstein loss','Error function with 0-1 loss']
    col = ['black', 'black', 'black']
    ls = ['-','-','-']
    file = ['our_loss','frog_loss','01_loss']


    for i in xrange(len(data)):

        meanYArray = np.mean(data[i], axis=1)
        stdYArray = np.std(data[i], axis=1)

        max_W = np.max(meanYArray)

        plt.figure(i,figsize = (15,13))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.tick_params(axis="both", which="both", bottom="off", top="off",
                        labelbottom="on", left="off", right="off", labelleft="on")


        for y in np.linspace(0, max_W+0.2*max_W, 10):
            plt.plot(np.linspace(0, max_m, 100), [y] * len(np.linspace(0, max_m, 100)), "--", lw=1.5, color="black", alpha=0.3)

        plt.plot(np.linspace(0,1,10)*100, meanYArray, c = col[i], linestyle = ls[i], lw = 8)
        plt.title(title[i],fontsize = 50)

        plt.xlim(0,100)
        plt.ylim(0,max_W+0.2*max_W)

        plt.xlabel('Percentage of flipped labels', fontsize = 50)
        plt.ylabel('Error value', fontsize = 50)
        plt.xticks(fontsize=35)
        plt.yticks(fontsize=35)

        plt.show()







