######################################################################
#The codes are based on Python2.7. 
#Please install numpy, scipy, matplotlib packages before using.
#Thank you for your suggestions!
#
#@version 1.0
#@author CSE6740/CS7641 TA
######################################################################
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from scipy import stats
from scipy.stats import multivariate_normal
from matplotlib import cm

##
# load wine dataset which is in csv format;
data = genfromtxt('wine.data', delimiter=',')
y = data[:,0]
data = data[:,1:data.shape[1]]

##  pca the data;
ndata = stats.zscore(data, axis = 0)
covariance = np.cov(ndata.T)

d = 2
S, V = np.linalg.eig(covariance)
sortidx = S.argsort()[::-1] 
v = V[:,sortidx][:, 0:d]

# project the data to the top 2 principal directions;
pdata = ndata.dot(v)

# visualize the data; 
plt.figure(1) 
plt.scatter(pdata[y == 1, 0], pdata[y == 1, 1], color='red')
plt.hold(True)
plt.scatter(pdata[y == 2, 0], pdata[y == 2, 1], color='blue')
plt.hold(True) 
plt.scatter(pdata[y == 3, 0], pdata[y == 3, 1], color='green')

##
# histogram for first dimension of pdata; 
# find the range of the data; 
datano = pdata.shape[0]

min_data = np.min(pdata[:, 0])
max_data = np.max(pdata[:, 0])
nbin = 10
sbin = (max_data - min_data) / nbin
# create the bins; 
boundary = np.arange(min_data, max_data, sbin)

# just loop over the data points, and count how many of data points are in
# each bin; 
myhist = np.zeros((nbin, 1))
for i in range(0, datano):
    
    which_bin = np.max(np.nonzero(boundary <= pdata[i,0])[0])
    myhist[which_bin] = myhist[which_bin] + 1

myhist = myhist * nbin / datano

# bar plot; 
fig = plt.figure(2)
ax = fig.add_subplot(111)
bins = range(myhist.shape[0])
ax.bar(bins, myhist[:,0], align='center')
ax.set_xticks(range(myhist.shape[0]))
ax.set_xticklabels(range(1, myhist.shape[0] + 1))
plt.title('Histogram') 

## 
# for 2 dimensional data;

min_data = np.min(pdata, axis=0)
max_data = np.max(pdata, axis=0)
nbin = 30 # you can change the number of bins in each dimension; 
sbin = (max_data - min_data) / nbin

# create the bins; 
array1 = np.arange(min_data[0], max_data[0] + sbin[0], sbin[0])
array2 = np.arange(min_data[1], max_data[1] + sbin[1], sbin[1])
array1 = array1.reshape(1, array1.shape[0])
array2 = array2.reshape(1, array2.shape[0])
boundary = np.concatenate((array1, array2), axis = 0)

# just loop over the data points, and count how many of data points are in
# each bin; 
myhist2 = np.zeros((nbin, nbin))
for i in range(0, datano):
    
    which_bin1 = np.max(np.nonzero(boundary[0,:] <= pdata[i,0])[0])
    which_bin2 = np.max(np.nonzero(boundary[1,:] <= pdata[i,1])[0])
    myhist2[which_bin1, which_bin2] = myhist2[which_bin1, which_bin2] + 1

myhist2 = myhist2 * nbin / datano

# two dimensional bar plot;
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

xpos = np.arange(myhist2.shape[0])
ypos = np.arange(myhist2.shape[1])
xposM, yposM = np.meshgrid(xpos, ypos)
zpos = myhist2
zpos = zpos.ravel()
dx = 0.5
dy = 0.5   
dz = zpos

values = np.linspace(0, 1, xposM.ravel().shape[0])
colors = cm.rainbow(values)

ax.bar3d(yposM.ravel(), xposM.ravel(), dz * 0, dx, dy, dz, color=colors)
plt.title('Histogram of the 2D data')

##
# kernel density estimator; 

# create an evaluation grid; 
gridno = 40
inc1 = (max_data[0] - min_data[0]) / gridno
inc2 = (max_data[1] - min_data[1]) / gridno

array1 = np.arange(min_data[0], max_data[0], inc1)
array2 = np.arange(min_data[1], max_data[1], inc2)
array1 = np.append(array1, np.array(max_data[0]))
array2 = np.append(array2, np.array(max_data[1]))

array1 = array1.reshape(1, array1.shape[0])
array2 = array2.reshape(1, array2.shape[0])

gridx, gridy = np.meshgrid(array1, array2)

# reshape everything to fit in one matrix;
gridxr = gridx.ravel(order='F')
gridyr = gridy.ravel(order='F')
gridxr = gridxr.reshape(gridxr.shape[0], 1)
gridyr = gridyr.reshape(gridyr.shape[0], 1)
gridall = np.concatenate((gridxr, gridyr), axis = 1)
v  
gridallno = gridall.shape[0]

norm_pdata = np.sum(np.power(pdata,2), 1)
norm_gridall = np.sum(np.power(gridall,2), 1)
cross = pdata.dot(gridall.T)

# compute squared distance between each data point and the grid point; 
norm_pdata = norm_pdata.reshape(norm_pdata.shape[0], 1)
dist2 = np.tile( norm_pdata, (1, gridallno)) + \
        np.tile( norm_gridall.T, (datano, 1)) - 2 * cross

# choose kernel bandwidth 1; please also experiment with other bandwidth; 
bandwidth = 1
# evaluate the kernel function value for each training data point and grid
# point; 
kernelvalue = np.exp(-dist2 / np.power(bandwidth, 2))

# sum over the training data point to the density value on the grid points;
# here I dropped the normalization factor in front of the kernel function,
# and you can add it back. It is just a constant scaling; 
mkde = np.sum(kernelvalue, 0) / datano

# reshape back to grid; 
mkde = mkde.reshape(gridno + 1, gridno + 1)

# plot density as surface; 
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(gridx, gridy, mkde, cmap='coolwarm', rstride=1, cstride=1)
plt.gca().invert_xaxis()
plt.title('kernel density estimator')

print 'Please close all the figures to continue EM!'
plt.show()
raw_input('press key to run em ...\n')

##
# em algorithm for fitting mixture of gaussians; 

# fit a mixture of 3 gaussians; 
K = 3
# randomly initialize the paramters; 
# mixing proportion; 
pi = np.random.rand(K, 1)
pi = pi / np.sum(pi)
# mean or center of gaussian; 
mu = np.random.randn(2, K)
# covariance, and make sure it is positive semidefinite; 
sigma = np.zeros((2, 2, K))
for i in range(0, K):
    tmp = np.random.randn(2, 2)
    sigma[:, :, i] = tmp.dot(tmp.T)
# poster probability of component indicator variable; 
tau = np.zeros((datano, K))

# we just choose to run 100 iterations, but you can change the termination
# criterion for the loop to whether the solution changes big enough between
# two adjacent iterations; 
iterno = 100
plt.figure(5)
plt.ion
for it in range(0, iterno):
    
    print "--iteration %d of %d\n" % (it + 1,iterno)  
    # alternate between e and m step;
    
    # E-step; 
    for i in range(0, K):
        try:
            tau[:, i] = pi[i] * multivariate_normal.pdf(pdata, mean = mu[:,i].T, cov = sigma[:,:,i])
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.message:
                print 'Warning: singular matrix for mvnpdf!'
                continue
                   
    sum_tau = np.sum(tau, 1)
    sum_tau = sum_tau.reshape(sum_tau.shape[0], 1)
    # normalize
    tau = tau / np.tile( sum_tau, (1, K))  
    
    # M-step
    for i in range(0, K):
        # update mixing proportion; 
        pi[i] = np.sum(tau[:, i], 0) / datano
        # update gaussian center; 
        mu[:, i] = pdata.T.dot(tau[:,i]) / np.sum(tau[:, i], 0)
        mu_i = mu[:, i].reshape(mu[:, i].shape[0], 1)
        # update gaussian covariance;
        tmpdata = pdata - np.tile( mu_i.T, (datano, 1)) 
        
        sigma[:,:,i] = tmpdata.T.dot(np.diag(tau[:,i])).dot(tmpdata) / np.sum(tau[:,i], 0) 
        
    # plot data points using the mixing proportion tau as colors;    
    # the data point locations will not change over iterations, but the
    # color may change; 
    
    plt.scatter(pdata[:, 0], pdata[:, 1], s = 16 * np.ones((datano, 1)), c = tau)
    plt.hold(True)
    # also plot the centers of the guassian;
    # the centers change locations each iteraction until the solution converges;  
    plt.scatter(mu[0, :].T, mu[1, :].T, s = 26 * np.ones((K, 1)))
    plt.draw()
    
    # also draw the contour of the fitted mixture of gaussian density; 
    # first evaluate the density on the grid points;
    tmppdf = np.zeros((gridall.shape[0], 1))
    for i in range(0, K):
        try:      
            temp = pi[i] * multivariate_normal.pdf(gridall, mean = mu[:,i].T, cov = sigma[:,:,i])
            temp = temp.reshape(temp.shape[0], 1)
            tmppdf = tmppdf + temp
        except np.linalg.linalg.LinAlgError as err:
            if 'Singular matrix' in err.message:
                print 'Warning: singular matrix for mvnpdf!'
                continue     
        
    tmppdf = tmppdf.reshape(gridno + 1, gridno + 1)
    
    # draw contour;
    plt.contour(gridx, gridy, tmppdf.T)
    plt.title('Fitting Gaussian using EM algorithm')
    plt.hold(False)
    plt.draw()
    plt.pause(0.001)

plt.show()