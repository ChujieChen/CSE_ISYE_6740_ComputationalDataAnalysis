######################################################################
#The codes are based on Python2.7. 
#Please install numpy, scipy, matplotlib packages before using.
#Thank you for your suggestions!
#
#@version 1.0
#@author CSE6740/CS7641 TA
######################################################################
# ===============================================================
# Data preparation
# (Please copy and paste one section of the below.)
# ===============================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import multivariate_normal

X = np.loadtxt('background_scores.txt')
m = X.shape[0]
np.random.seed(1)

mean01 = np.mean(X[:,0:2], axis = 1)
mean01 = mean01.reshape(mean01.shape[0], 1)
X = np.concatenate((mean01, X[:,2].reshape(X[:,2].shape[0], 1)), axis = 1) + \
     0.1 * np.random.randn(m, 2)
     
B = X 

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1] , marker = '+') 
plt.title('statistics vs linear algebra')
plt.xlabel('statistics score')
plt.ylabel('linear algebra score')

#raw_input('press any key to continue ...\n')

plt.figure(2)
plt.hist(B[:, 0], 20)
plt.xlabel('statistics score')
plt.figure(3)
plt.hist(B[:, 1], 20)
plt.xlabel('linear algebra score')

#raw_input('press any key to continue ...\n')

# ===============================================================
# Histogram for 2D data
# ===============================================================

# Adjustable parameters:
bincount = 10

mindata = np.min(B, axis=0)
maxdata = np.max(B, axis=0)
minlevel = np.floor(mindata)
maxlevel = np.ceil(maxdata)
leveldiff = (maxlevel - minlevel) / bincount

myhist3 = np.zeros((bincount, bincount, 1))
for i in range(0, B.shape[0]):
    bin = (B[i,:] - minlevel) / leveldiff
    bin = np.ceil(bin)
    myhist3[bin[0]-1, bin[1]-1, 0] = myhist3[bin[0]-1, bin[1]-1, 0] + 1

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')

xpos = np.arange(myhist3.shape[0])
ypos = np.arange(myhist3.shape[1])
xposM, yposM = np.meshgrid(xpos, ypos)
zpos = myhist3
zpos = zpos.ravel()
dx = 0.5
dy = 0.5   
dz = zpos

ax.w_xaxis.set_ticks(np.arange(1, bincount, bincount / 10))
ax.w_xaxis.set_ticklabels(np.arange(minlevel[1], maxlevel[1], leveldiff[1] * (bincount / 10)))
ax.w_yaxis.set_ticks(np.arange(1, bincount, bincount / 10))
ax.w_yaxis.set_ticklabels(np.arange(minlevel[0], maxlevel[0], leveldiff[0] * (bincount / 10)))

values = np.linspace(0, 1, xposM.ravel().shape[0])
colors = cm.rainbow(values)
plt.xlabel('linear algebra score')
plt.ylabel('statistics score')

ax.bar3d(yposM.ravel(), xposM.ravel(), dz * 0, dx, dy, dz, color=colors) 

#raw_input('press any key to continue ...\n')

# ===============================================================
# Kernel density estimation
# ===============================================================

bincount = 100
leveldiff = (maxlevel - minlevel) / bincount

# Adjustable parameters:
kernelwidth = 0.60

gridx, gridy = np.meshgrid( mindata[0] + leveldiff[0] * np.arange(1,bincount + 1), \
           mindata[1] + leveldiff[1] * np.arange(1,bincount + 1), copy = False)

norm2B = np.sum(np.power(B,2), 1)
norm2grid = np.power(gridx.ravel(order='F'),2) + np.power(gridy.ravel(order='F'),2)

norm2B = norm2B.reshape(norm2B.shape[0], 1)
norm2grid = norm2grid.reshape(norm2grid.shape[0], 1)

gridxr = gridx.ravel(order='F')
gridyr = gridy.ravel(order='F')
gridxr = gridxr.reshape(gridxr.shape[0], 1)
gridyr = gridyr.reshape(gridyr.shape[0], 1)
dist2mat = norm2B + norm2grid.T - 2*B.dot(np.concatenate((gridxr, gridyr), axis = 1).T)

# Using Gaussian Kernel
kernelmatrix = np.exp(  -dist2mat/(2 * np.power(kernelwidth,2))  ) /   \
                (np.power(kernelwidth, 2) * (2 * np.pi))

mykde2 = np.mean(kernelmatrix, axis = 0)

fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')

X1 = np.arange(bincount)
Y = np.arange(bincount)
X1, Y = np.meshgrid(X1, Y, copy = False)
Z = mykde2.reshape(bincount, bincount).T
ax.plot_surface(X1, Y, Z, cmap='coolwarm', rstride=2, cstride=2)

ax.w_xaxis.set_ticks(np.arange(1, bincount, bincount / 10))
ax.w_xaxis.set_ticklabels(np.arange(minlevel[1], maxlevel[1], leveldiff[1] * (bincount / 10)))
ax.w_yaxis.set_ticks(np.arange(1, bincount, bincount / 10))
ax.w_yaxis.set_ticklabels(np.arange(minlevel[0], maxlevel[0], leveldiff[0] * (bincount / 10)))   
plt.gca().invert_xaxis()
plt.xlabel('linear algebra score')
plt.ylabel('statistics score')
plt.title('kernel density estimator')

#raw_input('press any key to continue ...\n')

# overlay contour plot with density. 
plt.figure(6)
plt.scatter(X[:, 0], X[:, 1] , marker = '+') 
plt.hold(True)
plt.contour(gridx, gridy, mykde2.reshape(bincount, bincount))
plt.title('statistics vs linear algebra')
plt.xlabel('statistics score')
plt.ylabel('linear algebra score')

#raw_input('press any key to continue ...\n')

# fit with Gaussian cdf
mu = np.mean(B, 0)
sigma = np.cov(B.T)

tmppdf = multivariate_normal.pdf(np.concatenate((gridxr, gridyr), axis = 1), mean = mu, cov = sigma)
plt.figure(7)
plt.scatter(B[:, 0], B[:, 1] , marker = '+')
plt.title('statistics vs linear algebra (Gaussian)')
plt.hold(True)
plt.contour(gridx, gridy, tmppdf.reshape(bincount, bincount))
plt.xlabel('statistics score')
plt.ylabel('linear algebra score')

fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')
X1 = np.arange(bincount)
Y = np.arange(bincount)
X1, Y = np.meshgrid(X1, Y)
Z = tmppdf.reshape(bincount, bincount).T
ax.plot_surface(X1, Y, Z, cmap='coolwarm', rstride=2, cstride=2)

ax.w_xaxis.set_ticks(np.arange(1, bincount, bincount / 10))
ax.w_xaxis.set_ticklabels(np.arange(minlevel[1], maxlevel[1], leveldiff[1] * (bincount / 10)))   
ax.w_yaxis.set_ticks(np.arange(1, bincount, bincount / 10))
ax.w_yaxis.set_ticklabels(np.arange(minlevel[0], maxlevel[0], leveldiff[0] * (bincount / 10)))

plt.gca().invert_xaxis()
plt.xlabel('linear algebra score')
plt.ylabel('statistics score')
plt.title('Gaussian density')
plt.show()