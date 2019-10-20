######################################################################
#The codes are based on Python2.7. 
#Please install numpy, scipy, matplotlib packages before using.
#Thank you for your suggestions!
#
#@version 1.0
#@author CSE6740/CS7641 TA
######################################################################
# Histogram and KDE example with mixture of gaussians
# In this example we have a randomly generated data derived from 3 normal
# distribution centred around 3 different points. We show the density
# estimation using both histogram methond and KDE
# ===============================================================
# Data preparation
# (Please copy and paste one section of the below.)
# =============================================================== 

#  When each gaussian component does not overlap:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# When each gaussian component does not overlap:
A = np.zeros((300, 1))
A[0:100] = np.random.randn(100, 1) * 0.1 + 5
A[100:200] = np.random.randn(100, 1) * 0.1 + 3
A[200:300] = np.random.randn(100, 1) * 0.1 + 7

# When each gaussian component overlaps a lot:
A[0:100] = np.random.randn(100, 1) * 1 + 5
A[100:200] = np.random.randn(100, 1) * 1 + 3
A[200:300] = np.random.randn(100, 1) * 1 + 7

# 3D example
B = np.zeros((300, 2))
B[0:100,:]= np.random.randn(100,2) + np.tile(np.array([4,10]),(100,1))
B[100:200,:]= np.random.randn(100,2) + np.tile(np.array([10,4]),(100,1))
B[200:300,:]= np.random.randn(100,2) + np.tile(np.array([15,15]),(100,1))

# ===============================================================
# Histogram method
# ===============================================================
# Adjustable parameters:
bincount = 50

myhist = np.zeros((bincount,1))
mindata = np.min(A)
maxdata = np.max(A)
minlevel = np.floor(mindata)
maxlevel = np.ceil(maxdata)
leveldiff = (maxlevel - minlevel) / bincount

for i in range(0, A.shape[0]):
    bin = (A[i,0] - minlevel) / leveldiff
    bin = np.ceil(bin)
    myhist[bin - 1,0] = myhist[bin - 1,0] + 1

plt.figure(1) 
bins = range(myhist.shape[0])
plt.bar(bins, myhist[:,0])
ax=plt.gca()
ax.set_xticks(np.arange(1, bincount, bincount / 10)) 
ax.set_xticklabels(np.arange(minlevel, maxlevel, leveldiff * (bincount / 10)))

# ===============================================================
# Kernel density estimation
# ===============================================================
# Here we use the tophat kernel.
#  Adjustable parameters:

kernelwidth = 0.2 # kernel width determines how smooth estimate will be

bincount = 1000
mykde = np.zeros((bincount,1))
mindata = np.min(A)
maxdata = np.max(A)
minlevel = np.floor(mindata)
maxlevel = np.ceil(maxdata)
leveldiff = (maxlevel - minlevel) / bincount

for i in range(0, A.shape[0]):   
    bin = (A[i,0] - minlevel) / leveldiff
    bin = np.ceil(bin)
    
    temprange = np.floor(kernelwidth / leveldiff)
    
    added = np.zeros((bincount, 1))
    minindex = np.max([1, bin - temprange])
    maxindex = np.min([bincount, bin + temprange])
    added[minindex - 1 : maxindex - 1, 0] = 1
    
    mykde = mykde + added #  tophat kernel

plt.figure(2) 
plt.plot(mykde)
ax=plt.gca()
ax.set_xticks(np.arange(1, bincount, bincount / 10)) 
ax.set_xticklabels(np.arange(minlevel, maxlevel, leveldiff * (bincount / 10)))

# ===============================================================
# Histogram for 3D
# ===============================================================

# Adjustable parameters:
bincount = 50

myhist3 = np.zeros((bincount,bincount,1))
mindata = np.min(B, axis=0)
maxdata = np.max(B, axis=0)
minlevel = np.floor(mindata)
maxlevel = np.ceil(maxdata)
leveldiff = (maxlevel - minlevel) / bincount

for i in range(0, B.shape[0]):
    bin = (B[i,:] - minlevel) / leveldiff
    bin = np.ceil(bin)
    myhist3[bin[0]-1, bin[1]-1, 0] = myhist3[bin[0]-1, bin[1]-1, 0] + 1
    
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
elements = ( myhist3.shape[0] - 1) * ( myhist3.shape[1] - 1)  

xpos = np.arange(myhist3.shape[0])
ypos = np.arange(myhist3.shape[1])
xposM, yposM = np.meshgrid(xpos, ypos, copy = False)
zpos = myhist3
zpos = zpos.ravel()
dx = 0.5
dy = 0.5   
dz = zpos

ax.w_xaxis.set_ticks(np.arange(1, bincount, bincount / 10))
ax.w_xaxis.set_ticklabels(np.arange(minlevel[0], maxlevel[0], leveldiff[0] * (bincount / 10)))
ax.w_yaxis.set_ticks(np.arange(1, bincount, bincount / 10))
ax.w_yaxis.set_ticklabels(np.arange(minlevel[1], maxlevel[1], leveldiff[1] * (bincount / 10)))   


values = np.linspace(0, 1, xposM.ravel().shape[0])
colors = cm.rainbow(values)

ax.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors) 
plt.title('Histogram for 2D data')

# ===============================================================
# Kernel density estimation
# ===============================================================

# Adjustable parameters:
kernelwidth = 0.5

bincount = 100
mindata = np.min(B, axis=0)
maxdata = np.max(B, axis=0)
minlevel = np.floor(mindata)
maxlevel = np.ceil(maxdata)
leveldiff = (maxlevel - minlevel) / bincount

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

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')

X = np.arange(bincount)
Y = np.arange(bincount)
X, Y = np.meshgrid(X, Y, copy = False)
Z = mykde2.reshape(bincount, bincount)
ax.plot_surface(X, Y, Z, cmap='coolwarm', rstride=2, cstride=2)

ax.w_xaxis.set_ticks(np.arange(1, bincount, bincount / 10))
ax.w_xaxis.set_ticklabels(np.arange(minlevel[0], maxlevel[0], leveldiff[0] * (bincount / 10)))
ax.w_yaxis.set_ticks(np.arange(1, bincount, bincount / 10))
ax.w_yaxis.set_ticklabels(np.arange(minlevel[1], maxlevel[1], leveldiff[1] * (bincount / 10)))   
plt.gca().invert_xaxis()
plt.title('kernel density estimator for 2D data')
plt.show()