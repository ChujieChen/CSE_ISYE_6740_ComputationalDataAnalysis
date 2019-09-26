#############################################################################
## k-means algorithm; 
# greed algorithm trying to minimize the objective function; 
# A highly matricized version of kmeans.
# --------------------------------------------------------------------------
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
# @author CSE6740/CS7641 TA
#############################################################################
## run kmeans
import numpy as np
from scipy.sparse import csc_matrix

def kmeans_python(x, k):
    x = x.T
    cno = k 
    m = x.shape[1]
    
    # randomly initialize centroids with data points; 
    c = x[:, np.random.permutation(x.shape[1])[0:cno]]
            
    iterno = 101;
    for iter in range(1, iterno):
        print "--iteration %d \n" % iter
                   
        # norm squared of the centroids; 
        c2 = np.sum(np.power(c, 2), axis = 0, keepdims = True);
        
        # for each data point, computer max_j -2 * x' * c_j + c_j^2;
        tmpdiff = (2 * np.dot(x.T,c) - c2)
        labels = np.argmax(tmpdiff, axis = 1)
       
        # Update data assignment matrix;
        P = csc_matrix( (np.ones(m) ,(np.arange(0,m,1), labels)), shape=(m, cno) )
        count = P.sum(axis=0)
        
        # Recompute centroids; 
        if 0 in count:
            c = x[:, np.random.permutation(x.shape[1])[0:cno]]
        else:
            c = np.array((P.T.dot(x.T)).T / count)
               
    return labels    