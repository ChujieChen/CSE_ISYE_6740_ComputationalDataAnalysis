#############################################################################
# Randomized Eigen Computation
# --------------------------------------------------------------------------
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
# @author CSE6740/CS7641 TA
#############################################################################
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

def rand_eig_function(A, k):
    
    n = A.shape[1]
    
    Omega = np.random.rand(n, k) # returns n x k matrix
    Y = csr_matrix.dot(A, Omega)
    
    q, r = np.linalg.qr(Y) # Orthogonal-triangular decomposition (QR decomposition) 
    
    B = csc_matrix.dot((A.T),q).T # Compute Laplacian
    B = B.dot(q)
    
    S, U_ = np.linalg.eig(B)
    sortidx = S.argsort()[::-1] 
    X = U_[:,sortidx][:,0:k]
    
    U = q.dot(X)
    
    return U, S