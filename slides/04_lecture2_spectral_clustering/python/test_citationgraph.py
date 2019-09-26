############################################################################
# SPECTRAL CLUSTERING ALGORITHM
# ==============================
#
# Type of Algorithm:
# -------------------
# Normalized Laplacian (Ref: Ng. et. al. 2001) 
#
#
# DATASET:
# --------
# A citation graph containing information about the citations of papers.  
# It is expected that papers linked by citation form good clusters.
#
# INPUT FILES: 
# ------------
# citation_graph.txt - Contains edges between nodes in graph. Nodes are 
#                      papers. If there is an edge, the source was cited 
#                      by target.
#                      Format: Source_node Target_node
# title_inverse.txt - Contains paper names for each node. Serves as labels. 
#                     Extracted from dataset. Format: Node_Id Paper_Name
# OUTPUT FILES:
# -------------
# clusters.txt      - Contains cluster id and all the papers belonging to
#                     that cluster.
# --------------------------------------------------------------------------
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
# @author CSE6740/CS7641 TA
############################################################################
import numpy as np
from scipy.sparse import csc_matrix, spdiags
import kmeans
import time
import rand_eig

# Initialize no. of nodes n and no. of clusters k 
k = 100
n = 27694 # number of papers

# load graph G
G = np.loadtxt('citation_graph.txt')

# Creat Adjacency Matrix A (size:nxn) from graph G
A = csc_matrix((np.ones(G.shape[0]) ,(G[:,0] - 1, G[:,1] - 1)), shape=(n, n))

# Convert the directed graph G into its undirected version. Makes the
# matrix A symmetric.
A = (A + A.T) / 2

sumMat = np.array(A.sum(axis=0)).reshape((A.shape[0],))
D = spdiags(1 / np.sqrt(sumMat), 0, n, n) # Degree Matrix (normalized)
L = D.dot(A) # Compute Laplacian
L = L.dot(D)

print 'performing randomized eigendecomposition ...\n'
tic = time.time()
X, V = rand_eig.rand_eig_function(L, k)
toc = time.time()
print 'Elapsed time (Rand_Eig) is %f seconds \n' % float(toc - tic)


norm2 = np.power(X, 2).sum(axis = 1) # Normalize X row-wise
norm2.shape = (norm2.shape[0], 1)
X = X / (np.sqrt(norm2)) 
print 'performing our vectorized kmeans ...\n'
tic = time.time()
c_idx = kmeans.kmeans_python(X, k) # Partition X by k-means
toc = time.time()
print 'Elapsed time (k_mean) is %f seconds \n' % float(toc - tic)

# Clustering algorithm End


## Get node labels (paper names)
idx2names = {};
for line in open('title_inverse_index.txt'):
    (index, name) = line.split("\t")
    idx2names[index] = name.replace('\n','')
    
## Output team names partitioned by clusters obtained from above
with open('clusters.txt', 'w') as fid:
    for i in range(0, k):
        fid.write('Cluster' + str(i + 1) + '\n***************\n')
        idx = np.where(c_idx == i)[0]
        idx = idx + 1
       
        for j in idx:
            fid.write( idx2names[str(j)] + '\n')
        fid.write('\n')
fid.close()