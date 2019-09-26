#########################################################################
# SPECTRAL CLUSTERING ALGORITHM
# ==============================
# 
# Types of Algorithm:
# -------------------
# [A] Un-normalized Laplacian 
# [B] Normalized Laplacian (Ref: Ng. et. al. 2001)
#
#
# DATASET:
# --------
# URL: http://www.cbssports.com/collegefootball/schedules/FBS/week1 
#
# The URL contains the schedules for NCAA FBS college football teams. 
# College football teams form conferences, and intra-conference games are
# played more frequently than inter-conference games.
#
# Therefore, the groups of teams that form conferences corresponds to 
# natural clusters.
#
# INPUT FILES:
# ------------
# play_graph.txt - Contains edges between nodes in graph. Extracted from
#                  dataset. Format: Source_node Target_node
# inverse_teams.txt - Contains team names for each node. Serves as labels. 
#                     Extracted from dataset. Format: Node_Id Team_Name
# OUTPUT FILES:
# -------------
# nodes.csv      - Contains node no. and cluster id to which node belongs 
# edges.csv      - Edges between nodes in the graph
# norm_nodes.csv - Same as nodes.csv for normalized version
# norm_edges.csv - Same as edges.csv for normalized version
#
# --------------------------------------------------------------------------
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib packages before using.
# Thank you for your suggestions!
#
# @version 1.0
# @author CSE6740/CS7641 TA
#########################################################################
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import kmeans
import csv

# Initialize no. of nodes n and no. of clusters k 
n = 321
k = 25

# Load Graph G
G = np.loadtxt('play_graph.txt')

# Creat Adjacency Matrix A (size:nxn) from graph G
A = csc_matrix((np.ones(G.shape[0]) ,(G[:,0] - 1, G[:,1] - 1)), shape=(n, n))

# Convert the directed graph G into its undirected version. Makes the
# matrix A symmetric.
A = (A + A.T) / 2

## Clustering Algorithm
# k eigenvectors are computed and then k-means is used to cluster points by
# their respective k components in these eigenvectors
#
# 'replicates' argument in k-means uses 20 different centroid initializa-
# tions to find the local minima

# [A] Unnormalized case
# c_idx_un: Clusters identified using unnormalized Laplacian version

# Degree Matrix
sumMat = np.array(A.sum(axis=0)).reshape((A.shape[0],))
D = csc_matrix((sumMat ,(np.arange(0,A.shape[0],1), np.arange(0,A.shape[1],1))))
L = D - A # Compute Laplacian

S, V = np.linalg.eig(np.array(L.todense()).squeeze()) # Compute k smallest eigenvalues and eigenvec. of L
sortidx = S.argsort()                        
V = V[:,sortidx]
    
c_idx_un = kmeans.kmeans_python(V[:,0:k], k) # Partition X by k-means

# [B] Normalized case
# c_idx: Clusters identified using normalized Laplacian version

# Degree Matrix (normalized) 
D = np.diag(1 / np.sqrt(sumMat)) 
L = csc_matrix.dot(csr_matrix.tocsc(A.T),D.T).T # Compute Laplacian
L = L.dot(D)
S, V = np.linalg.eig(L) # Compute k largest eigenvalues and eigenvec. of L 
sortidx = S.argsort()[::-1] 
X = V[:,sortidx][:,0:k]
norm2 = np.power(X, 2).sum(axis = 1) # Normalize X row-wise
norm2.shape = (norm2.shape[0], 1)
X = X / (np.sqrt(norm2)) 
c_idx = kmeans.kmeans_python(X, k) # Partition X by k-means

# Clustering algorithm End


## Get node labels
idx2names = {};
for line in open('inverse_teams.txt'):
    (index, name) = line.split("\t")
    idx2names[index] = name.replace('\n','')

## Print team names partitioned by clusters obtained from above
# Unnormalized Case
print 'Unnormalized Case\n************\n'
for i in range(0, k):
    print 'Cluster %i\n***************\n' % (i + 1)
    idx = np.where(c_idx_un == i)[0]
    idx = idx + 1
        
    for j in idx:
        print '%s\t' % idx2names[str(j)],
    print "\n"

# Normalized Case
print '\n\nNormalized Case\n***************\n'
for i in range(0, k):
    print 'Cluster %i\n***************\n' % (i + 1)
    idx = np.where(c_idx == i)[0]
    idx = idx + 1
    
    for j in idx:
        print '%s\t' % idx2names[str(j)],
    print "\n"

## Write graph and cluster information to output files for visualization
# Unnormalized case
un_node_file = 'nodes.csv'
un_edge_file = 'edges.csv'

with open(un_edge_file, 'wb') as fid:
    writer = csv.writer(fid)
    writer.writerow(['Source', 'Target'])
    for i in range(0, G.shape[0]):
        writer.writerow([G[i,0], G[i,1]])
fid.close()
        
with open(un_node_file, 'wb') as fid:
    writer = csv.writer(fid)
    writer.writerow(['Id', 'Label', 'Color'])
    for i in range(0, len(idx2names)):
        writer.writerow([i + 1, idx2names[str(i + 1)], c_idx_un[i] + 1])
fid.close()

# Normalized case
node_file = 'norm_nodes.csv'
edge_file = 'norm_edges.csv'

with open(edge_file, 'wb') as fid:
    writer = csv.writer(fid)
    writer.writerow(['Source', 'Target'])
    for i in range(0, G.shape[0]):
        writer.writerow([G[i,0], G[i,1]])
fid.close()

with open(node_file, 'wb') as fid:
    writer = csv.writer(fid)
    writer.writerow(['Id', 'Label', 'Color'])
    for i in range(0, len(idx2names)):
        writer.writerow([i + 1, idx2names[str(i + 1)], c_idx[i] + 1])
fid.close()