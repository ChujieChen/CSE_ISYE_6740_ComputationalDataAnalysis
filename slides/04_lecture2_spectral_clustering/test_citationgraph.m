clc; clear; close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPECTRAL CLUSTERING ALGORITHM
% ==============================
% 
% Type of Algorithm:
% -------------------
% Normalized Laplacian (Ref: Ng. et. al. 2001)
%
% 
% DATASET:
% --------
% A citation graph containing information about the citations of papers.  
% It is expected that papers linked by citation form good clusters.  
%
% INPUT FILES: 
% ------------
% citation_graph.txt - Contains edges between nodes in graph. Nodes are 
%                      papers. If there is an edge, the source was cited 
%                      by target. 
%                      Format: Source_node Target_node
% title_inverse.txt - Contains paper names for each node. Serves as labels. 
%                     Extracted from dataset. Format: Node_Id Paper_Name
% OUTPUT FILES:
% -------------
% clusters.txt      - Contains cluster id and all the papers belonging to
%                     that cluster.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize no. of nodes n and no. of clusters k 

k = 100;
n = 27694; % number of papers

% load graph G
G = dlmread('citation_graph.txt');

% Creat Adjacency Matrix A (size:nxn) from graph G 
A = sparse(G(:, 1), G(:, 2), ones(size(G, 1), 1), n, n);

% Convert the directed graph G into its undirected version. Makes the
% matrix A symmetric.
A = (A + A') / 2;

n = size(A, 1);
D = spdiags(1 ./ sqrt(sum(A, 1))', 0, n, n); % Degree Matrix (normalized)
L = D * A * D; % Compute Laplacian

fprintf('performing randomized eigendecomposition ...\n'); 
tic
[X, V] = rand_eig(L, k); % Compute k largest eigenvalues and eigenvec. of L
toc
X = bsxfun(@rdivide, X, sqrt(sum(X .* X, 2))); % Normalize X row-wise
fprintf('performing our vectorized kmeans ...\n'); 
tic 
c_idx = kmeans_mat(X, k); % Partition X by k-means
toc

% Clustering algorithm End

%% Get node labels (paper names)
fid = fopen('title_inverse_index.txt');

idx2names = cell(1, 1);

tline = fgetl(fid);
count = 1;
while ischar(tline)
    sep = strfind(tline, '	');
    idx = str2double(tline(1:sep-1));
    team_name = tline(sep+1:end);
    idx2names{count} = team_name;
    count = count + 1;
    tline = fgetl(fid);
end
fclose(fid);

%% Output team names partitioned by clusters obtained from above
fid = fopen('clusters.txt', 'w');
for i = 1:k
    fprintf(fid, 'Cluster %i\n***************\n', i);
    idx = c_idx == i;
    content = sprintf('%s\n', idx2names{idx});
    fprintf(fid, '%s\n', content);
end
fclose(fid);
