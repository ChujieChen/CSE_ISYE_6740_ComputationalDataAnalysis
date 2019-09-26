clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPECTRAL CLUSTERING ALGORITHM
% ==============================
% 
% Types of Algorithm:
% -------------------
% [A] Un-normalized Laplacian 
% [B] Normalized Laplacian (Ref: Ng. et. al. 2001)
%
% 
% DATASET:
% --------
% URL: http://www.cbssports.com/collegefootball/schedules/FBS/week1 
% 
% The URL contains the schedules for NCAA FBS college football teams. 
% College football teams form conferences, and intra-conference games are 
% played more frequently than inter-conference games.
% 
% Therefore, the groups of teams that form conferences corresponds to 
% natural clusters.
%
% INPUT FILES: 
% ------------
% play_graph.txt - Contains edges between nodes in graph. Extracted from
%                  dataset. Format: Source_node Target_node
% inverse_teams.txt - Contains team names for each node. Serves as labels. 
%                     Extracted from dataset. Format: Node_Id Team_Name
% OUTPUT FILES:
% -------------
% nodes.csv      - Contains node no. and cluster id to which node belongs 
% edges.csv      - Edges between nodes in the graph
% norm_nodes.csv - Same as nodes.csv for normalized version
% norm_edges.csv - Same as edges.csv for normalized version
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%Initialize no. of nodes n and no. of clusters k 
n = 321;
k = 25;

% Load Graph G
G = dlmread('play_graph.txt');

% Creat Adjacency Matrix A (size:nxn) from graph G 
A = sparse(G(:, 1), G(:, 2), ones(size(G, 1), 1), n, n);

% Convert the directed graph G into its undirected version. Makes the
% matrix A symmetric.
A = (A + A') / 2;

%% Clustering Algorithm
% k eigenvectors are computed and then k-means is used to cluster points by
% their respective k components in these eigenvectors
%
% 'replicates' argument in k-means uses 20 different centroid initializa-
% tions to find the local minima

% [A] Unnormalized case
% c_idx_un: Clusters identified using unnormalized Laplacian version

D = sparse(1:size(A,1),1:size(A,2),sum(A,1)); % Degree Matrix 
L = D - A; % Compute Laplacian
[X, ~] = eig(full(L)); % Compute k smallest eigenvalues and eigenvec. of L  
c_idx_un = kmeans(X(:,1:k), k, 'replicates', 20); % Partition X by k-means 

% [B] Normalized case
% c_idx: Clusters identified using normalized Laplacian version

D = diag(1 ./ sqrt(sum(A, 1))); % Degree Matrix (normalized) 
L = D * A * D; % Compute Laplacian
[X, ~] = eigs(L, k); % Compute k largest eigenvalues and eigenvec. of L
X = bsxfun(@rdivide, X, sqrt(sum(X .* X, 2))); % Normalize X row-wise
c_idx = kmeans(X, k, 'replicates', 20); % Partition X by k-means

% Clustering algorithm End

%% Get node labels
fid = fopen('inverse_teams.txt');
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


%% Print team names partitioned by clusters obtained from above
% Unnormalized Case
fprintf('Unnormalized Case\n************\n');
for i = 1:k
    fprintf('Cluster %i\n***************\n', i);
    idx = c_idx_un == i;
    content = sprintf('%s\t', idx2names{idx});
    fprintf('%s\n', content);
end

% Normalized Case
fprintf('\n\nNormalized Case\n***************\n');
for i = 1:k
    fprintf('Cluster %i\n***************\n', i);
    idx = c_idx == i;
    content = sprintf('%s\t', idx2names{idx});
    fprintf('%s\n', content);
end


%% Write graph and cluster information to output files for visualization

% Unnormalized case
un_node_file = 'nodes.csv';
un_edge_file = 'edges.csv';

fid = fopen(un_edge_file, 'w');
fprintf(fid, 'Source\tTarget\n');
for i = 1:size(G, 1)
    fprintf(fid, '%i\t%i\n', G(i, 1), G(i, 2));
end
fclose(fid);

fid = fopen(un_node_file, 'w');
fprintf(fid, 'Id\tLabel\tColor\n');
for i = 1:length(idx2names)
    fprintf(fid, '%i\t%s\t%i\n', i, idx2names{i}, c_idx_un(i));
end
fclose(fid);

% Normalized case
node_file = 'norm_nodes.csv';
edge_file = 'norm_edges.csv';

fid = fopen(edge_file, 'w');
fprintf(fid, 'Source\tTarget\n');
for i = 1:size(G, 1)
    fprintf(fid, '%i\t%i\n', G(i, 1), G(i, 2));
end
fclose(fid);

fid = fopen(node_file, 'w');
fprintf(fid, 'Id\tLabel\tColor\n');
for i = 1:length(idx2names)
    fprintf(fid, '%i\t%s\t%i\n', i, idx2names{i}, c_idx(i));
end
fclose(fid);