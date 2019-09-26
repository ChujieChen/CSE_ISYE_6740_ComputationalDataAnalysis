%%

clear; 
close all; 

load swiss_roll_data

x = X_data(:,1:10:end); 
y = Y_data(:,1:10:end); 

% number of data points to work with; 
m = size(x, 2); 

figure()

scatter3(x(1,:), x(2,:), x(3,:), 18*ones(1,m), y(1,:), 'fill');
G = sqrt(sum(x.^2,1)'*ones(1,m) + ones(m,1)*sum(x.^2,1) - 2*(x'*x));
e = 0.2*median(G(:));  
G(G>e) = 0; 
sG = sum(G, 1); 

% get rid of Inf distance for simplicity; 
i = find(sG == 0); 
idx = setdiff((1:m), i); 
G = real(G(idx,idx)); 
m = size(G, 1); 
% figure()
% spy(sparse(G)); 
% drawnow; 

D = graphallshortestpaths(sparse(G), 'directed', 'false'); 
D2 = D.^2; 

H = eye(m) - ones(m,1)*ones(m,1)'./m; 

Dt = -0.5 * H * D2 * H; 

k = 10; 
[V, S, U] = svds(Dt, k);

dim1 = V(:,1) * sqrt(S(1,1)); 
dim2 = V(:,2) * sqrt(S(2,2)); 

figure() 
scatter(dim1, dim2, 18*ones(1,m), y(1,:), 'fill');
