function [labels, c] = kmeans_mat(x, k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% k-means algorithm; 
% greed algorithm trying to minimize the objective function; 
% A highly matricized version of kmeans.

%% run kmeans
x = x'; 
cno = k;
m = size(x, 2);

% randomly initialize centroids with data points; 
c = x(:, randsample(m, cno));

iterno = 100; 
for iter = 1:iterno
  fprintf('--iteration %d\n', iter); 
  
  % norm2 of the centroids; 
  c2 = sum(c.^2, 1);  
  
  % for each data point, computer max_j -2 * x' * c_j + c_j^2; 
  tmpdiff = bsxfun(@minus, 2*x'*c, c2); 
  [val, labels] = max(tmpdiff, [], 2); 
  
  % update data assignment matrix; 
  P = sparse(1:m, labels, 1, m, cno, m); 
  count = sum(P, 1); 
   
  % recompute centroids; 
  c = bsxfun(@rdivide, x*P, count); 
end