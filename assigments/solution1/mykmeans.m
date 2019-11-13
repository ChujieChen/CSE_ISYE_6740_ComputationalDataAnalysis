function [ class, centroid ] = mykmeans( pixels, K )
%
% Your goal of this assignment is implementing your own K-means.
%
% Input:
%     pixels: data set. Each row contains one data point. For image
%     dataset, it contains 3 columns, each column corresponding to Red,
%     Green, and Blue component.
%
%     K: the number of desired clusters. Too high value of K may result in
%     empty cluster error. Then, you need to reduce it.
%
% Output:
%     class: the class assignment of each data point in pixels. The
%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
%     of class should be either 1, 2, 3, 4, or 5. The output should be a
%     column vector with size(pixels, 1) elements.
%
%     centroid: the location of K centroids in your result. With images,
%     each centroid corresponds to the representative color of each
%     cluster. The output should be a matrix with size(K, 1) rows and
%     3 columns. The range of values should be [0, 255].
%     
%
% You may run the following line, then you can see what should be done.
% For submission, you need to code your own implementation without using
% the kmeans matlab function directly. That is, you need to comment it out.

	%[class, centroid] = kmeans(pixels, K);
    
cno = K; % number of centers
% randomly initialize centroids with data points; 
x = pixels';%xrow is point dim=3, xcol is number of samples
% number of data points to work with; 
m = size(x, 2); 
c = x(:,randsample(size(x,2),cno)); % size(c)=3*K

c_old = c*0; iter=1;
%objold = sum(sqrt(sum(x.^2,1)));
while (norm(c - c_old, 'fro') > 1e-6)
  fprintf('--iteration %d\n', iter); 
  c_old = c;
  % norm2 of the centroids; 
  c2 = sum(c.^2, 1);  
  
  % for each data point, computer min_j -2 * x' * c_j + c_j^2; 
  tmpdiff = bsxfun(@minus, 2*x'*c, c2);  % tmpdiff size is pixel*K, tmpdiff=2 * x' * c_j - c_j^2
  [val, labels] = max(tmpdiff, [], 2); % labels stores the closest center lable for each point, thus size is pixel*1
  
  % update data assignment matrix; 
  P = sparse(1:m, labels, 1, m, cno, m); %P is like r{nk} in the homework
  count = sum(P, 1); 
   
  % recompute centroids; 
  c = bsxfun(@rdivide, x*P, count); 
  
  xmc_square=zeros(K,m);
  for k = 1:K
      xmc_square(k,:)=sqrt(sum((x-repmat(c(:,k),1,m)).^2,1));
  end
  %obj = trace(xmc_square*P); % the ojective function to be minimized
  
  iter = iter+1;
end    

class = labels;
centroid = c';    
    
end

