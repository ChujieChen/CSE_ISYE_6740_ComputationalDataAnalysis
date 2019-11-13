function [ class, centroid ] = mykmedoids( pixels, K )
%
% Your goal of this assignment is implementing your own K-medoids.
% Please refer to the instructions carefully, and we encourage you to
% consult with other resources about this algorithm on the web.
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
%     cluster. The output should be a matrix with size(pixels, 1) rows and
%     3 columns. The range of values should be [0, 255].
%     
%
% You may run the following line, then you can see what should be done.
% For submission, you need to code your own implementation without using
% the kmeans matlab function directly. That is, you need to comment it out.

	%[class, centroid] = kmeans(pixels, K);
    
options = optimoptions('fmincon','Display','off');

function f = distancef(p)
    if p<inf
         f = @(x,y) sum(abs(x-y).^p);
    else
         f = @(x,y) max(abs(x-y));
    end
end

function f = obj_c(x,p) % function used to update center value
    if p<inf
         f = @(c) sum(sum(abs(bsxfun(@minus,x,c)).^p))/m;
    else
         f = @(c) sum(max(abs(bsxfun(@minus,x,c))))/m;
    end
end    
    
cno = K; % number of centers
% randomly initialize centroids with data points; 
x = pixels';%xrow is point dim=3, xcol is number of samples
% number of data points to work with; 
m = size(x, 2); 
c = x(:,randsample(size(x,2),cno)); % size(c)=3*K
%c = ones(size(x,1),cno)*255; % extreme intitialiation

%distance description, here p=1,2,...,inf;
p= 1;

distance = distancef(p);
c_old = c*0; iter=1; labels=zeros(m,1); obj=inf;
while (norm(c - c_old, 'fro') > 1e-4)
    fprintf('--iteration %d; --obj %.4f\n', iter,obj); 
    
    % record previous c; 
    c_old = c; 
    
    % assign data points to current cluster; 
    for j = 1:m % loop through data points; 
        tmp_distance = zeros(1, cno); 
        for k = 1:cno % through centers; 
            tmp_distance(k) = distance(x(:,j), c(:,k)); 
            %tmp_distance(k) = sum((x(:,j) - c(:,k)).^2);
        end
        [~,k_star] = min(tmp_distance); % ~ ignores the first argument; k_star stores the label that x(:,j) belongs to
        P(j, :) = zeros(1, cno); 
        P(j, k_star) = 1; 
        labels(j) = k_star;
    end
        
    % adjust the cluster centers according to current assignment;     
    obj = 0; 
    for k = 1:cno
        idx = find(P(:,k)>0); % idx corresponds to the points that belong to k
        nopoints = length(idx);  
        if (nopoints == 0) 
            % a cener has never been assigned a data point; 
            % re-initialize the center; 
            c(:,k) = x(:,randsample(size(x,2),1));  
        else
            % use fmincon to find the optimal center under norm p     
            f = obj_c(x(:,idx),p);
            c(:,k) = fmincon(f, c(:,k),[],[],[],[],min(x(:,idx),[],2),max(x(:,idx),[],2),[],options);
            %c(:,k) = x * P(:,k) ./ nopoints;   
        end
        obj = obj + sum(sum((x(:,idx) - repmat(c(:,k), 1, nopoints)).^2)); 
    end
    
    iter = iter+1; 
end   

class = labels;
centroid = c';     
    
    
end