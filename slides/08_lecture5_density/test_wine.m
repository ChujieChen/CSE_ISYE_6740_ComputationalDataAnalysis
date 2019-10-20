%% 
% load wine dataset which is in csv format; 
clear ; close all
data = csvread('wine.data'); 
y = data(:,1); 
data = data(:,2:end); 

%% pca the data;
[ndata, mu, sigma] = zscore(data); 
covariance = cov(ndata); 
d = 2; 
[V, S] = eigs(covariance, d); 

% project the data to the top 2 principal directions;
pdata = ndata * V;

% visualize the data; 
figure;
scatter(pdata(y==1,1), pdata(y==1,2), 'r'); hold on;
scatter(pdata(y==2,1), pdata(y==2,2), 'b');
scatter(pdata(y==3,1), pdata(y==3,2), 'g');

%%
% histogram for first dimension of pdata; 
% find the range of the data; 
datano = size(pdata, 1); 

min_data = min(pdata(:,1)); 
max_data = max(pdata(:,1)); 
nbin = 10; 
sbin = (max_data - min_data) / nbin; 
% create the bins; 
boundary = (min_data:sbin:max_data);

% just loop over the data points, and count how many of data points are in
% each bin; 
myhist = zeros(nbin, 1); 
for i = 1:datano
    which_bin = max(find(pdata(i,1) > boundary));
    myhist(which_bin) = myhist(which_bin) + 1; 
end
myhist = myhist * nbin ./ datano; 

% bar plot; 
figure; 
bar(myhist); 
title('Histogram');

%%
% for 2 dimensional data; 

min_data = min(pdata, [], 1); 
max_data = max(pdata, [], 1); 
nbin = 30; % you can change the number of bins in each dimension; 
sbin = (max_data - min_data) ./ nbin; 
% boundary = (min_data:sbin:max_data);
% create the bins; 
boundary = [min_data(1):sbin(1):max_data(1); min_data(2):sbin(2):max_data(2)]

% just loop over the data points, and count how many of data points are in
% each bin; 
myhist2 = zeros(nbin, nbin);
for i = 1:datano
    which_bin1 = max(find(pdata(i,1) > boundary(1,:)));
    which_bin2 = max(find(pdata(i,2) > boundary(2,:)));
    myhist2(which_bin1, which_bin2) = myhist2(which_bin1, which_bin2) + 1; 
end
myhist2 = myhist2 * nbin ./ datano; 

% two dimensional bar plot; 
figure; 
bar3(myhist2);
title('Histogram of the 2D data');

%% 
% kernel density estimator; 

% create an evaluation grid; 
gridno = 40; 
inc1 = (max_data(1) - min_data(1)) / gridno; 
inc2 = (max_data(2) - min_data(2)) / gridno; 
[gridx,gridy] = meshgrid(min_data(1):inc1:max_data(1), min_data(2):inc2:max_data(2)); 

% reshape everything to fit in one matrix;
gridall = [gridx(:), gridy(:)]; 
gridallno = size(gridall, 1); 

norm_pdata = sum(pdata.^2, 2); 
norm_gridall = sum(gridall.^2, 2); 
cross = pdata * gridall'; 

% compute squared distance between each data point and the grid point; 
dist2 = repmat(norm_pdata, 1, gridallno) + repmat(norm_gridall', datano, 1) ...
    - 2 * cross; 

% choose kernel bandwidth 1; please also experiment with other bandwidth; 
bandwidth = 1; 
% evaluate the kernel function value for each training data point and grid
% point; 
kernelvalue = exp(-dist2 ./ bandwidth.^2); 

% sum over the training data point to the density value on the grid points;
% here I dropped the normalization factor in front of the kernel function,
% and you can add it back. It is just a constant scaling; 
mkde = sum(kernelvalue, 1) ./ datano; 

% reshape back to grid; 
mkde = reshape(mkde, gridno+1, gridno+1); 

% plot density as surface; 
figure; 
surf(gridx, gridy, mkde); 
title('Kernel Density Estimate');

input('press key to run em ...\n'); 

%%
% em algorithm for fitting mixture of gaussians; 

% fit a mixture of 3 gaussians; 
K = 3; 
% randomly initialize the paramters; 
% mixing proportion; 
pi = rand(K,1); 
pi = pi./sum(pi); 
% mean or center of gaussian; 
mu = randn(2, K); 
% covariance, and make sure it is positive semidefinite; 
sigma = zeros(2, 2, K); 
for i = 1:K
    tmp = randn(2, 2); 
    sigma(:,:,i) = tmp * tmp'; 
end
% poster probability of component indicator variable; 
tau = zeros(datano, K); 

% we just choose to run 100 iterations, but you can change the termination
% criterion for the loop to whether the solution changes big enough between
% two adjacent iterations; 
iterno = 50; 
figure; 
for it = 1:iterno
    fprintf(1, '--iteration %d of %d\n', it, iterno); 
    % alternate between e and m step; 
    
    % E-step; 
    for i = 1:K
        tau(:,i) = pi(i) * mvnpdf(pdata, mu(:,i)', sigma(:,:,i)); 
    end
    sum_tau = sum(tau, 2); 
    % normalize
    tau = tau ./ repmat(sum_tau, 1, K);
        
    % M-step; 
    for i = 1:K
        % update mixing proportion; 
        pi(i) = sum(tau(:,i), 1) ./ datano; 
        % update gaussian center; 
        mu(:, i) = pdata' * tau(:,i) ./ sum(tau(:,i), 1); 
        % update gaussian covariance;
        tmpdata = pdata - repmat(mu(:,i)', datano, 1); 
        sigma(:,:,i) = tmpdata' * diag(tau(:,i)) * tmpdata ./ sum(tau(:,i), 1); 
    end
    
    % plot data points using the mixing proportion tau as colors; 
    % the data point locations will not change over iterations, but the
    % color may change; 
    scatter(pdata(:,1), pdata(:,2), 16*ones(datano, 1), tau, 'filled');     
    hold on; 
    % also plot the centers of the guassian; 
    % the centers change locations each iteraction until the solution converges;  
    scatter(mu(1,:)', mu(2,:)', 26*ones(K, 1), 'filled'); 
    drawnow; 
    
    % also draw the contour of the fitted mixture of gaussian density; 
    % first evaluate the density on the grid points; 
    tmppdf = zeros(size(gridall,1), 1);
    for i = 1:K        
        tmppdf = tmppdf + pi(i) * mvnpdf(gridall, mu(:,i)', sigma(:,:,i));
    end
    tmppdf = reshape(tmppdf, gridno+1, gridno+1); 
    
    % draw contour; 
    [c, h] = contour(gridx, gridy, tmppdf);
    title('Fitting Gaussian using EM algorithm');
    hold off; 
    
    pause(0.1);
end





























