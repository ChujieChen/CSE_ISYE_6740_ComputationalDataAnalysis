clear; close all; 

%===============================================================
% Data preparation
%  (Please copy and paste one section of the below.)
%===============================================================

X = load('background_scores.txt');
[m, ~] = size(X); 
randn('seed', 1); 
X = [mean(X(:,1:2), 2), X(:,3)] + 0.1*randn(m,2); 
B = X; 

figure;
scatter(X(:,1), X(:,2), '+'); title('statistics vs linear algebra'); 
xlabel('statistics score'); 
ylabel('linear algebra score'); 
saveas(gcf, 'stats_vs_linear', 'png'); 

input('press any key to continue ...\n'); 

figure; hist(B(:,1), 20); xlabel('statistics score'); 
saveas(gcf, 'stats_only', 'png'); 
figure; hist(B(:,2), 20); xlabel('linear algebra score'); 
saveas(gcf, 'linear_only', 'png'); 

input('press any key to continue ...\n'); 

%===============================================================
% Histogram for 2D data
%===============================================================

% Adjustable parameters:
bincount = 10;

mindata = min(B, [], 1);
maxdata = max(B, [], 1);
minlevel = floor(mindata);
maxlevel = ceil(maxdata);
leveldiff = (maxlevel - minlevel)/bincount;

bin = ceil(bsxfun(@rdivide, bsxfun(@minus, B, minlevel), leveldiff)); 
addpath('tensor_toolbox_2.6'); 
tmp = sptensor([bin, ones(m,1)], ones(m, 1), [bincount, bincount, m]); 
myhist3 = (bincount^2/prod(maxlevel - minlevel)) * sum(double(tmp), 3) ./ m; 

%% avoid for loops in matlab; 
% myhist3 = zeros(bincount,bincount,1);
% for i=1:size(B,1)
% 	bin = (B(i,:) - minlevel) ./ leveldiff;
% 	bin = ceil(bin);
% 	myhist3(bin(1),bin(2)) = myhist3(bin(1),bin(2)) + 1;
% end

figure; 
bar3(myhist3);
set(gca,'XTick',1:(bincount/10):bincount);
set(gca,'XTickLabel',minlevel(1):leveldiff(1)*(bincount/10):maxlevel(1));
set(gca,'YTick',1:(bincount/10):bincount);
set(gca,'YTickLabel',minlevel(2):leveldiff(2)*(bincount/10):maxlevel(2));
axis tight; 
xlabel('statistics score'); 
ylabel('linear algebra score'); 
saveas(gcf, 'histogram', 'png'); 

input('press any key to continue ...\n'); 

%===============================================================
% Kernel density estimation
%===============================================================

bincount = 100;
leveldiff = (maxlevel - minlevel)/bincount;

% Adjustable parameters:
kernelwidth = 0.60;

[gridx, gridy] = meshgrid( ...
    mindata(1) + leveldiff(1)*(1:bincount), ...
    mindata(2) + leveldiff(2)*(1:bincount) ...
    ); 

norm2B = sum(B.^2, 2); 
norm2grid = gridx(:).^2 + gridy(:).^2; 
dist2mat = bsxfun(@plus, norm2B, norm2grid') - 2 * B * [gridx(:), gridy(:)]'; 
kernelmatrix = exp(-dist2mat ./ (2 * kernelwidth^2)) ./ (kernelwidth^2 * (2*pi)); 

mykde2 = mean(kernelmatrix, 1); 

figure; 
% bar3(reshape(mykde2, [bincount, bincount]));
surf(reshape(mykde2, [bincount, bincount]));
set(gca,'XTick',1:(bincount/10):bincount);
set(gca,'XTickLabel',minlevel(1):leveldiff(1)*(bincount/10):maxlevel(1));
set(gca,'YTick',1:(bincount/10):bincount);
set(gca,'YTickLabel',minlevel(2):leveldiff(2)*(bincount/10):maxlevel(2));
xlabel('statistics score'); 
ylabel('linear algebra score'); 
title('kernel density estimator'); 
saveas(gcf, 'kde', 'png'); 

input('press any key to continue ...\n'); 

saveas(gcf, 'kde', 'png'); 

% overlay contour plot with density. 
figure; 
scatter(B(:,1), B(:,2), '+'); title('statistics vs linear algebra'); 
hold on; 
[c, h] = contour(gridx, gridy, reshape(mykde2, [bincount, bincount]));
xlabel('statistics score'); 
ylabel('linear algebra score'); 
saveas(gcf, 'overlay', 'png'); 

input('press any key to continue ...\n'); 

% fit with Gaussian cdf
mu = mean(B, 1); 
sigma = cov(B); 
tmppdf = mvnpdf([gridx(:),gridy(:)], mu, sigma);
figure; 
scatter(B(:,1), B(:,2), '+'); title('statistics vs linear algebra (Gaussian)'); 
hold on; 
[c, h] = contour(gridx, gridy, reshape(tmppdf, [bincount, bincount]));
xlabel('statistics score'); 
ylabel('linear algebra score'); 
saveas(gcf, 'gaussian_overlay', 'png'); 

figure; 
% bar3(reshape(mykde2, [bincount, bincount]));
surf(reshape(tmppdf, [bincount, bincount]));
set(gca,'XTick',1:(bincount/10):bincount);
set(gca,'XTickLabel',minlevel(1):leveldiff(1)*(bincount/10):maxlevel(1));
set(gca,'YTick',1:(bincount/10):bincount);
set(gca,'YTickLabel',minlevel(2):leveldiff(2)*(bincount/10):maxlevel(2));
xlabel('statistics score'); 
ylabel('linear algebra score'); 
title('Gaussian density'); 
saveas(gcf, 'Gaussian', 'png'); 
