
% Histogram and KDE example with mixture of gaussians
% In this example we have a randomly generated data derived from 3 normal
% distribution centred around 3 different points. We show the density
% estimation using both histogram methond and KDE
%===============================================================
% Data preparation
%  (Please copy and paste one section of the below.)
%===============================================================
clc,close all
% When each gaussian component does not overlap:
A = zeros(300, 1);
A(1:100) = randn(100,1) * 0.1 + 5;
A(101:200) = randn(100,1) * 0.1 + 3;
A(201:300) = randn(100,1) * 0.1 + 7;

%When each gaussian component overlaps a lot:
A(1:100) = randn(100,1) * 1 + 5;
A(101:200) = randn(100,1) * 1 + 3;
A(201:300) = randn(100,1) * 1 + 7;

% 3D example
B(1:100,:) = randn(100,2) + repmat([4 10],100,1);
B(101:200,:) = randn(100,2) + repmat([10 4],100,1);
B(201:300,:) = randn(100,2) + repmat([15 15],100,1);


%===============================================================
% Histogram method
%===============================================================

% Adjustable parameters:
bincount = 50;

myhist = zeros(bincount,1);
mindata = min(A);
maxdata = max(A);
minlevel = floor(mindata);
maxlevel = ceil(maxdata);
leveldiff = (maxlevel - minlevel)/bincount;

for i=1:size(A,1)
	bin = (A(i) - minlevel) / leveldiff;
	bin = ceil(bin);
	myhist(bin) = myhist(bin) + 1;
end
figure;
bar(myhist);
set(gca,'XTick',1:(bincount/10):bincount);
set(gca,'XTickLabel',minlevel:leveldiff*(bincount/10):maxlevel);


%===============================================================
% Kernel density estimation
%===============================================================
% Here we use the tophat kernel.
% Adjustable parameters:
kernelwidth = 0.2; % kernel width determines how smooth estimate will be

bincount = 1000;
mykde = zeros(bincount,1);
mindata = min(A);
maxdata = max(A);
minlevel = floor(mindata);
maxlevel = ceil(maxdata);
leveldiff = (maxlevel - minlevel)/bincount;

for i=1:size(A,1)
	bin = (A(i) - minlevel) / leveldiff;
	bin = ceil(bin);

	range = floor(kernelwidth/leveldiff);

	added = zeros(bincount,1);
	minindex = max(1, bin-range);
	maxindex = min(bincount, bin+range);
	added(minindex:maxindex) = 1;

	mykde = mykde + added; % tophat kernel
end
figure;
plot(mykde);
set(gca,'XTick',1:(bincount/10):bincount);
set(gca,'XTickLabel',minlevel:leveldiff*(bincount/10):maxlevel);


%===============================================================
% Histogram for 3D
%===============================================================

% Adjustable parameters:
bincount = 50;

myhist3 = zeros(bincount,bincount,1);
mindata = min(B);
maxdata = max(B);
minlevel = floor(mindata);
maxlevel = ceil(maxdata);
leveldiff = (maxlevel - minlevel)/bincount;

for i=1:size(B,1)
	bin = (B(i,:) - minlevel) ./ leveldiff;
	bin = ceil(bin);
	myhist3(bin(1),bin(2)) = myhist3(bin(1),bin(2)) + 1;
end
figure;
bar3(myhist3);
set(gca,'XTick',1:(bincount/10):bincount);
set(gca,'XTickLabel',minlevel(1):leveldiff(1)*(bincount/10):maxlevel(1));
set(gca,'YTick',1:(bincount/10):bincount);
set(gca,'YTickLabel',minlevel(2):leveldiff(2)*(bincount/10):maxlevel(2));
title('Histogram for 2D data');


%===============================================================
% Kernel density estimation
%===============================================================

% Adjustable parameters:
kernelwidth = 0.5;

bincount = 100;

mindata = min(B, [], 1);
maxdata = max(B, [], 1);
minlevel = floor(mindata);
maxlevel = ceil(maxdata);
leveldiff = (maxlevel - minlevel)/bincount;

[gridx, gridy] = meshgrid( ...
    mindata(1) + leveldiff(1)*(1:bincount), ...
    mindata(2) + leveldiff(2)*(1:bincount) ...
    ); 

norm2B = sum(B.^2, 2); 
norm2grid = gridx(:).^2 + gridy(:).^2; 
dist2mat = bsxfun(@plus, norm2B, norm2grid') - 2 * B * [gridx(:), gridy(:)]';
% Using Gaussian Kernel
kernelmatrix = exp(-dist2mat ./ (2 * kernelwidth^2)) ./ (kernelwidth^2 * (2*pi)); 

mykde2 = mean(kernelmatrix, 1); 

figure; 
% bar3(reshape(mykde2, [bincount, bincount]));
surf(reshape(mykde2, [bincount, bincount]));
set(gca,'XTick',1:(bincount/10):bincount);
set(gca,'XTickLabel',minlevel(1):leveldiff(1)*(bincount/10):maxlevel(1));
set(gca,'YTick',1:(bincount/10):bincount);
set(gca,'YTickLabel',minlevel(2):leveldiff(2)*(bincount/10):maxlevel(2));
title('kernel density estimator for 2D data'); 

