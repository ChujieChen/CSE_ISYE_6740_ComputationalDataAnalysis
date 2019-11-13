clear all
close all

% read data
load('usps_all');


%numbers
X = [data(:,:,2)' ; data(:,:,3)'];
X1=[data(:,:,2)'];
X2=[data(:,:,3)'];
Y = [ones(1100,1); - ones(1100,1)];

H = 16; 
W = 16; 

% Create a training set
Xtrain = [X(1:1100*0.8,:) ; X(1101:1980,:)];
Ytrain = [Y(1:1100*0.8,:); Y(1101:1980,:)];

% test set
Xtest = [X(1100*0.8+1:1100,:); X(1981:2200,:)];
Ytest = [Y(1100*0.8+1:1100,:); Y(1981:2200,:)];
figure;
show_image(Xtest, H, W); 
title('Test set','FontSize', 10);
train_size = size(Ytrain, 1);
test_size = size(Ytest, 1);

drawnow; 

Xtrain = double(Xtrain);
Xtest = double(Xtest);

fprintf(1, '--running svm\n'); 
[beta, beta0] = svm(Xtrain, Ytrain, 1e0);clc

Y_hat_train = sign(Xtrain * beta' + beta0);

precision1 =  sum(sum(Ytrain == 1 & Y_hat_train == 1)) / sum(sum(Ytrain == 1));
precision2 = sum(sum(Ytrain == -1 & Y_hat_train == -1)) / sum(sum(Ytrain == -1));
precision = (sum(sum(Ytrain == 1 & Y_hat_train == 1))+sum(sum(Ytrain == -1 & Y_hat_train == -1)))/length(Ytrain);
fprintf('train precision on class1 %g\n', precision1);
fprintf('train precision on class2 %g\n', precision2);
fprintf('train precision on all %g\n\n', precision);
Y_hat_test = sign(Xtest * beta' + beta0);
precision1 =  sum(sum(Ytest == 1 & Y_hat_test == 1)) / sum(sum(Ytest == 1));
precision2 = sum(sum(Ytest == -1 & Y_hat_test == -1)) / sum(sum(Ytest == -1));
precision = (sum(sum(Ytest == 1 & Y_hat_test == 1))+sum(sum(Ytest == -1 & Y_hat_test == -1)))/length(Ytest);
fprintf('test precision on class1 %g\n', precision1);
fprintf('test precision on class2 %g\n', precision2);
fprintf('test precision on all %g\n', precision);

[m,n]=size(Xtest);
incorrect_image_estimation=zeros(m,n);
ind_disp = find(Y_hat_test == 1);
incorrect_image_estimation(ind_disp,:)=Xtest(ind_disp,:);

figure()
show_image(incorrect_image_estimation, H, W);
title('Being classified as 2','FontSize', 10);
drawnow; 

incorrect_image_estimation=zeros(m,n);
ind_disp=find(Y_hat_test == -1);
incorrect_image_estimation(ind_disp,:)=Xtest(ind_disp,:);

figure()
show_image(incorrect_image_estimation, H, W);
title('Being classified as 3','FontSize', 10);
drawnow; 