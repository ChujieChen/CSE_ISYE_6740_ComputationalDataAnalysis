clear all
clc
close all

irow=65;
icol=65; 
%%
% read data

load('boys');
boys_data = double(data_b) / 255;
nboys=size(boys_data,2);


load('girls');
girls_data = double(data_g) / 255;
ngirls=size(girls_data,2); 

% read data

ind1=crossvalind('Kfold',data_b(1,:),5);
ind2=crossvalind('Kfold',data_g(1,:),5);
H = 65; 
W = 65; 

%numbers
X = [boys_data' ; girls_data'];
Y = [ones(116,1); - ones(63,1)];

%% Training and testing on all samples

[beta_all, beta0_all] = svm(X, Y, 1);
Y_hat=sign(X * beta_all' + beta0_all);
precision1_t =  sum(sum(Y == 1 & Y_hat == 1)) / sum(sum(Y == 1));
precision2_t = sum(sum(Y == -1 & Y_hat == -1)) / sum(sum(Y == -1));
precision_t = (sum(sum(Y == 1 & Y_hat == 1))+sum(sum(Y == -1 & Y_hat == -1)))/length(Y);
fprintf('training and testing on all images, precision: %g\n\n', precision_t);

ind3=find(Y_hat==1);
ind4=find(Y_hat==-1);

X_b=zeros(size(X));
X_g=zeros(size(X));

X_b(ind3,:)=X(ind3,:);
X_g(ind4,:)=X(ind4,:);
figure;
show_image(X_b, H, W); 
title('Images Classified as Boys','FontSize', 10);


figure;
show_image(X_g, H, W); 
title('Images Classifeid as Girls','FontSize', 10);clc

fprintf('training and testing on all images, precision: %g\n\n', precision_t);
%%

input('--press any key to continue');

% Cross Validation
for i=1:5
test_b=find(ind1 == i);
train_b=find(ind1 ~= i);    

test_g=find(ind2 == i);
train_g=find(ind2 ~= i); 

% Create a training set
Xtrain = [X(train_b,:) ; X(116+train_g,:)];
Ytrain = [Y(train_b,:); Y(116+train_g,:)];

% For Visualisation
Xtrain2=[X(1:93,:);X([117:156,171:179],:)];
Ytrain2=[Y(1:93,:);Y([117:156,171:179],:)];
Xtest2=[X(94:116,:);X(157:170,:)];
Ytest2=[Y(94:116,:);Y(157:170,:)];

% test set
Xtest = [X(test_b,:); X(116+test_g,:)];
Ytest = [Y(test_b,:); Y(116+test_g,:)];

train_size = size(Ytrain, 1);
test_size = size(Ytest, 1);

Xtrain = double(Xtrain);
Xtest = double(Xtest);

[beta2, beta20] = svm(Xtrain2, Ytrain2, 1);
[beta, beta0] = svm(Xtrain, Ytrain, 1);clc

Y_hat_train = sign(Xtrain * beta' + beta0);
Y_hat_train2 = sign(Xtrain2 * beta2' + beta20);

precision1(i) =  sum(sum(Ytrain == 1 & Y_hat_train == 1)) / sum(sum(Ytrain == 1));
precision2(i) = sum(sum(Ytrain == -1 & Y_hat_train == -1)) / sum(sum(Ytrain == -1));
precision(i) = (sum(sum(Ytrain == 1 & Y_hat_train == 1))+sum(sum(Ytrain == -1 & Y_hat_train == -1)))/length(Ytrain);


Y_hat_test = sign(Xtest * beta' + beta0);
Y_hat_test2 = sign(Xtest2 * beta2' + beta20);
precision1_test(i) =  sum(sum(Ytest == 1 & Y_hat_test == 1)) / sum(sum(Ytest == 1));
precision2_test(i) = sum(sum(Ytest == -1 & Y_hat_test == -1)) / sum(sum(Ytest == -1));
precision_test(i) = (sum(sum(Ytest == 1 & Y_hat_test == 1))+sum(sum(Ytest == -1 & Y_hat_test == -1)))/length(Ytest);

end



%% Visualisation
figure;
show_image(Xtrain2, H, W); 
title('Training set','FontSize', 10);

figure;
show_image(Xtest2, H, W); 
title('Test set','FontSize', 10);
% 
% [p,q]=size(Xtrain2);
% incorrect_image_estimation=zeros(p,q);
% ind_disp=find(Ytrain2 ~= Y_hat_train2);
% incorrect_image_estimation(ind_disp,:)=Xtrain2(ind_disp,:);
% figure()
% show_image(incorrect_image_estimation, H, W);
% title('Incorrectly classified images in the Training set set','FontSize', 10);
% 


[m,n]=size(Xtest2);
incorrect_image_estimation_test=zeros(m,n);
ind_disp=find(Ytest2 ~= Y_hat_test2);
incorrect_image_estimation_test(ind_disp,:)=Xtest2(ind_disp,:);
figure()
show_image(incorrect_image_estimation_test, H, W);
title('Incorrectly classified images in the Test set','FontSize', 10);

clc

pre1_m=mean(precision1);
pre2_m=mean(precision2);
pre_m=mean(precision);
fprintf('train precision on boys %g\n', pre1_m);
fprintf('train precision on girls %g\n', pre2_m);
fprintf('train precision on all %g\n\n', pre_m);
pre1_m_t=mean(precision1_test);
pre2_m_t=mean(precision2_test);
pre_m_t=mean(precision_test);
fprintf('test precision on boys %g\n', pre1_m_t);
fprintf('test precision on girls %g\n', pre2_m_t);
fprintf('test precision on all %g\n', pre_m_t);
