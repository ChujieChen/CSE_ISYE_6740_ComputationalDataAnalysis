% **********
% This demo is related to the boys and girls image dataset, including 2 parts.
% 1. Implement the feature selection algorithm.
% 2. Based on the previous feature sorting, repeadly add 200 feature at a
% time to do the classification and estimate the error.
% **********

clear; clc; close all;

%% load boys and girls dataset
% new_boys.mat and new_girls.mat are larger dataset which covers all students'
% faces, including 461 boys and 176 girls. The original images are stored in
% the folder boys and girls. The script generating these two data sets is
% 'generate_new_dataset.m'.

% boys.mat and girls.mat are the old and smaller dataset which only
% includes students in 2014 Fall class(the images are those only having 
% numbers as names in the two folders). There are 116 boys and 63 girls.

load('new_boys');
boys_data = double(boyRaw) / 255;
load('new_girls');
girls_data = double(girlRaw) / 255;
d = size(girlRaw, 1);

% load('boys');
% boys_data = double(data) / 255;
% load('girls');
% girls_data = double(data) / 255;
% d = size(data, 1);

%% Feature Selection
% Discretize the density from KDE
n_samples = 100;
sigma = 1;
normalization_const = 1 / sqrt(2 * pi * sigma^2);
mi = zeros(d, 1);

n_boys = size(boys_data, 2);
n_girls = size(girls_data, 2);
boys_marginal = n_boys / (n_boys + n_girls);
girls_marginal = n_girls / (n_boys + n_girls);

% For each feature
for i = 1:d
    sample_points = linspace(0, 1, n_samples);
    
    boys_pix_i_density = pdist2(sample_points', boys_data(i, :)');
    boys_pix_i_density = normalization_const * exp(- boys_pix_i_density.^2 / (2 * sigma^2));
    boys_pix_i_density = mean(boys_pix_i_density, 2)';
    
    girls_pix_i_density = pdist2(sample_points', girls_data(i, :)');
    girls_pix_i_density = normalization_const * exp(- girls_pix_i_density.^2 / (2 * sigma^2));
    girls_pix_i_density = mean(girls_pix_i_density, 2)';
    
    % Calculate joint distribution between feature i and label/class.
    joint_distr = [boys_marginal * boys_pix_i_density; girls_marginal * girls_pix_i_density];
    all_sum = sum(sum(joint_distr));
    
    % Normalize the joint distribution
    joint_distr = joint_distr / all_sum;
    
    % Calculate marginal distribution of feature i and label/class.
    % Marginal feature distribution P(X)
    feat_distr = sum(joint_distr, 1);
    % Marginal class distribution P(Y)
    class_distr = sum(joint_distr, 2);
    % Cross Product P(X) * P(Y)
    cross_prod = class_distr * feat_distr;
    
    % Mutual information \sum_x,y P(X, Y) log( P(X, Y) / (P(X)P(Y)) )
    tmp = joint_distr .* log(joint_distr ./ cross_prod);
    % We define 0 * log 0 to be 0
    tmp(isnan(tmp)) = 0;
    mi(i) = sum(tmp(:));
end

% Visualize feature mutual information 
imagesc(reshape(mi, 65, 65))
colorbar

[~, sorted_feat_idx] = sort(mi, 1, 'descend');


%% Classification 
% Use the logistic regression model to do the classification
% Use 10-fold cross validation to evaluate the model
n_total = n_boys + n_girls;
full_data = [boys_data'; girls_data'];
trueY = [ones(n_boys, 1); 2 * ones(n_girls, 1)];
full_label = sparse(1:n_total, trueY, 1, n_total, 2);

% Choose features
% Each feature block contains 200 features and there are 20 blocks in total
feat_block = 200;
total_blocks = 20;
feat_select_err_list = zeros(1, total_blocks);
data_rand_idx = randperm(n_total);
n_folds = 10;
cv_size = ceil(n_total / n_folds);

% Choose different numbers of features to build models
for f_i = 1:total_blocks
    % Choose top f_i * feat_block features
    fprintf('feature selection block: %i\n', f_i)
    top_feat_idx = sorted_feat_idx(1:(feat_block*f_i));
    
    % Leave-one-out error
    cv_err = zeros(1, n_folds);
    for i = 1:n_folds
        % Seperate the training and testing data
        test_idx = data_rand_idx((1+(i-1)*cv_size):min(n_total, i*cv_size));
        data_idx = setdiff(1:n_total, test_idx);
        cv_trainX = full_data(data_idx, top_feat_idx);
        cv_trainY = full_label(data_idx, :);
        % Building logistic regression model
        B = logistic_regression(cv_trainX, cv_trainY);
        % Test the model on testing data and compute the error
        [~, pred] = max(full_data(test_idx, top_feat_idx) * B, [], 2);
        cv_err(i) = mean(pred ~= trueY(test_idx));
    end
    feat_select_err_list(f_i) = mean(cv_err);
end
figure
plot(feat_select_err_list)
xlabel('Feature blocks')
ylabel('Cross validation error')
