% **********
% This demo is related to the leaf dataset, including 2 parts.
% 1. Implement the feature selection algorithm.
% 2. Based on the previous feature sorting, repeadly add 1 feature at a
% time to do the classification and estimate the error.
% **********

clear; clc; close all
% leaf dataset
data = importdata('../PCA_leaf_example/leaf.mat');

% Dataset description
% The provided data comprises the following shape (attributes 3 to 9) and 
% texture (attributes 10 to 16) features:
% 1. Class (Species)
% 2. Specimen Number
% 3. Eccentricity
% 4. Aspect Ratio
% 5. Elongation
% 6. Solidity
% 7. Stochastic Convexity
% 8. Isoperimetric Factor
% 9. Maximal Indentation Depth
% 10. Lobedness
% 11. Average Intensity
% 12. Average Contrast
% 13. Smoothness
% 14. Third moment
% 15. Uniformity
% 16. Entropy

feature_names = {'Eccentricity', 'Aspect Ratio', 'Elongation', 'Solidity', 'Stochastic Convexity',...
    'Isoperimetric Factor', 'Maximal Indentation Depth', 'Lobedness', 'Average Intensity',...
    'Average Contrast', 'Smoothness', 'Third moment', 'Uniformity', 'Entropy'};

% Extract attributes from the raw data.
X = data(:,3:16);
[n, d] = size(X);
Y = data(:, 1);

% Map the class indexes to 1 to n_classes 
% (Class indexes are not continuous in this dataset)
[uniq_symbols, ~, Y] = unique(Y); 
n_classes = length(uniq_symbols);


%% Feature Selection 
n_samples = 100;
sigma = 1;
normalization_const = 1 / sqrt(2 * pi * sigma^2);
mi = zeros(d, 1);

% For each value of the label y = k, estimate density P(y = k)
class_prior = histc(Y, 1:n_classes)';
class_prior = class_prior / sum(class_prior);

% For each feature x_i
for i = 1:d
    joint_distr = zeros(n_samples, n_classes);
    feat_val_min = min(X(:, i));
    feat_val_max = max(X(:, i));
    
    % Estimate the feature density
    % Discretize KDE
    sample_points = linspace(feat_val_min, feat_val_max, n_samples)';
    sample_train_data_dist = pdist2(sample_points, X(:, i));
    densities = normalization_const * exp(- sample_train_data_dist.^2 / (2 * sigma^2));
    
    % For each value of the label y = j, estimate p(x_i/y = j) 
    for j = 1:n_classes
        class_density = mean(densities(:, Y == j), 2);
        joint_distr(:, j) = class_density * class_prior(j);
    end
    
    % Normalize joint distribution
    joint_distr = joint_distr / sum(sum(joint_distr));
    
    % Marginal feature distribution P(X)
    feat_distr = sum(joint_distr, 2);
    % Marginal class distribution P(Y)
    class_distr = sum(joint_distr, 1);
    % Cross product P(X) * P(Y)
    cross_prod = feat_distr * class_distr;
    
    % Mutual information \sum_x,y P(X, Y) log( P(X, Y) / (P(X)P(Y)) )
    tmp = joint_distr .* log(joint_distr ./ cross_prod);
    % We define 0 * log 0 to be 0
    tmp(isnan(tmp)) = 0;
    % Score feature x_i
    mi(i) = sum(tmp(:));
end

% Sort features based on the scores and show the top 5 features
[~, sorted_mi_idx] = sort(mi,1, 'descend');
fprintf('Top 5 informative features\n')
for i = 1:5
    fprintf('%d. %s\n',i, feature_names{sorted_mi_idx(i)})
    % feature_names{sorted_mi_idx(1:5)}
end

stem(mi, 'filled')
xlabel('Features')
ylabel('Mutual information')


%% Classification
% Use the logistic regression model to do the classification
% Use 10-fold cross validation to evaluate the model
n_total = size(X, 1);
full_data = X;
trueY = Y;
full_label = sparse(1:n_total, trueY, 1, n_total, n_classes);

feat_block = 1;
total_blocks = 14;
feat_select_err_list = zeros(1, total_blocks);
data_rand_idx = randperm(n_total);
n_folds = 10;
cv_size = ceil(n_total / n_folds);

% Choose different numbers of features to build models
for f_i = 1:total_blocks
    % Choose top f_i features
    fprintf('feature selection block: %i\n', f_i)
    top_feat_idx = sorted_mi_idx(1:(feat_block*f_i));
    
    % Leave-one-out error
    cv_err = zeros(1, n_folds);
    for i = 1:n_folds
        % Seperate training and testing data
        test_idx = data_rand_idx((1+(i-1)*cv_size):min(n_total, i*cv_size));
        data_idx = setdiff(1:n_total, test_idx);
        cv_trainX = full_data(data_idx, top_feat_idx);
        cv_trainY = full_label(data_idx, :);
        % Build logistic regression model
        B = logistic_regression(cv_trainX, cv_trainY);
        % Test the model on testing data and compute the error
        [~, pred] = max(full_data(test_idx, top_feat_idx) * B, [], 2);
        cv_err(i) = mean(pred ~= trueY(test_idx));
    end
    feat_select_err_list(f_i) = mean(cv_err);
end
figure
plot(feat_select_err_list)
xlabel('Number of Features')
ylabel('Cross validation error')