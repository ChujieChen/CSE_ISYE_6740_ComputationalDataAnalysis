% **********
% This demo shows how to build logistic regression model to classify boys and girls image.
% 1. Implement the logistic regression model with batch gradient descent method
% 2. Show the error for both training and testing data set
% **********
clear
clc
close all


irow=65;
icol=65; 

%% load boys and girls dataset
% (1) new_boys.mat and new_girls.mat and the original images are available in
% the feature selection lecture code.
% 
% (2) new_boys.mat and new_girls.mat are larger dataset which covers all students'
% faces, including 461 boys and 176 girls. The original images are stored in
% the folder boys and girls. The script generating these two data sets is
% 'generate_new_dataset.m'.

% (3) boys.mat and girls.mat are the old and smaller dataset which only
% includes students in 2014 Fall class(the images are those only having 
% numbers as names in the two folders). There are 116 boys and 63 girls.

load('new_boys');
boys_data = double(boyRaw) / 255;
nboys = size(boys_data,2); % 461

figure;
show_image(boyRaw', irow, icol);
title('All 461 Boys','FontSize', 16)
 

load('new_girls');
girls_data = double(girlRaw) / 255;
ngirls = size(girls_data,2);  % 176

figure;
show_image(girlRaw', irow, icol);
title('All 176 Girls','FontSize', 16)

rand('seed', 1); 
randn('seed', 1); 

%% Merge dataset and seperate training and testing set

X = [boys_data' ; girls_data'];
Y = [ones(nboys,1); 2 * ones(ngirls,1)]; % boy as label 1, girl as label 2

% Use p percent data as training data
p = 0.8;
nboys_train = round(nboys * p);
nboys_test = nboys - nboys_train;
ngirls_train = round(ngirls*p);
ngirls_test = ngirls-ngirls_train;

% Training set
Xtrain = [X(1:nboys_train,:) ; X((nboys + 1):(nboys + ngirls_train),:)];
Ytrain = [Y(1:nboys_train,:); Y((nboys + 1):(nboys + ngirls_train),:)];

% Testing set
Xtest = [X(nboys_train + 1:nboys,:); X(nboys + 1 + ngirls_train:(nboys + ngirls),:)];
Ytest = [Y(nboys_train + 1:nboys,:); Y(nboys + 1 + ngirls_train:(nboys + ngirls),:)];

train_size = size(Ytrain, 1);
test_size = size(Ytest, 1);

Xtrain = double(Xtrain);
Xtest = double(Xtest);

%% Building classifier using batch gradient descent method

% Parameters
mu = 0.0001; % learning rate
threshold = 1e-6;
max_iter = 4000;

% Random initialization
theta = randn(1, size(Xtrain, 2));

currnorm = 10; % some value larger than threshold
round = 1;

while ((currnorm > threshold) && (round <= max_iter)) 
	% Take the direction of the negative gradient
	temp = Xtrain * theta';
	sgn = Ytrain * -2 + 3;
	diff = ((-1 ./ (1 + exp (sgn .* temp))) .* sgn)' * Xtrain;

    % Update the parameters
	theta = theta - mu * diff;
	currnorm = norm (diff) / size(Xtrain, 1);
	fprintf('Round = %d:\t%f\n', round, currnorm);
    
	round = round + 1;
end

% Evaluate the classifier on the training data set
% (1) Calculate the classification error for training data
% (2) Show the images which are incorrect classified
% +1: Label 1(Boys)
% -1: Label 2(Girls)
cc_train = 0;
cr_train = 0;
rc_train = 0;
rr_train = 0;
cc_test = 0;
cr_test = 0;
rc_test = 0;
rr_test = 0;

[m,n] = size(Xtrain);
incorrect_image_train = zeros(m,n);

for i = 1: nboys_train
	% Training set of Class 1(boys)
	logloss1 = log (1 + exp (-1 * Xtrain(i,:) * theta'));
	logloss2 = log (1 + exp (Xtrain(i,:) * theta'));

	if (logloss1 == inf) && (logloss2 < inf)
		cr_train = cr_train + 1;
        incorrect_image_train(i,:) = Xtrain(i,:);
	elseif (logloss1 < inf) && (logloss2 == inf)
		cc_train = cc_train + 1;
	elseif (logloss1 < logloss2)
		cc_train = cc_train + 1;
	elseif (logloss1 > logloss2)
		cr_train = cr_train + 1;
        incorrect_image_train(i,:) = Xtrain(i,:);
    end
end

for i = nboys_train + 1: nboys_train + ngirls_train
	% Training set of Class 2(girls)
	logloss1 = log (1 + exp (-1 * Xtrain(i,:) * theta'));
	logloss2 = log (1 + exp (Xtrain(i,:) * theta'));

	if (logloss1 == inf) && (logloss2 < inf)
		rr_train = rr_train + 1;
	elseif (logloss1 < inf) && (logloss2 == inf)
		rc_train = rc_train + 1;
        incorrect_image_train(i,:) = Xtrain(i,:);
	elseif (logloss1 < logloss2)
		rc_train = rc_train + 1;
        incorrect_image_train(i,:) = Xtrain(i,:);
	elseif (logloss1 > logloss2)
		rr_train = rr_train + 1;
    end
end
 
figure;
show_image(Xtrain * 255,irow, icol); 
title('Training Data','FontSize', 16)

figure;
show_image(incorrect_image_train * 255,irow, icol); 
title('Incorrect Classification for Training Data','FontSize', 16)


%% For testing data, we use the classifier we have built to do the classification and evaluate the error
% (1) Calculate the classification error for testing data
% (2) Show the images which are incorrect classified

[m,n] = size(Xtest);
incorrect_image_test = zeros(m,n);

for i = 1: nboys_test
	% Testing set of Class 1(boys)
	logloss1 = log (1 + exp (-1 * Xtest(i,:) * theta'));
	logloss2 = log (1 + exp (Xtest(i,:) * theta'));

	if (logloss1 == inf) && (logloss2 < inf)
		cr_test = cr_test + 1;
        incorrect_image_test(i,:) = Xtest(i,:);
	elseif (logloss1 < inf) && (logloss2 == inf)
		cc_test = cc_test + 1;
	elseif (logloss1 < logloss2)
		cc_test = cc_test + 1;
	elseif (logloss1 > logloss2)
		cr_test = cr_test + 1;
        incorrect_image_test(i,:) = Xtest(i,:);
	end
end

for i = nboys_test + 1: ngirls_test
	% Testing set of Class 2(girls)
	logloss1 = log (1 + exp (-1 * Xtest(i,:) * theta'));
	logloss2 = log (1 + exp (Xtest(i,:) * theta'));

	if (logloss1 == inf) && (logloss2 < inf)
		rr_test = rr_test + 1;
	elseif (logloss1 < inf) && (logloss2 == inf)
		rc_test = rc_test + 1;
        incorrect_image_test(i,:) = Xtest(i,:);
	elseif (logloss1 < logloss2)
		rc_test = rc_test + 1;
         incorrect_image_test(i,:) = Xtest(i,:);
	elseif (logloss1 > logloss2)
		rr_test = rr_test + 1;
	end
end

figure;
show_image(Xtest * 255,irow, icol); 
title('Testing Data','FontSize', 16)

figure; 
show_image(incorrect_image_test * 255,irow, icol); 
title('Incorrect Classification for Testing Data','FontSize', 16)

% Calculate the training error and testing error
train_err = (cr_train + rc_train) / (cc_train + cr_train + rc_train + rr_train);
test_err = (cr_test + rc_test) / (cc_test + cr_test + rc_test + rr_test);

fprintf('Training Error = %f\n', train_err);
fprintf('Testing Error = %f\n', test_err);