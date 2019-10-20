% This example uses Naive bayes and K-Nearest neighbour to classify image
% as a boy or a girl
clear ; close all;
%% 
% load dataset for boys and girls

% boys = load('boys.mat');
% girls = load('girls.mat');
% 
% % dividing into 5 equally split datasets for cross validation
% % choosing five fold split as 10 results in highly uneven splits
% boys_data = double(boys.data(:,:));
% girls_data = double(girls.data(:,:));

% This load the dataset containing images of boy and girls
boys = load('new_boys.mat');
girls = load('new_girls.mat');


rand('seed', 1); 
randn('seed', 1); 

boys_data = double(boys.boyRaw(:,:));
girls_data = double(girls.girlRaw(:,:));

% We are splitting the dataset in 80:20 ratio for training and testing
train_size_boys = round(0.8 * size(boys_data, 2));
train_size_girls = round(0.8 * size(girls_data, 2));

test_size_boys = size(boys_data, 2) - train_size_boys;
test_size_girls = size(girls_data, 2) - train_size_girls;


% p controls the number of times the two algorithm are run on random 80:20
% splits of the dataset
for p=1:1  
    boys_indices = randperm(size(boys_data, 2));
    girls_indices = randperm(size(girls_data, 2));
    
    boys_train = boys_indices(1:train_size_boys);
    girls_train = girls_indices(1:train_size_girls);
    
    boys_test = boys_indices(train_size_boys+1:size(boys_data, 2));
    girls_test = girls_indices(train_size_girls+1:size(girls_data, 2));
    
    trainx = ([boys_data(:,boys_train), girls_data(:,girls_train)]);
    testx = ([boys_data(:,boys_test), girls_data(:, girls_test)]); 
    trainy = [zeros(1, train_size_boys), ones(1, train_size_girls)]; 
    testy = [zeros(1, test_size_boys), ones(1, test_size_girls)];

    %show_image(trainx', 65, 65);
    trainno = length(trainy); 
    dimno = size(trainx, 1);
    testno = length(testy); 


    %%
    % Naive Bayes

    % estimate class prior distribution i.e p(boys) and p(girls); 
    % it is simply the number of examples of boys by total number of
    % examples
    py = zeros(2, 1);
    for i = 1:2
        py(i) = sum(trainy==i-1) ./ trainno; 
    end

    % estimate the class conditional distribution by finding mean and std 
    %dev for each dimesion for each class seperately; 
    mu_x_y = zeros(dimno, 2); 
    sigma_x_y = zeros(dimno, 2); 

    for i = 1:dimno
        % taking data points from appropriate class for each dimension and finding mean and variance
        mu_x_y(i,1) = mean(trainx(i,1:train_size_boys));
        mu_x_y(i,2) = mean(trainx(i,train_size_boys+1:train_size_boys+train_size_girls));
        sigma_x_y(i, 1) = std(trainx(i,1:train_size_boys), 0);
        sigma_x_y(i, 2) = std(trainx(i,train_size_boys+1:train_size_boys+train_size_girls), 0);
    end

    pytest = zeros(testno, 2);
    predy = zeros(testno, 1);
    img_class = zeros(2,dimno,testno);
    count_acc = 0;
    for i = 1:testno

        % for each class
        % for each and every dimension, we sum the predicted log probability based
        % on mean and variance calculated from training data
        % and prior from training data
        for k =1:2
            pytest(i, k) = log10(py(k)); 
            for j = 1:dimno
                pytest(i, k) = pytest(i, k) + log10(normpdf(testx(j,i), mu_x_y(j,k), sigma_x_y(j,k)+1e-3)); 
            end
        end

        % select maximum proability among classes
        %size(pytest(i,:))
        [maxm, index] = max(pytest(i,:));

        predy(i) = index-1;

        if predy(i) == testy(i)
            count_acc = count_acc + 1;
        end

        img_class(index,:,i) = testx(:,i);

    end

    nb_accuracy(p) = count_acc/testno

    if p==1
        % plot all images predicted by class
        for i=1:2
            size(img_class(i,:,:));
            arr(:,:) = img_class(i,:,:);
            figure;
            show_image(arr', 65, 65);
        end
    end

    %% K-nearest neighbor

    % For each test point find the nearest K training points and classify
    % accordingly
    K = 20 ; % Chosing K as 20
    predy_knn = zeros(testno, 1);
    img_class_knn = zeros(2,dimno,testno);
    count_acc_knn = 0;
    for i = 1:testno

        % returns a n cross K matrix of indices, if looking for K nearest neighbors
        [n,d] = knnsearch(trainx', testx(:,i)', 'k', K);

        count = zeros(20);

        % incrementing count, since trainy can contain zero, incrementing by 1
        for j=1:size(n,2)
            count(trainy(n(j)) + 1) = count(trainy(n(j)) + 1) + 1;
        end

        [maxm, index] = max(count(:));

        predy_knn(i) = index-1;

        if predy_knn(i) == testy(i)
            count_acc_knn = count_acc_knn + 1;
        end

        img_class_knn(index,:,i) = testx(:,i);

    end

    knn_accuracy(p) = count_acc_knn/testno
    
    if p==1
        for i=1:2
            arr(:,:) = img_class_knn(i,:,:);
            figure;
            show_image(arr',65,65);
        end
    end
    
end

% nb_accuracy_final = mean(nb_accuracy)
% knn_accuracy_final = mean(knn_accuracy)
