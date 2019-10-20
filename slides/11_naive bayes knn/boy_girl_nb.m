%% 
% load dataset for boys and girls

boys = load('boys.mat');
girls = load('girls.mat');

% dividing into 5 equally split datasets for cross validation
% choosing five fold split as 10 results in highly uneven splits
boys_data = double(boys.data(:,:));
girls_data = double(girls.data(:,:));

train_size_boys = size(boys_data, 2);
train_size_girls = size(girls_data, 2);

test_size_boys = train_size_boys;
test_size_girls = train_size_girls;

randn('seed', 1); 
rand('seed', 1); 

% iterating through random splits of data
for p = 1:1

    boys_indices = randperm(size(boys_data, 2));
    girls_indices = randperm(size(girls_data, 2));
    
    boys_train = boys_indices(1:train_size_boys);
    girls_train = girls_indices(1:train_size_girls);
    
    boys_test = boys_train;
    girls_test = girls_train;
    
    trainx = ([boys_data(:,boys_train), girls_data(:,girls_train)]);
    testx = ([boys_data(:,boys_test), girls_data(:, girls_test)]); 
    trainy = [zeros(1, train_size_boys), ones(1, train_size_girls)]; 
    testy = [zeros(1, test_size_boys), ones(1, test_size_girls)];

    %show_image(trainx', 65, 65);
    trainno = length(trainy); 
    dimno = size(trainx, 1);
    testno = length(testy); 
    
    figure; 
    show_image(testx', 65, 65); 
    
    input('press a key to continue ....\n'); 

    %%
    % Naive Bayes

    % estimate class prior distribution; 
    py = zeros(2, 1);
    for i = 1:2
        py(i) = sum(trainy==i-1) ./ trainno; 
    end

    % estimate the class conditional distribution; 
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
        % for each and every dimension, we sum the predicted probability based
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
        arr(:,:) = img_class(1,:,:);
        figure;            
        show_image(arr', 65, 65);
        title('boys'); 
        
        arr(:,:) = img_class(2,:,:);
        figure;            
        show_image(arr', 65, 65);
        title('girls');         
    end
    
end

