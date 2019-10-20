% Multinomial logistic regression
function B = logistic_regression(X, Y)  
    % X is n x d 
    % Y is n x k
    % B is d x k
    [n, d] = size(X);
    k = size(Y, 2);
    B = zeros(d, k);
    max_iter = 200;
    step_size = 0.1;

    err_list = [];
    for i = 1:max_iter
        pred = softmax_fn(X * B);
        err = Y - pred;
        grad = X' * err / n;
        B = B + step_size * grad;
        err_list = [err_list sum(sum(err .* err))];
        %fprintf('--train error: %f\n', err_list(end));
    end
    % plot(err_list)
end

% Numerically stable softmax
function p = softmax_fn(y)
% [n, k] = size(y)
    max_y = max(y, [], 2);
    ny = exp(bsxfun(@minus, y, max_y));
    p = bsxfun(@rdivide, ny, sum(ny, 2));

    if any(isnan(p))
        error('Run into NaN!')
    end
end