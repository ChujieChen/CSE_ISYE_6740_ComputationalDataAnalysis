function [beta, beta0] = svm(X, Y, C)

% number of samples
N = size(X,1);
% get NxN kernel matrix (here is linear kernel)
K = X * X';

% used for quadratic term in obj func
G = (Y * Y') .* K;
% used for linear term in obj func
b = - ones(N, 1);

% lowerbound of alpha
lb = zeros(N,1);
% upperbound of alpha
ub = C .* ones(N,1);
opts = optimset('Algorithm', 'interior-point-convex');

%  quadratic optimizatino
%  solve the dual problem
alpha = quadprog(G, b, [], [], Y', 0, lb, ub, [], opts);

% findout the support vector indexes, i.e., the vectors whose alpha is
% larger than zero
sv = find(alpha > 1e-10);

% Calculate the decision bound
% beta (i.e., w) is a linear combination of support vectors
beta = sum(bsxfun(@times, alpha(sv) .* Y(sv), X(sv,:)));
% beta0 (i.e., bias) is calculated using KKT condition
beta0 = mean(Y(sv) - sum(bsxfun(@times, alpha(sv)' .* Y(sv)', K(sv, sv)),2));

end

