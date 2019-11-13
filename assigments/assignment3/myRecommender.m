function [ U, V ] = myRecommender( rateMatrix, lowRank )
    % Please type your name here:
    name = 'Chen, Chujie';
    disp(name); % Do not delete this line.

    % Parameters
    maxIter = 5e3; % Choose your own.
    learningRate = 1e-4; % Choose your own.
    regularizer = 1e-2; % Choose your own.
    
    % Random initialization:
    [n1, n2] = size(rateMatrix);
    U = rand(n1, lowRank) / lowRank;
    V = rand(n2, lowRank) / lowRank;

    % Gradient Descent:
    
    % IMPLEMENT YOUR CODE HERE.
    %% change of reconstruction error (see if it is converged) and iteration used to stop loop
    deltaRecoError = 1e-4 * nnz(rateMatrix > 0);
    recoError = Inf;
    newRecoError = 5 * 5 * numel(rateMatrix);
    it = 0;
    %% update U and V
    while abs(newRecoError - recoError) > deltaRecoError && it < maxIter
        recoError = newRecoError;
        % note that if sum(Uuk*Vik) holds non-zero value for
        % corresponding zero value Mui, we should not take it into account
        % since we want UV to be similar to M as much as possible
        % steal code from homework3.m
        U = U + 2 * learningRate * ((rateMatrix - U * V') .* (rateMatrix > 0)) * V - 2 * learningRate * regularizer * U;
        V = V + 2 * learningRate * ((rateMatrix - U * V') .* (rateMatrix > 0))' * U - 2 * learningRate * regularizer * V;
        % update the objective funtion
        newRecoError = sumsqr((rateMatrix - U * V') .* (rateMatrix > 0)) + regularizer * sumsqr(U) + regularizer * sumsqr(V);
%         newRecoError = sumsqr((rateMatrix - U * V') .* (rateMatrix > 0));
        it = it + 1;
    end
    %% some checks for myself
%     disp(it);
end