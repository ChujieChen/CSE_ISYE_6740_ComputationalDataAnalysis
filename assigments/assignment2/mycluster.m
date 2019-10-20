function [ class ] = mycluster( bow, K )
%
% Your goal of this assignment is implementing your own text clustering algo.
%
% Input:
%     bow: data set. Bag of words representation of text document as
%     described in the assignment.
%
%     K: the number of desired topics/clusters. 
%
% Output:
%     class: the assignment of each topic. The
%     assignment should be 1, 2, 3, etc. 
%
% For submission, you need to code your own implementation without using
% any existing libraries

% YOUR IMPLEMENTATION SHOULD START HERE!
    %%%%%%%%%%%%%%%%%% by Chujie %%%%%%%%%%%%%%%%
    % number of documents - nd
    nd = size(bow, 1);
    % number of words - nw
    nw = size(bow, 2);
    % initialize mu_jc; make sure that sum_c mu_jc = 1
    mu = rand(nw,K);
    mu = mu ./ sum(mu,2);
    % initialize pi_c; make sure that sum_c pi_c = 1
    pi = rand(1, K);
    pi = pi / sum(pi);
    % initialize old_mu and old_pi
    old_mu = zeros(nw, K) - 1;
    old_pi = zeros(1, K) - 1;
    %% iterate E&M steps
%     tic
    while(norm(mu - old_mu) > 10^(-9) || norm(pi - old_pi) > 10^(-9))
%         disp("enter loop");
        old_mu = mu;
        old_pi = pi;
        %% E-step: update gamma_ic
        gamma = zeros(nd, K);
        for i = 1: nd
            for c = 1: K
                gamma(i, c) = getGamma(i, c, pi, mu, bow, nw, K);
            end
        end
        %% M-step: update mu, pi from gamma
        %%% update mu_jc
        for j = 1: nw
            for c = 1: K
                mu(j, c) = getMu(j, c, gamma, bow, nd, nw);
            end
        end
        %%% update pi
        pi = sum(gamma, 1) / nd;
    end
%     disp("Runtime in loop:");
%     disp(toc);
    %% assign document with clusters based on gamma
    [M, class] = max(gamma, [], 2);
end

function gamma_ic = getGamma(i, c, pi, mu, T, nw, K)
    numerator = pi(c) * bigPi(i, c, mu, T, nw);
    denominator = 0;
    for k = 1: K
        inner_pi = 1;
        for j = 1: nw
            inner_pi = inner_pi * mu(j, k)^(T(i,j));
        end
        denominator = denominator + pi(k) * inner_pi;
    end
    
    gamma_ic = numerator / denominator;
end

function product = bigPi(i, c, mu, T, nw)
    product = 1;
    for j = 1: nw
        product = product * (mu(j, c)^(T(i, j)));
    end
end

function mu_jc = getMu(j, c, gamma, T, nd, nw)
    numerator = sum(gamma(:, c) .* T(:, j));
    inner_sum = zeros(nd, 1);
    for i = 1: nd
        for l = 1: nw
            inner_sum(i) = inner_sum(i) + gamma(i,c) * T(i,l);
        end
    end
    denominator = sum(inner_sum);
    mu_jc = numerator / denominator;
end


