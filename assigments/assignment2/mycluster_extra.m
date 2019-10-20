function [ class ] = mycluster_extra( bow, K )
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
%% By Chujie on 10/16/2019
%% initialization
% use sparse
bow = sparse(bow);
% number of documents
D = size(bow, 1);
% number of words
W = size(bow, 2);
% initialize P(w|z) as pwgz, make sure that sum_w pwgz = 1
pwgz = rand(W, K);
pwgz = pwgz ./ sum(pwgz, 1);
% initialize P(d|z) as pdgz, make sure that sum_d pdgz = 1
pdgz = rand(D, K);
pdgz = pdgz ./ sum(pdgz, 1);
% initialize P(z) as pz, make sure that sum_z pz = 1
pz = rand(K, 1);
pz = pz ./ sum(pz, 1);
% record pwgz, pdgz and pz
old_pwgz = zeros(W, K) - 1;
old_pdgz = zeros(D, K) - 1;
old_pz = zeros(K, 1) - 1;
%% iterate E&M steps
threshold = 10^(-2);
% tic;
while(norm(pwgz-old_pwgz)>threshold || norm(pdgz-old_pdgz)>threshold || norm(pz-old_pz)>threshold)
    %% record before update
    old_pwgz = pwgz;
    old_pdgz = pdgz;
    old_pz = pz;
    %% E-step
    %% update P(z|d,w) as pzgdw
    pzgdw = zeros(K, D, W);
    for z = 1: K
        for d = 1: D
            for w = 1: W
                pzgdw(z,d,w) = updatePzgdw(z,d,w,pz,pwgz,pdgz);
            end
        end
    end
    %% M-step
    %% helper element Tdw * P(z|d,w)
    TP = getTP(K, D, W, bow, pzgdw);
    %% helper element sum_d sum_w Tdw P(z|d,w)
    swsdTP = getSwSdTP(K, TP);
    swsdTP = sparse(swsdTP);
    %% update pwgz
    for w = 1: W
        for z = 1: K
            pwgz(w, z) = updatePwgz(w,z,TP,swsdTP);
        end
    end
    %% update pdgz
    for d = 1: D
        for z= 1: K
            pdgz(d, z) = updatePdgz(d, z, TP,swsdTP);
        end
    end
    %% update pz
    for z = 1: K
        pz(z, 1) = updatePz(z,swsdTP);
    end
end
% disp(toc);
class = pwgz;
end

%% update P(z|d,w)
function pzgdwATzdw = updatePzgdw(z,d,w,pz,pwgz,pdgz)
    numerator = pz(z, 1) * pwgz(w,z) * pdgz(d,z);
    denominator = 0;
    for zprime = 1: size(pz, 1)
        denominator = denominator + pz(zprime) * pwgz(w,zprime) * pdgz(d,zprime);
    end
    pzgdwATzdw = numerator / denominator;
end    

%% calculate Tdw * P(z|d,w)
function TP = getTP(K, D, W, T, pzgdw)
    TP = zeros(K, D, W);
    for z = 1: K
        % squeeze is important here!
        TP(z,:,:) = T .* sparse(squeeze(pzgdw(z,:,:)));
    end
end

%% calculate sum_w sum_d Tdw * P(z|d,w)
function swsdTP = getSwSdTP(K, TP)
    swsdTP = zeros(K, 1);
    for z = 1: K
        swsdTP(z, 1) = sum(sum(sparse(squeeze(TP(z, :, :)))));
    end
end
%% update P(w|z)
function pwgzATwz = updatePwgz(w,z, TP, swsdTP)
    numerator = sum(TP(z, :, w));
    denominator = swsdTP(z, 1);
    pwgzATwz = numerator/denominator;
end

%% update P(d|z)
function pdgzATdz = updatePdgz(d, z, TP,swsdTP)
    numerator = sum(TP(z,d,:));
    denominator = swsdTP(z, 1);
    pdgzATdz = numerator/denominator;
end

%% update P(z)
function pzATz = updatePz(z,swsdTP)
    numerator = swsdTP(z, 1);
    denominator = sum(swsdTP);
    pzATz = numerator/denominator;
end