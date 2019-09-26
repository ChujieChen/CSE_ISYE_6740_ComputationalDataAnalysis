%% Randomized Eigen Computation
function [U, S] = rand_eig(A, k)

[m, n] = size(A);

Omega = randn(n, k); % returns n x k matrix
Y = A * Omega;
[Q, ~] = qr(Y, 0); % Orthogonal-triangular decomposition (QR decomposition)

B = Q' * A * Q;
[U_, S] = eigs(B, k); 
U = Q * U_(:,1:k);
