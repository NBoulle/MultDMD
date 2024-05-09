function [B,W,C,Sumd] = construct_basis(X, Y, N, varargin)
% X is an M x d matrix of inputs, where M is the number of samples and d is
% the spatial dimension
% Y is an M x d matrix of outputs
% N is the dimension of the piecewise constant basis
% The last parameter is a set used to select the centroids

% Get parameter values
M = size(X,1);

if nargin >= 4
    X_kmeans = varargin{1};
else
    X_kmeans = X;
end

% Construct centroids using kmeans
if N == size(X,1)
    C = X;
    Sumd = zeros(size(X,1),1);
else
    [~, C, Sumd] = kmeans(X_kmeans, N, 'display', 'iter');
end

% Quadrature weights for Monte-Carlo integration
w = ones(M,1)/M;

% Create W_ij matrix
xi = knnsearch(C,X);
xj = knnsearch(C,Y);

[H,ID1,ID2] = findgroups(xi,xj);
T = accumarray(H, w);
W = sparse(ID1,ID2,T);
W = full(W);

% Compute vector q = sum of W along each row
G = sum(W,2);

% Compute matrix of minimization problem
M1 = (G - 2*full(W))./G';

% Construct B matrix
[~, I] = min(M1, [], 2); % Find index of minimum along each row
Bi = 1:N;
v = ones(size(I));
B = sparse(Bi, I, v, N, N);
end