function [y_pred, Z, S, W, clusternum] = SAPC(X, spLabel, m, c, r, alpha1)
%% 
% Input: 
%       X: 2D data matrix, each column is a pixel(sample).
%       spLabel: superpixel labels, column vector
%       m: anchor number (superpixel number).
%       c: cluster number.
%       r: Projection dimension.
% Output:
%       y_pred: Predict labels
%       Z: anchor graph, n by m
%       S: anchor-anchor graph, m by m.
%       W: projection matrix.
%       clusternum: The number of connected components of 'S'.
%%

Iter = [15,30];
k = 5;

% init anchor matrix A.
A = meanInd(X, spLabel,m);

% init Z
Z = solveZ(X, A, k);

St = X*X';

for iter1 = 1:Iter(1)
    
    W = updateW(X, A, Z, St, r);
%     W = eye(size(X,1));
    
    U = W' * A;
    
    Z = solveZ(W'*X, U, k);
    
    distU = L2_distance_1(U, U);
    [distU1, idx] = sort(distU, 2);

    % init S
    S = zeros(m);
    GAMMA = 0;
    for i = 1:m
        di = distU1(i,2:k+2); %Exclude itself
        GAMMA = GAMMA + 0.5*(k*di(k+1)-sum(di(1:k)));
        id = idx(i,2:k+2);
        S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
    end
    gamma = GAMMA/m;
    lambda = gamma;
    
    S = (S + S') / 2;
    Ds = diag(sum(S));
    Ls = Ds - S;
    [F, ~, ~]=eig1(Ls, c, 0);
    
    for iter2 = 1:Iter(2)
        S = updateS(U, F, alpha1, gamma, lambda, k);
        
        Ls = diag(sum(S)) - S;
        F_old = F;
        [F, ~, ev] = eig1(Ls, c, 0);
        
        fn1 = sum(ev(1:c));
        fn2 = sum(ev(1:c+1));
        if fn1 > 0.00000000001
            lambda = 2*lambda;
        elseif fn2 < 0.00000000001
            lambda = lambda/2;  
            F = F_old;
        else
            break;
        end
    end
    
    fprintf('%2d,',iter2);

    A = (X*Z)/(diag(sum(Z))+2*alpha1*Ls);

end

%%
% For older versions of MATLAB
% [clusternum, U_label] = graphconncomp(sparse(S));
%%

%%
% For newer versions of MATLAB
[U_label] = conncomp(graph(sparse(S)));
clusternum = length(unique(U_label));
%%

[~, subLabel] = max(Z, [], 2);
y_pred = U_label(subLabel);

end