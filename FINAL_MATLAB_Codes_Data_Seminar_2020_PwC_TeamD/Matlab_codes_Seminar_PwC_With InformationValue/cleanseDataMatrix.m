function [Xnew,ynew] = cleanseDataMatrix(X,y)
%% Get rid of Nan's and +/- inf values in the X matrix.

% first NaNs
 indices = any(isnan(X), 2);
 X  = X(logical(1-indices), :);
 y  = y(logical(1-indices), :);
 
 indices = any(X == -inf, 2);
 X  = X(logical(1-indices), :);
 y  = y(logical(1-indices), :);
 
 indices = any(X == inf, 2);
 Xnew  = X(logical(1-indices), :);
 ynew  = y(logical(1-indices), :);
 

end