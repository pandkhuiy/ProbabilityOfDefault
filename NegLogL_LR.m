function [LL] = NegLogL_LR(param_vec,X,y)
%% compute the negative logL value

w = param_vec(1:(end-1));
b = param_vec(end);

[n,k] = size(X);

prob_y1 = zeros(n,1);

LL = 0;

for i = 1:n
prob_y1(i) = 1/(1 +  ( exp(-w'*X(i,:)' -b) ) ) ;

LL = LL +y(i)*log(prob_y1(i)) + (1-y(i))*log(1-prob_y1(i)) ;

end

LL = -LL;

end
