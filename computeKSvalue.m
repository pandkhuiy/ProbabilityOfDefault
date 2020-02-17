function [KS_value] = computeKSvalue(Ytest,predicted_probs)
%% Compute the Kolmogorov-Smirnov statistic

[orderedProbs,i] = sort(predicted_probs);
orderdYtest = Ytest(i);

index = 1;

for a = 0:0.01:1
    
  
  IndicesFor1 = (orderdYtest ==  1);
  IndicesFor0 = (orderdYtest ==  0);
  n = sum(IndicesFor1);
  m = sum(IndicesFor0);
  
  Fn_default(index) = (1/n)*(sum(orderedProbs(IndicesFor1) <= a) );
  Fm_nondefault(index) = (1/m)*(sum(orderedProbs(IndicesFor0) <= a) );
  
  index = index + 1;
end

Fm_nondefault_minus_Fn_default_vector = abs(Fm_nondefault - Fn_default);

KS_value = max( Fm_nondefault_minus_Fn_default_vector );

end