function [AUC,PG_index, H_measure ] = computeAUC_PGindex_Hvalue(Ytest, predicted_probs, prior1, prior0)
%% Compute the area under the ROC curve.
% S = numel(Ytest);

alpha = 2; betaparam = 2;

b = 1;

%[orderedProbs,i] = sort(predicted_probs);
% orderdYtest = Ytest(i);

%partialProbs = orderedProbs <= b;

%partialTPRvector  = zeros(sum(partialProbs),1);
%partialFPRvector  = zeros(sum(partialProbs),1);

% construct full ROC curve.
% T is vector of thresholds c from 0 to 1 S+1 times 1 vector.
[FPRvector,TPRvector,T,AUC] = perfcurve(Ytest,predicted_probs, '1');

% not so efficient code piece, so comment it out.
%for s = S:-1:1
 % tau = orderedProbs(s);
 % Yhat = orderedProbs > tau;
 % IndicesFor1 = (orderdYtest ==  1);
 % IndicesFor0 = (orderdYtest ==  0);
  
 % TPRvector(S-s+1) = sum(Yhat(IndicesFor1))/sum(IndicesFor1);
 % FPRvector(S-s+1) = sum(Yhat(IndicesFor0))/sum(IndicesFor0);
%end

% construct partial ROC curve on the interval for which the threshold is <=
% b. (this can gives us the partial AUC and partial Gini index.

partialT = (T <= b);
partialFPRvector = FPRvector(partialT);
partialTPRvector = TPRvector(partialT);

% Comment out this piece of code, inefficient.
%for s = numel(partialTPRvector):-1:1
 % tau = orderedProbs(s);
 % Yhat = orderedProbs(1:numel(partialTPRvector)) > tau;
 % IndicesFor1 = (orderdYtest(1:numel(partialTPRvector)) ==  1);
 % IndicesFor0 = (orderdYtest(1:numel(partialTPRvector)) ==  0);
  
 % partialTPRvector(numel(partialTPRvector)-s+1) = sum(Yhat(IndicesFor1))/sum(IndicesFor1);
 % partialFPRvector(numel(partialTPRvector)-s+1) = sum(Yhat(IndicesFor0))/sum(IndicesFor0);
%end


%plot( FPRvector, TPRvector);

% plot(FPRvector,TPRvector, '-');

partial_AUC = trapz(partialFPRvector, partialTPRvector);
a = partialFPRvector(1);

if AUC >= 0.5
 PG_index = (2*partial_AUC)/((a+1)*(1-a)) -1;
else
 PG_index =  1 - (2*partial_AUC)/((a+1)*(1-a));
end
loopTerminator = 1;

currentTPRvalue = TPRvector(1);
currentFPRvalue = FPRvector(1);

TPRvectorx = TPRvector(2:end);
FPRvectorx = FPRvector(2:end);

index = 1;

Cvector(index) = 0;

index = index + 1;

VectorOfIndices(1) = 1;

I = 1;

while loopTerminator
  % Prior1 is the probability that an observation is of class 1 (i.e.
  % defaulted loan. Prior0 is the probability that an observation is of
  % class 0, that is non defaulted loan.
  CvalueVector  =  ( prior0*( FPRvectorx - currentFPRvalue) )./(prior1*( TPRvectorx - currentTPRvalue  ) + prior0*( FPRvectorx - currentFPRvalue )  );
  [cvalue,i] = min(CvalueVector);
  Cvector(index) = cvalue;
  
  I = I + i;
  VectorOfIndices(index) = I;
  
  index = index + 1;
  
  currentTPRvalue = TPRvectorx(i);
  currentFPRvalue = FPRvectorx(i);
  
  FPRvectorx = FPRvectorx(i+1:end);
  TPRvectorx = TPRvectorx(i+1:end);
  
  if( numel(FPRvectorx)  < 1)
     loopTerminator = 0;
  end
    
end

Cvector = [Cvector';1];

mFPRvector = FPRvector(VectorOfIndices);
mTPRvector = TPRvector(VectorOfIndices);

LhatBeta = 0;

for i = 1:numel(mTPRvector)
  LhatBeta = LhatBeta + prior1*(1 - mTPRvector(i) )*(  betainc(Cvector(i+1),1+alpha,betaparam)*beta(1+alpha,betaparam) - betainc(Cvector(i),1+alpha,betaparam)*beta(1+alpha,betaparam) )/ (betainc(1,alpha,betaparam)*beta(alpha,betaparam) )  + prior0*mFPRvector(i)*(  betainc(Cvector(i+1),alpha,1+betaparam)*beta(alpha,1+betaparam) - betainc(Cvector(i),alpha,1+betaparam)*beta(alpha,1+betaparam) )/ (betainc(1,alpha,betaparam)*beta(alpha,betaparam) ) ;
end
 
H_measure =  1 - (LhatBeta*betainc(1,alpha,betaparam)*beta(alpha,betaparam))/ ( prior1*betainc( prior0,1+alpha,betaparam)*beta(1+alpha,betaparam) +  prior0*betainc( 1,alpha,1+betaparam)*beta(alpha,1+betaparam) - prior0*betainc( prior0,alpha,1+betaparam)*beta(alpha,1+betaparam)   );

end