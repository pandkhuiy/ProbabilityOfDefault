function [AUC,PG_index, H_measure ] = computeAUC_PGindex_Hvalue(Ytest, predicted_probs, prior1, prior0)
%% Compute the area under the ROC curve.
S = numel(Ytest);

alpha = 2; betaparam = 2;

b = 0.5;

[orderedProbs,i] = sort(predicted_probs);
orderdYtest = Ytest(i);

TPRvector  = zeros(S,1);
FPRvector  = zeros(S,1);

partialProbs = orderedProbs <= b;

partialTPRvector  = zeros(sum(partialProbs),1);
partialFPRvector  = zeros(sum(partialProbs),1);

% construct full ROC curve.
for s = S:-1:1
  tau = orderedProbs(s);
  Yhat = orderedProbs > tau;
  IndicesFor1 = (orderdYtest ==  1);
  IndicesFor0 = (orderdYtest ==  0);
  
  TPRvector(S-s+1) = sum(Yhat(IndicesFor1))/sum(IndicesFor1);
  FPRvector(S-s+1) = sum(Yhat(IndicesFor0))/sum(IndicesFor0);
end

% construct partial ROC curve on the interval [0,b].
for s = numel(partialTPRvector):-1:1
  tau = orderedProbs(s);
  Yhat = orderedProbs(1:numel(partialTPRvector)) > tau;
  IndicesFor1 = (orderdYtest(1:numel(partialTPRvector)) ==  1);
  IndicesFor0 = (orderdYtest(1:numel(partialTPRvector)) ==  0);
  
  partialTPRvector(numel(partialTPRvector)-s+1) = sum(Yhat(IndicesFor1))/sum(IndicesFor1);
  partialFPRvector(numel(partialTPRvector)-s+1) = sum(Yhat(IndicesFor0))/sum(IndicesFor0);
end

% Add the last pair: true positive rate and false positive rate of both 1,
% and this corresponding to the threshold of tau=0.
TPRvector = [TPRvector;1];
FPRvector = [FPRvector;1];

partialTPRvector = [partialTPRvector;1];
partialFPRvector = [partialFPRvector;1];

% plot(FPRvector,TPRvector, '-');

AUC = trapz(FPRvector, TPRvector);
PG_index = trapz(partialFPRvector, partialTPRvector);

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