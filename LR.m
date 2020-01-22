function [PCC,AUC, PG,BS, KS, H] = LR(X,y)
%% This code implements Logistic regression
% We use Px2 cross validation, where P = 3 means we use three times 2-fold cv.
P = 3;

seed = 1;
rng('default');
rng(seed);

% This vector is (P*2 by 1) and represents the PCC values for each cv
% iteration. Same for the other five performance measures.
PCC_vector = zeros(P*2,1);
AUC_vector = zeros(P*2,1);
PGini_vector = zeros(P*2,1);
BScore_vector = zeros(P*2,1);
Hmeasure_vector = zeros(P*2,1);
KSvalue_vector = zeros(P*2,1);

 for  l = 1:P
  k = randperm(size(X,1));
 
  % X1 X2 Y1 and Y2 are training and test sets that can interchanged for
  % our 2 fold cross validation for each outerloop of index l.
  X1 = X(k(1:size(X,1)/2),:);
  X2 = X(k( ((size(X,1)/2)+1): end), :);
 
  Y1 = y(k(1:size(X,1)/2));
  Y2 = y(k(((size(X,1)/2)+1): end));

  
  % Start with X1 and Y1 as training sets, and X2 and Y2 as test sets.
  % initialization
  [~,numW] = size(X1);
  w = rand(numW,1);
  b = rand(); 

  format short

  startingvalues= [w;b];

  % Clear any pre-existing options
  clearvars options

  % Load some options
  options  =  optimset('fminunc');
  options  =  optimset(options , 'TolFun'      , 1e-6);
  options  =  optimset(options , 'TolX'        , 1e-6);
  options  =  optimset(options , 'Display'     , 'on');
  options  =  optimset(options , 'Diagnostics' , 'on');
  options  =  optimset(options , 'LargeScale'  , 'off');
  options  =  optimset(options , 'MaxFunEvals' , 10^6) ;
  options  =  optimset(options , 'MaxIter'     , 10^6) ; 

  % Perform ML maximisation (we actually minimize the negative likelihood)
  [MLE,NLL]= fminunc('NegLogL_LR', startingvalues ,options,X1,Y1 );

  LogL = -NLL;
  w = MLE(1:numW);
  b = MLE(end);
  
  % compute probability of default predictions for the test set features in
  % X2:
  predicted_probs =  ( 1./(  1  + exp( -w'*X2' - b    ) ) )';
  % Make predictions based on t, the fraction of defaulted loans in the
  % training set Y1. predicted_probs > t, then yhat2 = 1.
  t = mean(Y1);
  Yhat2 = predicted_probs > t;
  
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Y2 == Yhat2) )/numel(Y2);
  PCC_vector(l) = PCC;
  
  prior1 = mean(Y1); prior0 = 1 - prior1;
  
  [AUC,PG_index, H_measure ] = computeAUC_PGindex_Hvalue(Y2, predicted_probs, prior1, prior0);
  
  AUC_vector(l) = AUC;
  PGini_vector(l) = PG_index;
  Hmeasure_vector(l) = H_measure;
  BScore = mean( (predicted_probs - Y2).^2);
  BScore_vector(l) = BScore;
 
  KS_value = computeKSvalue(Y2,predicted_probs);
  KSvalue_vector(l) = KS_value;
  
  %%
  % Reverse the roles: use X2 and Y2 as training sets, and X1 and Y1 as test sets.
  % initialization
  [~,numW] = size(X2);
  w = rand(numW,1);
  b = rand(); 

  format short

  startingvalues= [w;b];

  % Clear any pre-existing options
  clearvars options

  % Load some options
  options  =  optimset('fminunc');
  options  =  optimset(options , 'TolFun'      , 1e-6);
  options  =  optimset(options , 'TolX'        , 1e-6);
  options  =  optimset(options , 'Display'     , 'on');
  options  =  optimset(options , 'Diagnostics' , 'on');
  options  =  optimset(options , 'LargeScale'  , 'off');
  options  =  optimset(options , 'MaxFunEvals' , 10^6) ;
  options  =  optimset(options , 'MaxIter'     , 10^6) ; 

  % Perform ML maximisation (we actually minimize the negative likelihood)
  [MLE,NLL]= fminunc('NegLogL_LR', startingvalues ,options,X2,Y2 );

  LogL = -NLL;
  w = MLE(1:numW);
  b = MLE(end);
  
  % compute probability of default predictions for the test set features in
  % X1:
  predicted_probs =  ( 1./(  1  + exp( -w'*X1' - b    ) ) )';
  % Make predictions based on t, the fraction of defaulted loans in the
  % training set Y2. predicted_probs > t, then yhat1 = 1.
  t = mean(Y2);
  Yhat1 = predicted_probs > t;
  
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Y1 == Yhat1) )/numel(Y1);
  PCC_vector(l+3) = PCC;
  
  BScore = mean( (predicted_probs - Y1).^2    );
  
  prior1 = mean(Y2); prior0 = 1 - prior1;
  
  [AUC,PG_index, H_measure ] = computeAUC_PGindex_Hvalue(Y1, predicted_probs, prior1, prior0);
  
  AUC_vector(l+3) = AUC;
  PGini_vector(l+3) = PG_index;
  Hmeasure_vector(l+3) = H_measure;
  BScore_vector(l+3) = BScore;
  
  KS_value = computeKSvalue(Y1,predicted_probs);
  KSvalue_vector(l+3) = KS_value;
  
 end
 
PCC = mean(PCC_vector);
AUC = mean(AUC_vector);
PG  = mean(PGini_vector);
BS  = mean( BScore_vector);
H   = mean( Hmeasure_vector);
KS  = mean( KSvalue_vector);
 
end