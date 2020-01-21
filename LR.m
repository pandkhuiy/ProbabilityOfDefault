function [PCC,BScore,w,b, LogL] = LR(X,y)
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
  
  %AUC = computeAUC( Y2, predicted_probs);
  %AUC_vector(l) = AUC;
  
  % Reading the paper how to compute PGini_vector = zeros(P*2,1);
  BScore = mean( (predicted_probs - Y2).^2    );
  BScore_vector(l) = BScore;
  % Reading the paper how to compute Hmeasure = computeHmeasure(P*2,1);
  % Reading the paper how to compute KSvalue = computeKSvalue(P*2,1);
  % Reading the paper how to compute PGini = computePGini(P*2,1);
  
  
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
  
  %AUC = computeAUC( Y1, predicted_probs);
  
  %AUC_vector(l+3) = AUC;
  
  BScore = mean( (predicted_probs - Y1).^2    );
  BScore_vector(l+3) = BScore;
  % Reading the paper how to compute Hmeasure = computeHmeasure(P*2,1);
  % Reading the paper how to compute KSvalue = computeKSvalue(P*2,1);
  % Reading the paper how to compute PGini = computePGini(P*2,1);
  
 end
 
 
end