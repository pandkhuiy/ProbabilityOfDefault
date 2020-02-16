function [PCC,AUC, PG,BS, KS, H] = LRMW(X1,X2,y1,y2)
%% This code implements Logistic regression used for Moving window training/testing quarterly.
% We use 4 fold cross validation to compute the performance measures.

useAdaSyn = 1;
seed = 1;
rng('default');
rng(seed);

%X = normalize(X);

% This vector is (P*2 by 1) and represents the PCC values for each cv
% iteration. Same for the other five performance measures.
PCC_vector = zeros(1,1);
AUC_vector = zeros(1,1);
PGini_vector = zeros(1,1);
BScore_vector = zeros(1,1);
Hmeasure_vector = zeros(1,1);
KSvalue_vector = zeros(1,1);


% Start with X123 and Y123 as training sets, and X4 and Y4 as test sets.
 
 % Start with X1 and Y1 as training sets, and X4 and Y4 as test sets.
    if useAdaSyn == 1
        number = 0;
        while number <5
            temp = datasample([X1 y1],5000, 'Replace',false);
            Xtrain123 = temp(:,1:end-1); 
            ytrain123 = temp(:,end);
            number = sum(ytrain123);
        end
    else
        number = 0;
        while number <5
            temp = datasample([X1 y1],10000, 'Replace',false);
            Xtrain123 = temp(:,1:end-1); 
            ytrain123 = temp(:,end);
            number = sum(ytrain123);
        end
    end
  
number = 0;
while number <5 
    temp = datasample([X2 y2],5000, 'Replace',false);
    Xtest4 = temp(:,1:end-1);
    Ytest4 = temp(:,end);
    number = sum(Ytest4);
end

 if useAdaSyn == 1
   
   [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
   
   % initialization
  [~,numW] = size(Xtrain123);
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
  [MLE,~]= fminunc('NegLogL_LR', startingvalues ,options,[XAdaSyn;Xtrain123],[double(yAda);ytrain123] );

 % LogL = -NLL;
  w = MLE(1:numW);
  b = MLE(end);
  
  % compute probability of default predictions for the test set features in
  % Xtest4:
  predicted_probs =  ( 1./(  1  + exp( -w'*Xtest4' - b    ) ) )';
  % Make predictions based on t, the fraction of defaulted loans in the
  % synthetic training set Y1. predicted_probs > t, then yhat2 = 1.
  t = mean([double(yAda);ytrain123]);
  Yhat2 = predicted_probs > t;
  
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Ytest4 == Yhat2) )/numel(Ytest4);
  PCC_vector(1) = PCC;
  
  prior1 = mean([double(yAda);ytrain123]); prior0 = 1 - prior1;
  
  [AUC,PG_index, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probs, prior1, prior0);
  
  AUC_vector(1) = AUC;
  PGini_vector(1) = PG_index;
  Hmeasure_vector(1) = H_measure;
  BScore = mean( (predicted_probs - Ytest4).^2);
  BScore_vector(1) = BScore;
 
  KS_value = computeKSvalue(Ytest4,predicted_probs);
  KSvalue_vector(1) = KS_value;
   
 else
  % initialization
  [~,numW] = size(Xtrain123);
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
  [MLE,~]= fminunc('NegLogL_LR', startingvalues ,options,Xtrain123,ytrain123 );

 % LogL = -NLL;
  w = MLE(1:numW);
  b = MLE(end);
  
  % compute probability of default predictions for the test set features in
  % X2:
  predicted_probs =  ( 1./(  1  + exp( -w'*Xtest4' - b    ) ) )';
  % Make predictions based on t, the fraction of defaulted loans in the
  % training set Y1. predicted_probs > t, then yhat2 = 1.
  t = mean(ytrain123);
  Yhat2 = predicted_probs > t;
  
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Ytest4 == Yhat2) )/numel(Ytest4);
  PCC_vector(1) = PCC;
  
  prior1 = mean(ytrain123); prior0 = 1 - prior1;
  
  [AUC,PG_index, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probs, prior1, prior0);
  
  AUC_vector(1) = AUC;
  PGini_vector(1) = PG_index;
  Hmeasure_vector(1) = H_measure;
  BScore = mean( (predicted_probs - Ytest4).^2);
  BScore_vector(1) = BScore;
 
  KS_value = computeKSvalue(Ytest4,predicted_probs);
  KSvalue_vector(1) = KS_value;
 end
 
 
PCC = mean(PCC_vector);
AUC = mean(AUC_vector);
PG  = mean(PGini_vector);
BS  = mean( BScore_vector);
H   = mean( Hmeasure_vector);
KS  = mean( KSvalue_vector);
 
end