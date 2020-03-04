function [PCC,AUC, PG,BS, KS, H] = LRMW2(X1,X2,y1,y2)
%% This code implements Logistic regression used for Moving window training/testing quarterly.
% We use 4 fold cross validation to compute the performance measures.

useAdaSyn = 0;
seed = 1;
rng('default');
rng(seed);


% Start with X1 and Y1 as training sets, and X4 and Y4 as test sets.
if useAdaSyn == 1
% Construct (synthetic if possible) dataset such that 50% has 1s and 0s
% Xtrain123 and ytrain123 have 10000 samples.
[Xtrain,ytrain] =   BalancedData(X1,y1); 
else
number = 0;
while number <5
temp = datasample([X1 y1],10000, 'Replace',false);
Xtrain = temp(:,1:end-1); 
ytrain = temp(:,end);
number = sum(ytrain);
end
end

number = 0;
while number <5 
temp = datasample([X2 y2],10000, 'Replace',false);
Xtest = temp(:,1:end-1);
Ytest = temp(:,end);
number = sum(Ytest);
end

% initialization
[~,numW] = size(Xtrain);
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
[MLE,~]= fminunc('NegLogL_LR', startingvalues ,options,Xtrain,ytrain );

% LogL = -NLL;
w = MLE(1:numW);
b = MLE(end);

% compute probability of default predictions for the test set features in
% X2:
predicted_probs =  ( 1./(  1  + exp( -w'*Xtest' - b    ) ) )';
% Make predictions based on t, the fraction of defaulted loans in the
% training set Y1. predicted_probs > t, then yhat2 = 1.
LR1predsort = sort(predicted_probs,'descend'); %sort probabilities
t = LR1predsort(round(mean(y1)*size(predicted_probs,1)));

Yhat2 = predicted_probs > t;

% function that computes the PCC, requires real y-values, predicted_y
% values.
PCC =  sum( (Ytest == Yhat2) )/numel(Ytest);

prior1 = mean(ytrain); prior0 = 1 - prior1;

[AUC,PG, H ] = computeAUC_PGindex_Hvalue(Ytest, predicted_probs, prior1, prior0);

BS = mean( (predicted_probs - Ytest).^2);
KS = computeKSvalue(Ytest,predicted_probs);


end