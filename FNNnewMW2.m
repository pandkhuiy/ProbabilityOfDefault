function [PCC,AUC, PG,BS, KS, H] = FNNnewMW2(X1,X2,y1,y2)
%% This code uses feedforward neural networks (estimated with backpropagation and mini-batch stochastic gradient descent) to make classifications for the test sample
% we do not make use of 4 fold cross validation (fold for each quarter of each
% year). Also, we will use 5 fold cross validation to fine tune the
% hyperparameters 

useAdaSyn = 1;
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
  
% Do a grid search to find optimal hyperparameters for each measure.

[~, ~, ~, ~,...
      ~, ~,PCC,AUC, PG,BS, KS, H] =  gridSearchHyperParamFNNRevised(Xtrain,ytrain,Xtest,Ytest);


end