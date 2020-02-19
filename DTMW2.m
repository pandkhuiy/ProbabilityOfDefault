function [PCC,AUC, PG,BS, KS, H] = DTMW2(X1,X2,y1,y2)
%% This code use the Decisions Tree function fitctree to make classifications for the test sample
% we make use of 4 fold cross validation (fold for each quarter of each
% year). Also, we will use 5 fold cross validation to fine tune the
% hyperparameters a the validation set which

useAdaSyn = 1;
seed = 1;
rng('default');
rng(seed);

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
  
% Do grid search.
[~, ~, ~, ~,...
      ~, ~,PCC,AUC, PG,BS, KS, H] =  gridSearchHyperParamDT(Xtrain,ytrain,Xtest,Ytest);


end