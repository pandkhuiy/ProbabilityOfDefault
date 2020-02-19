function [PCC, AUC, PG,BS, KS, H] = rbfSVMMW2(X1,X2,y1,y2)
%% This code uses support vector machines with radial basis kernel functions to make classifications for the test sample
% where we make use of 4 fold cross validation, that is, we do 4 iterations
% where in each iteration we train the SVM model on three quarter data and
% test it on one test quarter data.
useAdaSyn = 1;
seed = 1;
rng('default');
rng(seed);

if useAdaSyn == 1
    %     index = y1 ==1;
    %     weightVector= ones(length(index),1)*0.25;
    %     weightVector(index) = 0.75;
    %     temp = datasample([X1 y1],5000, 'Replace',false,'Weights', weightVector);
    %     Xtrain123 = temp(:,1:end-1); 
    %     ytrain123 = temp(:,end);
    [Xtrain,ytrain] =   BalancedData(X1,y1); % Xtrain123 and ytrain123 have 10000 samples.
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

% Determine the optimal penalty constant C and optimal rbd kernel scale parameter s, with k fold cross validation
% with k = 5 on the training data.
[~, ~, ~, ~, ~, ~, ...
  ~, ~, ~, ~, ~, ~, PCC, AUC, PG,BS, KS, H] =  gridSearchHyperParamRbfSVM(Xtrain, ytrain,Xtest,Ytest);

end