function [PCC, AUC, PG,BS, KS, H, Optimal_HP_RbfSVM] = rbfSVMMW2(X1,X2,y1,y2)
%% This code uses support vector machines with radial basis kernel functions to make classifications for the test sample
% where we make use of 4 fold cross validation, that is, we do 4 iterations
% where in each iteration we train the SVM model on three quarter data and
% test it on one test quarter data.
useAdaSyn = 0;
seed = 1;
rng('default');
rng(seed);

% Generate random Xtest and Ytest from the second quarter dataset
% representing the original test set.
number = 0;
while number <5 
    temp = datasample([X2 y2],10000, 'Replace',false);
    Xtest = temp(:,1:end-1);
    Ytest = temp(:,end);
    number = sum(Ytest);
end

% Generate first a validation set Xval,Yval from the first quarter data
% set. Use the remaining observation for generating the training sets
% Xtrain and ytrain dependent on whether we use AdaSyn+undersampling or
% not.
if useAdaSyn == 1
    number = 0;
    while number <5 
        [temp, ind] = datasample([X1 y1],10000, 'Replace',false);
        Xvalid = temp(:,1:end-1); 
        Yvalid = temp(:,end);
        
        missIndex  = setdiff(1:size(X1,1),ind);
        X1remaining = X1(missIndex,:);
        y1remaining = y1(missIndex);
        
        % Use all the minority class examples for the training set
        % (undersampling) and if #minority examples is < 5000, fill the
        % remaining training sets with ADASYNed minority class
        % observations such that final Xtrain and ytrain has APPROXIMATELY 50% has 1s and 50% 0s
        % Xtrain and ytrain have 10000 samples.
        [Xtrain,ytrain] =   BalancedData(X1remaining,y1remaining); 
        
        number = sum(Yvalid);
    end
else
    number1 = 0;
    number2 = 0;
    while number1 <5 && number2 <5
        [temp, ind] = datasample([X1 y1],10000, 'Replace',false);
        Xvalid = temp(:,1:end-1); 
        Yvalid = temp(:,end);
        
        missIndex  = setdiff(1:size(X1,1),ind);
        X1remaining = X1(missIndex,:);
        y1remaining = y1(missIndex);
        
        % Use the remaining observations of the original training data X1 and y1
        % to generate 10000 random examples: Xtrain and ytrain
        temp = datasample([X1remaining y1remaining],10000, 'Replace',false);
        Xtrain = temp(:,1:end-1); 
        ytrain = temp(:,end);
             
        number1 = sum(ytrain);
        number2 = sum(Yvalid);
    end
end

% Determine the optimal penalty constant C and optimal rbd kernel scale
% parameter s, with a 2d grid search on vaidation data.
[C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS,...
    S_optimal_PCC, S_optimal_AUC, S_optimal_PG, S_optimal_BS, S_optimal_H, S_optimal_KS] =  gridSearchHyperParamRbfSVM(Xtrain, ytrain,Xvalid,Yvalid,y1);

% Now fit SVMs with rbf kernel with the optimal hyperparameters for each performance
% measures and compute the performance values on test data.
trainedSVMModelPCC = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_optimal_PCC,'KernelFunction','rbf','KernelScale',S_optimal_PCC,'ClassNames',[0,1]);
trainedSVMModelKS = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_optimal_KS,'KernelFunction','rbf','KernelScale',S_optimal_KS,'ClassNames',[0,1]);
trainedSVMModelAUC = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_optimal_AUC,'KernelFunction','rbf','KernelScale',S_optimal_AUC,'ClassNames',[0,1]);
trainedSVMModelPG = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_optimal_PG,'KernelFunction','rbf','KernelScale',S_optimal_H,'ClassNames',[0,1]);
trainedSVMModelH = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_optimal_H,'KernelFunction','rbf','KernelScale',S_optimal_PG,'ClassNames',[0,1]);
trainedSVMModelBS = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_optimal_BS,'KernelFunction','rbf','KernelScale',S_optimal_BS,'ClassNames',[0,1]);

trainedSVMModelPCC = fitPosterior(trainedSVMModelPCC, Xtrain,ytrain);
trainedSVMModelKS = fitPosterior(trainedSVMModelKS, Xtrain,ytrain);
trainedSVMModelAUC = fitPosterior(trainedSVMModelAUC, Xtrain,ytrain);
trainedSVMModelPG = fitPosterior(trainedSVMModelPG, Xtrain,ytrain);
trainedSVMModelH = fitPosterior(trainedSVMModelH, Xtrain,ytrain);
trainedSVMModelBS = fitPosterior(trainedSVMModelBS, Xtrain,ytrain);

[~,predicted_probsPCC] = predict(trainedSVMModelPCC, Xtest);
[~,predicted_probsKS] = predict(trainedSVMModelKS, Xtest);
[~,predicted_probsAUC] = predict(trainedSVMModelAUC, Xtest);
[~,predicted_probsPG] = predict(trainedSVMModelPG, Xtest);
[~,predicted_probsH] = predict(trainedSVMModelH, Xtest); 
[~,predicted_probsBS] = predict(trainedSVMModelBS, Xtest); 

sortedProbs = sort(predicted_probsPCC(:,2),'descend'); %sort probabilities
t = sortedProbs(round(mean(y1)*size(predicted_probsPCC,1)));

YhatPCC = predicted_probsPCC(:,2) > t;

% function that computes the PCC, requires real y-values, predicted_y values.
PCC =  sum( (Ytest == YhatPCC) )/numel(Ytest);

prior1 = mean(y1); prior0 = 1 - prior1;

[AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest, predicted_probsAUC(:,2), prior1, prior0);

[~,PG, ~ ] = computeAUC_PGindex_Hvalue(Ytest, predicted_probsPG(:,2), prior1, prior0);

[~,~, H] = computeAUC_PGindex_Hvalue(Ytest, predicted_probsH(:,2), prior1, prior0);

BS = mean( (predicted_probsBS(:,2) - Ytest).^2);

KS = computeKSvalue(Ytest,predicted_probsKS(:,2));

% Store the optimal HPs in a matrix and return it as output.
Optimal_HP_RbfSVM = zeros(6,2);

Optimal_HP_RbfSVM(1,1) = C_optimal_PCC;
Optimal_HP_RbfSVM(1,2) = S_optimal_PCC;

Optimal_HP_RbfSVM(2,1) = C_optimal_AUC;
Optimal_HP_RbfSVM(2,2) = S_optimal_AUC;

Optimal_HP_RbfSVM(3,1) = C_optimal_PG;
Optimal_HP_RbfSVM(3,2) = S_optimal_PG;

Optimal_HP_RbfSVM(4,1) = C_optimal_BS;
Optimal_HP_RbfSVM(4,2) = S_optimal_BS;

Optimal_HP_RbfSVM(5,1) = C_optimal_KS;
Optimal_HP_RbfSVM(5,2) = S_optimal_KS;

Optimal_HP_RbfSVM(6,1) = C_optimal_H;
Optimal_HP_RbfSVM(6,2) = S_optimal_H;

end