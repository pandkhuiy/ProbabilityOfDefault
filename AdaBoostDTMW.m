function [PCC, AUC, PG,BS, KS, H] = AdaBoostDTMW(X1,X2,y1,y2,Optimal_HP_DT )
%% This code implements the boosting method on decision trees to make classifications on the final test sample.
% we use boosting with optimally trained Decision trees (for MinLeafSize
% and whether we prune or not).

% extract the optimal Hyperparameters given as input:
Prune_and_MinLeafSize_optimal_PCC = Optimal_HP_DT(1,:); 
Prune_and_MinLeafSize_optimal_AUC = Optimal_HP_DT(2,:);
Prune_and_MinLeafSize_optimal_PG = Optimal_HP_DT(3,:); 
Prune_and_MinLeafSize_optimal_BS = Optimal_HP_DT(4,:); 
Prune_and_MinLeafSize_optimal_KS = Optimal_HP_DT(5,:); 
Prune_and_MinLeafSize_optimal_H = Optimal_HP_DT(6,:); 

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
  
% Determine optimal number of DT bags for each performance measure using validation
% data with a linear grid search, NumBags : [10 50 100 250 500 1000].
[cycles_optimal_PCC, cycles_optimal_AUC, cycles_optimal_PG, cycles_optimal_BS, cycles_optimal_H, cycles_optimal_KS] =  gridSearchHyperParamBoostingDT(Xtrain,ytrain, Xvalid,Yvalid,y1,...
                   Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG,...
                   Prune_and_MinLeafSize_optimal_BS, Prune_and_MinLeafSize_optimal_KS, Prune_and_MinLeafSize_optimal_H);
  
% Use the boosting method for Decision trees with previously obtained
% optimal hyperparameters for each performance measure.
templateTreePCC = templateTree('Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PCC{2} );
templateTreeAUC = templateTree('Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_AUC{2} );
templateTreePG = templateTree('Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PG{2} );
templateTreeBS = templateTree('Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_BS{2} );
templateTreeKS = templateTree('Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_KS{2} );
templateTreeH = templateTree('Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_H{2} );

MdlPCC = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreePCC,'NumLearningCycles',cycles_optimal_PCC,'ScoreTransform','logit');
MdlAUC = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreeAUC,'NumLearningCycles',cycles_optimal_AUC,'ScoreTransform','logit');
MdlPG = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreePG,'NumLearningCycles',cycles_optimal_PG,'ScoreTransform','logit');
MdlBS = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreeBS,'NumLearningCycles',cycles_optimal_BS,'ScoreTransform','logit');
MdlKS = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreeKS,'NumLearningCycles',cycles_optimal_KS,'ScoreTransform','logit');
MdlH = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreeH,'NumLearningCycles',cycles_optimal_H,'ScoreTransform','logit');

% make predictions with the boosted ensembles for each measure.
[~,predictedProbsPCC] = predict(MdlPCC,Xtest);
tempPCC = predictedProbsPCC(:,2);

[~,predictedProbsAUC] = predict(MdlAUC,Xtest);
tempAUC = predictedProbsAUC(:,2);

[~,predictedProbsPG] = predict(MdlPG,Xtest);
tempPG = predictedProbsPG(:,2);

[~,predictedProbsBS] = predict(MdlBS,Xtest);
tempBS = predictedProbsBS(:,2);

[~,predictedProbsKS] = predict(MdlKS,Xtest);
tempKS = predictedProbsKS(:,2);

[~,predictedProbsH] = predict(MdlH,Xtest);
tempH = predictedProbsH(:,2);

%sort the predicted probs for the PCC measure
sortedProbs = sort(tempPCC,'descend'); 
t = sortedProbs(round(mean(y1)*size(sortedProbs,1)));
YhatPCC = tempPCC > t;

% function that computes the PCC, requires true y-values and predicted y-values.
PCC =  sum( (Ytest == YhatPCC) )/numel(Ytest);

prior1 = mean(y1); prior0 = 1 - prior1;

[AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest, tempAUC, prior1, prior0);

[~,PG, ~ ] = computeAUC_PGindex_Hvalue(Ytest, tempPG, prior1, prior0);

[~,~, H] = computeAUC_PGindex_Hvalue(Ytest, tempH, prior1, prior0);

BS = mean( (tempBS - Ytest).^2);

KS = computeKSvalue(Ytest,tempKS);

end
