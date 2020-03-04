function [PCC,AUC, PG,BS, KS, H, Optimal_HP_DT] = DTMW2(X1,X2,y1,y2)
%% This code use the Decisions Tree function fitctree to make classifications for the test sample
% we make use of 4 fold cross validation (fold for each quarter of each
% year). Also, we will use 5 fold cross validation to fine tune the
% hyperparameters a the validation set which

useAdaSyn = 0;
seed = 1;
rng('default');
rng(seed);

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
  
% Do grid search.
[Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG, Prune_and_MinLeafSize_optimal_BS,...
          Prune_and_MinLeafSize_optimal_H, Prune_and_MinLeafSize_optimal_KS] =  gridSearchHyperParamDT(Xtrain,ytrain,Xvalid,Yvalid, y1);

% Train the model with optimal hyper parameters and compute the performance
% values on the test data.
TreePCC = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_PCC{2} ) );
TreeAUC = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_AUC{2} ) );
TreePG = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_PG{2} ) );
TreeBS = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_BS{2} ) );
TreeH = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize',     round( Prune_and_MinLeafSize_optimal_H{2} ) );
TreeKS = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_KS{2} ) );

[~, predicted_probsPCC,~,~] = predict(TreePCC,Xtest);
predicted_probsPCC = predicted_probsPCC(:,2);

[~, predicted_probsKS,~,~] = predict(TreeKS,Xtest);
predicted_probsKS = predicted_probsKS(:,2);

[~, predicted_probsH,~,~] = predict(TreeH,Xtest);
predicted_probsH = predicted_probsH(:,2);

[~, predicted_probsPG,~,~] = predict(TreePG,Xtest);
predicted_probsPG = predicted_probsPG(:,2);

[~, predicted_probsAUC,~,~] = predict(TreeAUC,Xtest);
predicted_probsAUC = predicted_probsAUC(:,2);

[~, predicted_probsBS,~,~] = predict(TreeBS,Xtest);
predicted_probsBS = predicted_probsBS(:,2);


sortedProbs = sort(predicted_probsPCC,'descend'); %sort probabilities
t = sortedProbs(round(mean(y1)*size(predicted_probsPCC,1)));

YhatPCC = predicted_probsPCC > t;

% function that computes the PCC, requires real y-values, predicted_y
% values.
PCC =  sum( (Ytest == YhatPCC) )/numel(Ytest);

prior1 = mean(y1); prior0 = 1 - prior1;

[AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest, predicted_probsAUC, prior1, prior0);

[~,PG, ~ ] = computeAUC_PGindex_Hvalue(Ytest, predicted_probsPG, prior1, prior0);

[~,~, H] = computeAUC_PGindex_Hvalue(Ytest, predicted_probsH, prior1, prior0);

BS = mean( (predicted_probsBS - Ytest).^2);

KS = computeKSvalue(Ytest, predicted_probsKS);      

% Store the optimal HPs in a cell array
Optimal_HP_DT = cell(6,2);

Optimal_HP_DT(1,:) = Prune_and_MinLeafSize_optimal_PCC;
Optimal_HP_DT(2,:) = Prune_and_MinLeafSize_optimal_AUC;
Optimal_HP_DT(3,:) = Prune_and_MinLeafSize_optimal_PG;
Optimal_HP_DT(4,:) = Prune_and_MinLeafSize_optimal_BS;
Optimal_HP_DT(5,:) = Prune_and_MinLeafSize_optimal_KS;
Optimal_HP_DT(6,:) = Prune_and_MinLeafSize_optimal_H;

end