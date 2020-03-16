function [PCC, AUC, PG,BS, KS, H] = BaggingDTMW(X1,X2,y1,y2, Optimal_HP_DT )
%% This code implements bagging ensemble on decision trees to make classifications for the test sample
% The number of bags is optimized with a linear grid search.

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
% data with a linear grid search : NumBags : [ 5 10 25 50 100 1000].
[Bags_optimal_PCC, Bags_optimal_AUC, Bags_optimal_PG, Bags_optimal_BS, Bags_optimal_H, Bags_optimal_KS] =  gridSearchHyperParamBaggingDT(Xtrain,ytrain, Xvalid,Yvalid,y1,...
                   Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG,...
                   Prune_and_MinLeafSize_optimal_BS, Prune_and_MinLeafSize_optimal_KS, Prune_and_MinLeafSize_optimal_H);
  

NumberBagsVector = [Bags_optimal_PCC, Bags_optimal_AUC, Bags_optimal_PG, Bags_optimal_BS, Bags_optimal_KS, Bags_optimal_H];               
               
% Construct matrix of PD estimates across bootstrap samples (columns) with
% test data.
PCCm = zeros(size(Xtest,1),Bags_optimal_PCC);
KSm  = zeros(size(Xtest,1),Bags_optimal_KS);
AUCm = zeros(size(Xtest,1),Bags_optimal_AUC);
PGm  = zeros(size(Xtest,1),Bags_optimal_PG);
Hm   = zeros(size(Xtest,1),Bags_optimal_H);
BSm  = zeros(size(Xtest,1),Bags_optimal_BS);

for i = 1:length(NumberBagsVector)

    for b = 1:NumberBagsVector(i)
        % Construct a bootstrap sample with replacement.
        TrainMatrix=[Xtrain,ytrain];
        Lengthdata=size(TrainMatrix,1);
        
        number = 0;
        while number <5 
            temp1=ceil(Lengthdata*rand(Lengthdata,1));
            bsdata=TrainMatrix(temp1,:);
            bsx=bsdata(:,1:end-1);
            bsy=bsdata(:,end);
            number = sum(bsy);
        end
             
        % Construct new probability score on test data.
        % PCC
        if i == 1
            TreePCC = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PCC{2} );
            [~, predicted_probsPCC,~,~] = predict(TreePCC,Xtest);
            predicted_probsPCC = predicted_probsPCC(:,2);
            PCCm(:,b) = predicted_probsPCC;
        end
        % AUC
        if i == 2
            TreeAUC = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_AUC{2} );
            [~, predicted_probsAUC,~,~] = predict(TreeAUC,Xtest);
            predicted_probsAUC = predicted_probsAUC(:,2);
            AUCm(:,b) = predicted_probsAUC;
        end
        % KS
        if i == 5
            TreeKS = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize',   Prune_and_MinLeafSize_optimal_KS{2} );
            [~, predicted_probsKS,~,~] = predict(TreeKS,Xtest);
            predicted_probsKS = predicted_probsKS(:,2);
            KSm(:,b) = predicted_probsKS;
        end
        % BS
        if i == 4
            TreeBS = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize',   Prune_and_MinLeafSize_optimal_BS{2} );
            [~, predicted_probsBS,~,~] = predict(TreeBS,Xtest);
            predicted_probsBS = predicted_probsBS(:,2);
            BSm(:,b) = predicted_probsBS;
        end
        % PG
        if i == 3
            TreePG = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize',   Prune_and_MinLeafSize_optimal_PG{2} );
            [~, predicted_probsPG,~,~] = predict(TreePG,Xtest);
            predicted_probsPG = predicted_probsPG(:,2);
            PGm(:,b) = predicted_probsPG;
        end
         % H measure
        if i == 6
            TreeH = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize',     Prune_and_MinLeafSize_optimal_H{2} );
            [~, predicted_probsH,~,~] = predict(TreeH,Xtest);
            predicted_probsH = predicted_probsH(:,2);
            Hm(:,b) = predicted_probsH;
        end
    end
end

tempPCC = mean(PCCm,2);
tempKS=mean(KSm,2);
tempH=mean(Hm,2);
tempPG=mean(PGm,2);
tempAUC=mean(AUCm,2);
tempBS=mean(BSm,2);

sortedProbs = sort(tempPCC,'descend'); %sort probabilities
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
