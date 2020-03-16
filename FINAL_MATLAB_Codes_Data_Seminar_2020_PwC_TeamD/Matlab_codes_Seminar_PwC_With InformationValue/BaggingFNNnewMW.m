function [PCC, AUC, PG,BS, KS, H] = BaggingFNNnewMW(X1,X2,y1,y2, Optimal_HP_FNN )
%% This code implements bagging ensemble on feedforward neural networks (FNNS) to make classifications for the test sample
% The number of bags is optimized with a linear grid search.
% The hyper parameters of the neural networks used for bagging were all set to the
% previously obtained optimized values with a 2d grid search.

% extract the optimal Hyperparameters given as input:
lambda_and_hidNodes_optimal_PCC = Optimal_HP_FNN(1,:); 
lambda_and_hidNodes_optimal_AUC = Optimal_HP_FNN(2,:);
lambda_and_hidNodes_optimal_PG  = Optimal_HP_FNN(3,:); 
lambda_and_hidNodes_optimal_BS  = Optimal_HP_FNN(4,:); 
lambda_and_hidNodes_optimal_KS  = Optimal_HP_FNN(5,:); 
lambda_and_hidNodes_optimal_H   = Optimal_HP_FNN(6,:); 

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
    while number1 <5 || number2 <5
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

% Determine optimal number of FNN bags for each performance measure using a
% linear grid search on the validation set, i.e., NumBags : [5 10 25 100].
[Bags_optimal_PCC, Bags_optimal_AUC, Bags_optimal_PG, Bags_optimal_BS, Bags_optimal_H, Bags_optimal_KS] =  gridSearchHyperParamBaggingFNN(Xtrain,ytrain, Xvalid,Yvalid,y1,...
                   lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG,...
                   lambda_and_hidNodes_optimal_BS, lambda_and_hidNodes_optimal_KS, lambda_and_hidNodes_optimal_H);

% Pre define patternet objects for each measure.
netPCC = patternnet(lambda_and_hidNodes_optimal_PCC(2));
netKS = patternnet(lambda_and_hidNodes_optimal_KS(2));
netAUC = patternnet(lambda_and_hidNodes_optimal_AUC(2));
netPG = patternnet(lambda_and_hidNodes_optimal_PG(2));
netH = patternnet(lambda_and_hidNodes_optimal_H(2));
netBS = patternnet(lambda_and_hidNodes_optimal_BS(2));

% Pre-set some neural networks options to our experimental design.
% criterion, methods, etc.
%netPCC.divideFcn = 'divideblock'; % this function divides the training set into a training set and validation set.
netPCC.divideParam.trainRatio = 100/100;
netPCC.divideParam.valRatio = 0/100;
netPCC.divideParam.testRatio = 0/100;
netPCC.performParam.regularization = lambda_and_hidNodes_optimal_PCC(1); % set regularization parameter to prevent overfitting.
netPCC.biasConnect(2)= 1;
netPCC.layers{1}.transferFcn = 'tansig'; %default activation function in hidden layer is tansig function.
netPCC.layers{2}.transferFcn = 'softmax';
netPCC.trainFcn = 'trainscg'; %default function is 'trainscg', that is stochastic conjugate gradient, we can change it to 'trainbfg'.
netPCC.performFcn = 'crossentropy';% default criterion is 'crossentropy', we can change it to 'mse' to chech which is more suitable.
netPCC.trainParam.showWindow = 0;
netPCC.trainParam.epochs = lambda_and_hidNodes_optimal_PCC(3);

%netKS.divideFcn = 'divideblock'; 
netKS.divideParam.trainRatio = 100/100;
netKS.divideParam.valRatio = 0/100;
netKS.divideParam.testRatio = 0/100;
netKS.performParam.regularization = lambda_and_hidNodes_optimal_KS(1); 
netKS.biasConnect(2)= 1; 
netKS.layers{1}.transferFcn = 'tansig'; %default activation function in hidden layer is tansig function.
netKS.layers{2}.transferFcn = 'softmax';
netKS.trainFcn = 'trainscg'; %default function is 'trainscg', that is stochastic conjugate gradient, we can change it to 'trainbfg'.
netKS.performFcn = 'crossentropy';% default criterion is 'crossentropy', we can change it to 'mse' to chech which is more suitable.
netKS.trainParam.showWindow = 0;
netKS.trainParam.epochs = lambda_and_hidNodes_optimal_KS(3);

%netAUC.divideFcn = 'divideblock'; 
netAUC.divideParam.trainRatio = 100/100;
netAUC.divideParam.valRatio = 0/100;
netAUC.divideParam.testRatio = 0/100;
netAUC.trainParam.showWindow = 0;
netAUC.performParam.regularization = lambda_and_hidNodes_optimal_AUC(1);
netAUC.biasConnect(2)= 1;
netAUC.layers{1}.transferFcn = 'tansig'; %default activation function in hidden layer is tansig function.
netAUC.layers{2}.transferFcn = 'softmax';
netAUC.trainFcn = 'trainscg'; %default function is 'trainscg', that is stochastic conjugate gradient, we can change it to 'trainbfg'.
netAUC.performFcn = 'crossentropy';% default criterion is 'crossentropy', we can change it to 'mse' to chech which is more suitable.
netAUC.trainParam.showWindow = 0;
netAUC.trainParam.epochs = lambda_and_hidNodes_optimal_AUC(3);

%netPG.divideFcn = 'divideblock'; 
netPG.divideParam.trainRatio = 100/100;
netPG.divideParam.valRatio = 0/100;
netPG.divideParam.testRatio = 0/100;
netPG.performParam.regularization = lambda_and_hidNodes_optimal_PG(1);
netPG.biasConnect(2)= 1;
netPG.layers{1}.transferFcn = 'tansig'; %default activation function in hidden layer is tansig function.
netPG.layers{2}.transferFcn = 'softmax';
netPG.trainFcn = 'trainscg'; %default function is 'trainscg', that is stochastic conjugate gradient, we can change it to 'trainbfg'.
netPG.performFcn = 'crossentropy';% default criterion is 'crossentropy', we can change it to 'mse' to chech which is more suitable.
netPG.trainParam.showWindow = 0;
netPG.trainParam.epochs = lambda_and_hidNodes_optimal_PG(3);

%netH.divideFcn = 'divideblock';
netH.divideParam.trainRatio = 100/100;
netH.divideParam.valRatio = 0/100;
netH.divideParam.testRatio = 0/100;
netH.performParam.regularization = lambda_and_hidNodes_optimal_H(1); 
netH.biasConnect(2)= 1;
netH.layers{1}.transferFcn = 'tansig'; %default activation function in hidden layer is tansig function.
netH.layers{2}.transferFcn = 'softmax';
netH.trainFcn = 'trainscg'; %default function is 'trainscg', that is stochastic conjugate gradient, we can change it to 'trainbfg'.
netH.performFcn = 'crossentropy';% default criterion is 'crossentropy', we can change it to 'mse' to chech which is more suitable.
netH.trainParam.showWindow = 0;
netH.trainParam.epochs = lambda_and_hidNodes_optimal_H(3);

%netBS.divideFcn = 'divideblock'; 
netBS.divideParam.trainRatio = 100/100;
netBS.divideParam.valRatio = 0/100;
netBS.divideParam.testRatio = 0/100;
netBS.performParam.regularization = lambda_and_hidNodes_optimal_BS(1);
netBS.biasConnect(2)= 1;
netBS.layers{1}.transferFcn = 'tansig'; %default activation function in hidden layer is tansig function.
netBS.layers{2}.transferFcn = 'softmax';
netBS.trainFcn = 'trainscg'; %default function is 'trainscg', that is stochastic conjugate gradient, we can change it to 'trainbfg'.
netBS.performFcn = 'crossentropy';% default criterion is 'crossentropy', we can change it to 'mse' to chech which is more suitable.
netBS.trainParam.showWindow = 0;      
netBS.trainParam.epochs = lambda_and_hidNodes_optimal_BS(3);

NumberBagsVector = [Bags_optimal_PCC, Bags_optimal_AUC, Bags_optimal_PG, Bags_optimal_BS, Bags_optimal_KS, Bags_optimal_H]; 

% Construct matrix of PD estimates across bootstrap samples (columns) with
% test data.
PCCm = zeros(size(Xtest,1),Bags_optimal_PCC);
KSm  = zeros(size(Xtest,1),Bags_optimal_KS);
AUCm = zeros(size(Xtest,1),Bags_optimal_AUC);
PGm  = zeros(size(Xtest,1),Bags_optimal_PG);
Hm   = zeros(size(Xtest,1),Bags_optimal_H);
BSm  = zeros(size(Xtest,1),Bags_optimal_BS);

% use optimal bags for each performance measure and compute the performance
% values on test data.
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

        % Construct new probability score on test data
        % PCC
        if i == 1
            [netPCC,~] = train(netPCC,bsx', bsy');
            predicted_probsPCC = netPCC(Xtest');
            predicted_probsPCC = predicted_probsPCC(1,:)';
            PCCm(:,b) = predicted_probsPCC;
        end
        % KS
        if i == 5
            [netKS,~] = train(netKS,bsx', bsy');
            predicted_probsKS = netKS(Xtest');
            predicted_probsKS = predicted_probsKS(1,:)';
            KSm(:,b) = predicted_probsKS;
        end
        % H measure
        if i == 6
            [netH,~] = train(netH, bsx', bsy');
            predicted_probsH = netH(Xtest');
            predicted_probsH = predicted_probsH(1,:)';
            Hm(:,b) = predicted_probsH;
        end
        % PG
        if i == 3
            [netPG,~] = train(netPG,bsx', bsy');
            predicted_probsPG = netPG(Xtest');
            predicted_probsPG = predicted_probsPG(1,:)';
            PGm(:,b) = predicted_probsPG;
        end
        % AUC
        if i == 2
            [netAUC,~] = train(netAUC,bsx', bsy');
            predicted_probsAUC = netAUC(Xtest');
            predicted_probsAUC = predicted_probsAUC(1,:)';
            AUCm(:,b) = predicted_probsAUC;
        end
        % BS
        if i == 4
            [netBS,~] = train(netBS, bsx', bsy');
            predicted_probsBS = netBS(Xtest');
            predicted_probsBS = predicted_probsBS(1,:)';
            BSm(:,b) = predicted_probsBS;
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
    
    % mean(predicted_probsPCC)
    % function that computes the PCC, requires real y-values, predicted_y
    % values.
    PCC =  sum( (Ytest == YhatPCC) )/numel(Ytest);
    
    prior1 = mean(y1); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest, tempAUC, prior1, prior0);

    [~,PG, ~ ] = computeAUC_PGindex_Hvalue(Ytest, tempPG, prior1, prior0);

    [~,~, H] = computeAUC_PGindex_Hvalue(Ytest, tempH, prior1, prior0);
   
    BS = mean( (tempBS - Ytest).^2);
   
    KS = computeKSvalue(Ytest,tempKS);
  
end

end
