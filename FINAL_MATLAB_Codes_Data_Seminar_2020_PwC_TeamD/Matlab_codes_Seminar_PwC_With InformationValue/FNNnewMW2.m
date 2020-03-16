function [PCC,AUC, PG,BS, KS, H, Optimal_HP_FNN] = FNNnewMW2(X1,X2,y1,y2)
%% This code uses feedforward neural networks (estimated with backpropagation and mini-batch stochastic gradient descent) to make classifications for the test sample
% we do not make use of 4 fold cross validation (fold for each quarter of each
% year). Also, we will use 5 fold cross validation to fine tune the
% hyperparameters 

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
  
% Do a grid search to find optimal hyperparameters for each measure.
[lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
          lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS] =  gridSearchHyperParamFNNRevised(Xtrain,ytrain,Xvalid,Yvalid,y1);
  
% pre define neural network objects for each measure.      
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

% train the networks on test data with optimally chosen hyperparameters.
[netPCC,~] = train(netPCC,Xtrain', ytrain');
[netKS,~] = train(netKS,Xtrain', ytrain');
[netAUC,~] = train(netAUC,Xtrain', ytrain');
[netPG,~] = train(netPG,Xtrain', ytrain');
[netH,~] = train(netH, Xtrain', ytrain');
[netBS,~] = train(netBS, Xtrain', ytrain');

% Construct new probability score (using default softmax activation function in the output node) on test data.
predicted_probsPCC = netPCC(Xtest');
predicted_probsPCC = predicted_probsPCC(1,:)';

predicted_probsKS = netKS(Xtest');
predicted_probsKS = predicted_probsKS(1,:)';

predicted_probsH = netH(Xtest');
predicted_probsH = predicted_probsH(1,:)';

predicted_probsPG = netPG(Xtest');
predicted_probsPG = predicted_probsPG(1,:)';

predicted_probsAUC = netAUC(Xtest');
predicted_probsAUC = predicted_probsAUC(1,:)';

predicted_probsBS = netBS(Xtest');
predicted_probsBS = predicted_probsBS(1,:)';

% Compute the threshold for classifications.
sortedProbs = sort(predicted_probsPCC,'descend'); %sort probabilities
t = sortedProbs(round(mean(y1)*size(predicted_probsPCC,1)));

YhatPCC = predicted_probsPCC > t;

% Compute the PCC, requires real y-values, predicted_y
% values.
PCC =  sum( (Ytest == YhatPCC) )/numel(Ytest);

prior1 = mean(y1); prior0 = 1 - prior1;

[AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest, predicted_probsAUC, prior1, prior0);

[~,PG, ~ ] = computeAUC_PGindex_Hvalue(Ytest, predicted_probsPG, prior1, prior0);

[~,~, H] = computeAUC_PGindex_Hvalue(Ytest, predicted_probsH, prior1, prior0);

BS = mean( (predicted_probsBS - Ytest).^2);

KS = computeKSvalue(Ytest,predicted_probsKS);

% Store the optimal HyperParameters HP in a matrix and return it as output.
Optimal_HP_FNN = zeros(6,3);

Optimal_HP_FNN(1,:) = lambda_and_hidNodes_optimal_PCC;
Optimal_HP_FNN(2,:) = lambda_and_hidNodes_optimal_AUC;
Optimal_HP_FNN(3,:) = lambda_and_hidNodes_optimal_PG;
Optimal_HP_FNN(4,:) = lambda_and_hidNodes_optimal_BS;
Optimal_HP_FNN(5,:) = lambda_and_hidNodes_optimal_KS;
Optimal_HP_FNN(6,:) = lambda_and_hidNodes_optimal_H;

end