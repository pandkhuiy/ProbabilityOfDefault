function [Bags_optimal_PCC, Bags_optimal_AUC, Bags_optimal_PG, Bags_optimal_BS, Bags_optimal_H, Bags_optimal_KS] =  gridSearchHyperParamBaggingFNN(Xtrain,ytrain, Xvalid,Yvalid,y1,...
                   lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG,...
                   lambda_and_hidNodes_optimal_BS, lambda_and_hidNodes_optimal_KS, lambda_and_hidNodes_optimal_H)
%% This function determines the optimal number bags for bagging multiple Feedforward neural networks with optimal hyperparameters (given as input)
NumberBagsVector = [5 10 25 100];

% columns are the six performance metrics and the rows are the four values
% of the hyperparameter No. of bags.
% PCC, KS ----- AUC, Hmeasure, PG index ----- BS (in this order).
PerformanceMeasuresMatrix =  zeros(numel(NumberBagsVector),6);

% Pre define patternnet objects with optimal regularization penalty values
% and optimal number of nodes for each performance measure given as input.
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

% Do grid search for number of bags
for c = 1:length(NumberBagsVector)
    c
    
    PCCm = zeros(size(Xvalid,1),NumberBagsVector(c));
    KSm  = zeros(size(Xvalid,1),NumberBagsVector(c));
    AUCm = zeros(size(Xvalid,1),NumberBagsVector(c));
    PGm  = zeros(size(Xvalid,1),NumberBagsVector(c));
    Hm   = zeros(size(Xvalid,1),NumberBagsVector(c));
    BSm  = zeros(size(Xvalid,1),NumberBagsVector(c));

    for b = 1:NumberBagsVector(c)
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

        [netPCC,~] = train(netPCC,bsx', bsy');
        [netKS,~] = train(netKS,bsx', bsy');
        [netAUC,~] = train(netAUC,bsx', bsy');
        [netPG,~] = train(netPG,bsx', bsy');
        [netH,~] = train(netH, bsx', bsy');
        [netBS,~] = train(netBS, bsx', bsy');

        % Construct new probability score (using the softmax activation function in the output node and the tanh function in hidden layer)
        % on test data Xtest and Ytest.
        predicted_probsPCC = netPCC(Xvalid');
        predicted_probsPCC = predicted_probsPCC(1,:)';
        PCCm(:,b) = predicted_probsPCC;

        predicted_probsKS = netKS(Xvalid');
        predicted_probsKS = predicted_probsKS(1,:)';
        KSm(:,b) = predicted_probsKS;

        predicted_probsH = netH(Xvalid');
        predicted_probsH = predicted_probsH(1,:)';
        Hm(:,b) = predicted_probsH;

        predicted_probsPG = netPG(Xvalid');
        predicted_probsPG = predicted_probsPG(1,:)';
        PGm(:,b) = predicted_probsPG;

        predicted_probsAUC = netAUC(Xvalid');
        predicted_probsAUC = predicted_probsAUC(1,:)';
        AUCm(:,b) = predicted_probsAUC;

        predicted_probsBS = netBS(Xvalid');
        predicted_probsBS = predicted_probsBS(1,:)';
        BSm(:,b) = predicted_probsBS;

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

    % function that computes the PCC, requires real y-values, predicted_y
    % values.
    PCC =  sum( (Yvalid == YhatPCC) )/numel(Yvalid);
    PerformanceMeasuresMatrix(c,1) = PCC;

    prior1 = mean(y1); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Yvalid, tempAUC, prior1, prior0);
    PerformanceMeasuresMatrix(c,3) = AUC;


    [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Yvalid, tempPG, prior1, prior0);
    PerformanceMeasuresMatrix(c,5) = PG_index;


    [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Yvalid, tempH, prior1, prior0);
    PerformanceMeasuresMatrix(c,4) = H_measure ;


    BScore = mean( (tempBS - Yvalid).^2);
    PerformanceMeasuresMatrix(c,6) = BScore;

    KS_value = computeKSvalue(Yvalid,tempKS);
    PerformanceMeasuresMatrix(c,2) = KS_value ;

end

% extract the indices of the corresponding optimal parameter C for each
% measure:
[~, ind] = max(PerformanceMeasuresMatrix(:,1)); % PCC
Bags_optimal_PCC =  NumberBagsVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,2)); % KS
Bags_optimal_KS =  NumberBagsVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,3)); % AUC
Bags_optimal_AUC =  NumberBagsVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,4)); % Hmeasure
Bags_optimal_H =  NumberBagsVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,5)); % PG_index
Bags_optimal_PG =  NumberBagsVector(ind);
[~, ind] = min(PerformanceMeasuresMatrix(:,6)); % BS
Bags_optimal_BS =  NumberBagsVector(ind);

end