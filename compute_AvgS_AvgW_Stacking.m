function [PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS,...
          PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW,...
          PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking]    = compute_AvgS_AvgW_Stacking( Xtrain,ytrain,Xtest,Ytest,y1,...
                                  Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG, Prune_and_MinLeafSize_optimal_BS,...
                                  Prune_and_MinLeafSize_optimal_H, Prune_and_MinLeafSize_optimal_KS,...
                                  lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
                                  lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS,...
                                  C_lin_optimal_PCC, C_lin_optimal_AUC, C_lin_optimal_PG, C_lin_optimal_BS, C_lin_optimal_H, C_lin_optimal_KS,...
                                  C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS,...
                                  S_optimal_PCC, S_optimal_AUC, S_optimal_PG, S_optimal_BS, S_optimal_H, S_optimal_KS,...
                                  beta_PCC, beta_AUC, beta_PG, beta_BS, beta_KS, beta_H, optim_w_LRR, optim_b_LRR )
%% This function computes the average PD (probability of default) estimates obtained from the five different base learners and computes the performances on test data.

NumBaseLearners = 5;

PCCm = zeros(size(Xtest,1),NumBaseLearners );
KSm  = zeros(size(Xtest,1),NumBaseLearners );
AUCm = zeros(size(Xtest,1),NumBaseLearners );
PGm  = zeros(size(Xtest,1),NumBaseLearners );
Hm   = zeros(size(Xtest,1),NumBaseLearners );
BSm  = zeros(size(Xtest,1),NumBaseLearners );


%% LR #1
% initialization
[~,numW] = size(Xtrain);
w = rand(numW,1);
b = rand(); 

format short

startingvalues= [w;b];

% Clear any pre-existing options
clearvars options

% Load some options
options  =  optimset('fminunc');
options  =  optimset(options , 'TolFun'      , 1e-6);
options  =  optimset(options , 'TolX'        , 1e-6);
options  =  optimset(options , 'Display'     , 'on');
options  =  optimset(options , 'Diagnostics' , 'on');
options  =  optimset(options , 'LargeScale'  , 'off');
options  =  optimset(options , 'MaxFunEvals' , 10^6) ;
options  =  optimset(options , 'MaxIter'     , 10^6) ; 

% Perform ML maximisation (we actually minimize the negative likelihood)
[MLE,~]= fminunc('NegLogL_LR', startingvalues ,options,Xtrain,ytrain );

% LogL = -NLL;
w = MLE(1:numW);
b = MLE(end);

% compute probability of default predictions for the test set features in
% X2:
predicted_probs =  ( 1./(  1  + exp( -w'*Xtest' - b    ) ) )';

PCCm(:,1) = predicted_probs;
KSm(:,1) = predicted_probs;
AUCm(:,1) = predicted_probs;
PGm(:,1) = predicted_probs;
Hm(:,1) = predicted_probs;
BSm(:,1) = predicted_probs;

%% DT #2

% Train the model with optimal hyper parameters and compute PD estimates.
TreePCC = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_PCC{2} ) );
TreeAUC = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_AUC{2} ) );
TreePG = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_PG{2} ) );
TreeBS = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_BS{2} ) );
TreeH = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize',     round( Prune_and_MinLeafSize_optimal_H{2} ) );
TreeKS = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_KS{2} ) );

[~, predicted_probsPCC,~,~] = predict(TreePCC,Xtest);
PCCm(:,2) = predicted_probsPCC(:,2);

[~, predicted_probsKS,~,~] = predict(TreeKS,Xtest);
KSm(:,2) = predicted_probsKS(:,2);

[~, predicted_probsH,~,~] = predict(TreeH,Xtest);
Hm(:,2) = predicted_probsH(:,2);

[~, predicted_probsPG,~,~] = predict(TreePG,Xtest);
PGm(:,2) = predicted_probsPG(:,2);

[~, predicted_probsAUC,~,~] = predict(TreeAUC,Xtest);
AUCm(:,2) = predicted_probsAUC(:,2);

[~, predicted_probsBS,~,~] = predict(TreeBS,Xtest);
BSm(:,2) = predicted_probsBS(:,2);

%% FNN #3
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
PCCm(:,3) = predicted_probsPCC(1,:)';

predicted_probsKS = netKS(Xtest');
KSm(:,3) = predicted_probsKS(1,:)';

predicted_probsH = netH(Xtest');
Hm(:,3) = predicted_probsH(1,:)';

predicted_probsPG = netPG(Xtest');
PGm(:,3) = predicted_probsPG(1,:)';

predicted_probsAUC = netAUC(Xtest');
AUCm(:,3) = predicted_probsAUC(1,:)';

predicted_probsBS = netBS(Xtest');
BSm(:,3) = predicted_probsBS(1,:)';


%% Linear SVM #4
% Train the linear SVM models with optimal penalty C value and compute the
% performance measures on test data.
trainedSVMModelPCC = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_PCC,'KernelFunction','linear','ClassNames',[0,1]);
trainedSVMModelKS = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_KS,'KernelFunction','linear','ClassNames',[0,1]);
trainedSVMModelAUC = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_AUC,'KernelFunction','linear','ClassNames',[0,1]);
trainedSVMModelPG = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_PG,'KernelFunction','linear','ClassNames',[0,1]);
trainedSVMModelH = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_H,'KernelFunction','linear','ClassNames',[0,1]);
trainedSVMModelBS = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_BS,'KernelFunction','linear','ClassNames',[0,1]);

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

PCCm(:,4) = predicted_probsPCC(:,2);
KSm(:,4) = predicted_probsKS(:,2);
AUCm(:,4) = predicted_probsAUC(:,2);
PGm(:,4) = predicted_probsPG(:,2);
Hm(:,4) = predicted_probsH(:,2);
BSm(:,4) = predicted_probsBS(:,2);

%% Rbf SVM #5

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

PCCm(:,5) = predicted_probsPCC(:,2);
KSm(:,5) = predicted_probsKS(:,2);
AUCm(:,5) = predicted_probsAUC(:,2);
PGm(:,5) = predicted_probsPG(:,2);
Hm(:,5) = predicted_probsH(:,2);
BSm(:,5) = predicted_probsBS(:,2);

%% Compute the avgS PD estimates for each performance measure and compute the PCC AUC PG BS KS and BS values on the test data.

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
PCC_avgS =  sum( (Ytest == YhatPCC) )/numel(Ytest);

prior1 = mean(y1); prior0 = 1 - prior1;

[AUC_avgS,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest, tempAUC, prior1, prior0);

[~,PG_avgS, ~ ] = computeAUC_PGindex_Hvalue(Ytest, tempPG, prior1, prior0);

[~,~, H_avgS ] = computeAUC_PGindex_Hvalue(Ytest, tempH, prior1, prior0);

BS_avgS = mean( (tempBS - Ytest).^2);

KS_avgS = computeKSvalue(Ytest,tempKS);

%% Compute the avgW PD estimates for each performance measure and compute the PCC AUC PG BS KS and BS values on the test data.

tempPCC = PCCm*beta_PCC;
tempKS  = KSm*beta_KS;
tempH   = Hm*beta_H;
tempPG  = PGm*beta_PG;
tempAUC = AUCm*beta_AUC;
tempBS  = BSm*beta_BS;

sortedProbs = sort(tempPCC,'descend'); %sort probabilities
t = sortedProbs(round(mean(y1)*size(sortedProbs,1)));
YhatPCC = tempPCC > t;

% function that computes the PCC, requires true y-values and predicted y-values.
PCC_avgW =  sum( (Ytest == YhatPCC) )/numel(Ytest);

prior1 = mean(y1); prior0 = 1 - prior1;

[AUC_avgW,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest, tempAUC, prior1, prior0);

[~,PG_avgW, ~ ] = computeAUC_PGindex_Hvalue(Ytest, tempPG, prior1, prior0);

[~,~, H_avgW ] = computeAUC_PGindex_Hvalue(Ytest, tempH, prior1, prior0);

BS_avgW = mean( (tempBS - Ytest).^2);

KS_avgW = computeKSvalue(Ytest,tempKS);

%% Compute the Stacking PD estimates for each performance measure and compute the PCC AUC PG BS KS and BS values on the test data.

% First, extract the optimal model parameters of the LR-R 
w_PCC = optim_w_LRR(:,1);
w_AUC = optim_w_LRR(:,2);
w_PG  = optim_w_LRR(:,3);
w_BS  = optim_w_LRR(:,4);
w_KS  = optim_w_LRR(:,5); 
w_H   = optim_w_LRR(:,6);

b_PCC = optim_b_LRR(1);
b_AUC = optim_b_LRR(2);
b_PG  = optim_b_LRR(3);
b_BS  = optim_b_LRR(4);
b_KS  = optim_b_LRR(5);
b_H   = optim_b_LRR(6);

tempPCC =  ( 1./(  1  + exp( -w_PCC'*PCCm' - b_PCC  ) ) )';
tempAUC =  ( 1./(  1  + exp( -w_AUC'*AUCm' - b_AUC  ) ) )';
tempPG =  ( 1./(  1  + exp( -w_PG'*PGm' - b_PG  ) ) )';
tempBS =  ( 1./(  1  + exp( -w_BS'*BSm' - b_BS  ) ) )';
tempKS =  ( 1./(  1  + exp( -w_KS'*KSm' - b_KS  ) ) )';
tempH =  ( 1./(  1  + exp( -w_H'*Hm' - b_H  ) ) )';

sortedProbs = sort(tempPCC,'descend'); %sort probabilities
t = sortedProbs(round(mean(y1)*size(sortedProbs,1)));
YhatPCC = tempPCC > t;

% function that computes the PCC, requires true y-values and predicted y-values.
PCC_Stacking =  sum( (Ytest == YhatPCC) )/numel(Ytest);

prior1 = mean(y1); prior0 = 1 - prior1;

[AUC_Stacking,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest, tempAUC, prior1, prior0);

[~,PG_Stacking, ~ ] = computeAUC_PGindex_Hvalue(Ytest, tempPG, prior1, prior0);

[~,~, H_Stacking ] = computeAUC_PGindex_Hvalue(Ytest, tempH, prior1, prior0);

BS_Stacking = mean( (tempBS - Ytest).^2);

KS_Stacking = computeKSvalue(Ytest,tempKS);

end