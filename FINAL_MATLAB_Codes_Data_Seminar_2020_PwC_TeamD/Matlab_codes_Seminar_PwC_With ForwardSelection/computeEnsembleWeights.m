function [beta_PCC, beta_AUC, beta_PG, beta_BS, beta_KS, beta_H, optim_w_LRR, optim_b_LRR ]  = computeEnsembleWeights( Xtrain,ytrain,Xvalid, Yvalid,Xvalid2,Yvalid2,y1,...
                                Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG, Prune_and_MinLeafSize_optimal_BS,...
                                Prune_and_MinLeafSize_optimal_H, Prune_and_MinLeafSize_optimal_KS,...
                                lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
                                lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS,...
                                C_lin_optimal_PCC, C_lin_optimal_AUC, C_lin_optimal_PG, C_lin_optimal_BS, C_lin_optimal_H, C_lin_optimal_KS,...
                                C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS,...
                                S_optimal_PCC, S_optimal_AUC, S_optimal_PG, S_optimal_BS, S_optimal_H, S_optimal_KS , useAdaSyn )
%% This function computes the weights for the base learners for each performance measure in the avgW ensemble method.
% The criterion is considered as a hyperparameter for the avgW ensemble
% method. However, as mentioned in Lessmann et al. (2015) the optimal
% criterion for each performance measure is the performance measure itself.
% For example, for the PCC we will use the PCC criterion for computing the
% performance measure based weights for the five base learners: LR, DT,
% FNN, SVM-L and SVM-Rbf. It does not make sense to use for example the KS
% criterion to construct optimal baselearner weights to maximize the PCC value 
% in the avgW ensemble.

NumBaseLearners = 5;

beta_PCC  = zeros(NumBaseLearners,1 );
beta_AUC  = zeros(NumBaseLearners,1 );
beta_PG   = zeros(NumBaseLearners,1 );
beta_BS   = zeros(NumBaseLearners,1 );
beta_KS   = zeros(NumBaseLearners,1 );
beta_H    = zeros(NumBaseLearners,1 );

% PD estimates matrix for validation set 1
PCCm = zeros(size(Xvalid,1),NumBaseLearners );
KSm  = zeros(size(Xvalid,1),NumBaseLearners );
AUCm = zeros(size(Xvalid,1),NumBaseLearners );
PGm  = zeros(size(Xvalid,1),NumBaseLearners );
Hm   = zeros(size(Xvalid,1),NumBaseLearners );
BSm  = zeros(size(Xvalid,1),NumBaseLearners );

% PD estimates matrix for validation set 2, which is used for HP
% optimization of the LR-R meta classifier.
PCCm2 = zeros(size(Xvalid2,1),NumBaseLearners );
KSm2  = zeros(size(Xvalid2,1),NumBaseLearners );
AUCm2 = zeros(size(Xvalid2,1),NumBaseLearners );
PGm2  = zeros(size(Xvalid2,1),NumBaseLearners );
Hm2   = zeros(size(Xvalid2,1),NumBaseLearners );
BSm2  = zeros(size(Xvalid2,1),NumBaseLearners );

if useAdaSyn == 1
    [Xada,yada] = ADASYN(Xvalid,Yvalid,1,[],[],false);
    
    % PD estimates matrix for balanced validation set 1
    PCCmb = zeros(size([Xada;Xvalid],1),NumBaseLearners );
    KSmb  = zeros(size([Xada;Xvalid],1),NumBaseLearners );
    AUCmb = zeros(size([Xada;Xvalid],1),NumBaseLearners );
    PGmb  = zeros(size([Xada;Xvalid],1),NumBaseLearners );
    Hmb   = zeros(size([Xada;Xvalid],1),NumBaseLearners );
    BSmb  = zeros(size([Xada;Xvalid],1),NumBaseLearners );
end
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
predicted_probs =  ( 1./(  1  + exp( -w'*Xvalid' - b    ) ) )';
% Make predictions based on t, the fraction of defaulted loans in the
% training set Y1. predicted_probs > t, then yhat2 = 1.
sortedProbs = sort(predicted_probs,'descend'); %sort probabilities
t = sortedProbs(round(mean(y1)*size(predicted_probs,1)));

Yhat2 = predicted_probs > t;

% function that computes the PCC, requires real y-values, predicted_y
% values.
beta_PCC(1) =  sum( (Yvalid == Yhat2) )/numel(Yvalid);
%recall = sum( (Ytest(Ytest == 1) == Yhat2(Ytest == 1) ) )/sum(Ytest)

prior1 = mean(y1); prior0 = 1 - prior1;

[AUC,PG, H ] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probs, prior1, prior0);
beta_AUC(1) = AUC;
beta_PG(1) = PG;
beta_H(1) = H;

BS = mean( (predicted_probs - Yvalid).^2);
beta_BS(1) = BS;

KS = computeKSvalue(Yvalid,predicted_probs);
beta_KS(1) = KS;

PCCm(:,1) = predicted_probs;
KSm(:,1) = predicted_probs;
AUCm(:,1) = predicted_probs;
PGm(:,1) = predicted_probs;
Hm(:,1) = predicted_probs;
BSm(:,1) = predicted_probs;

% If we use AdaSyn+Undersampling then, we construct probability estimates
% with balanced validation set. Use these estimates for training the meta
% classifier on balanced validation set.
if useAdaSyn == 1
    predicted_probs =  ( 1./(  1  + exp( -w'*[Xada;Xvalid]' - b    ) ) )';
   
    PCCmb(:,1) = predicted_probs;
    KSmb(:,1) = predicted_probs;
    AUCmb(:,1) = predicted_probs;
    PGmb(:,1) = predicted_probs;
    Hmb(:,1) = predicted_probs;
    BSmb(:,1) = predicted_probs;

end

%% DT #2

% Train the model with optimal hyper parameters and compute the performance
% values on the test data.
TreePCC = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_PCC{2} ) );
TreeAUC = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_AUC{2} ) );
TreePG = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_PG{2} ) );
TreeBS = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_BS{2} ) );
TreeH = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize',     round( Prune_and_MinLeafSize_optimal_H{2} ) );
TreeKS = fitctree(Xtrain,ytrain,'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_KS{2} ) );

[~, predicted_probsPCC,~,~] = predict(TreePCC,Xvalid);
predicted_probsPCC = predicted_probsPCC(:,2);

[~, predicted_probsKS,~,~] = predict(TreeKS,Xvalid);
predicted_probsKS = predicted_probsKS(:,2);

[~, predicted_probsH,~,~] = predict(TreeH,Xvalid);
predicted_probsH = predicted_probsH(:,2);

[~, predicted_probsPG,~,~] = predict(TreePG,Xvalid);
predicted_probsPG = predicted_probsPG(:,2);

[~, predicted_probsAUC,~,~] = predict(TreeAUC,Xvalid);
predicted_probsAUC = predicted_probsAUC(:,2);

[~, predicted_probsBS,~,~] = predict(TreeBS,Xvalid);
predicted_probsBS = predicted_probsBS(:,2);


sortedProbs = sort(predicted_probsPCC,'descend'); %sort probabilities
t = sortedProbs(round(mean(y1)*size(predicted_probsPCC,1)));

YhatPCC = predicted_probsPCC > t;

% function that computes the PCC, requires real y-values, predicted_y
% values.
beta_PCC(2) =  sum( (Yvalid == YhatPCC) )/numel(Yvalid);

prior1 = mean(y1); prior0 = 1 - prior1;

[beta_AUC(2),~, ~ ] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsAUC, prior1, prior0);

[~,beta_PG(2), ~ ] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsPG, prior1, prior0);

[~,~, beta_H(2)] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsH, prior1, prior0);

beta_BS(2) = mean( (predicted_probsBS - Yvalid).^2);

beta_KS(2) = computeKSvalue(Yvalid, predicted_probsKS);   

PCCm(:,2) = predicted_probsPCC;
KSm(:,2) = predicted_probsKS;
Hm(:,2) = predicted_probsH;
PGm(:,2) = predicted_probsPG;
AUCm(:,2) = predicted_probsAUC;
BSm(:,2) = predicted_probsBS;

% If we use AdaSyn+Undersampling then, we construct probability estimates
% with balanced validation set. Use these estimates for training the meta
% classifier on balanced validation set.
if useAdaSyn == 1
    [~, predicted_probsPCC,~,~] = predict(TreePCC,[Xada;Xvalid]);
    predicted_probsPCC = predicted_probsPCC(:,2);

    [~, predicted_probsKS,~,~] = predict(TreeKS,[Xada;Xvalid]);
    predicted_probsKS = predicted_probsKS(:,2);

    [~, predicted_probsH,~,~] = predict(TreeH,[Xada;Xvalid]);
    predicted_probsH = predicted_probsH(:,2);

    [~, predicted_probsPG,~,~] = predict(TreePG,[Xada;Xvalid]);
    predicted_probsPG = predicted_probsPG(:,2);

    [~, predicted_probsAUC,~,~] = predict(TreeAUC,[Xada;Xvalid]);
    predicted_probsAUC = predicted_probsAUC(:,2);

    [~, predicted_probsBS,~,~] = predict(TreeBS,[Xada;Xvalid]);
    predicted_probsBS = predicted_probsBS(:,2);
   
    PCCmb(:,2) = predicted_probsPCC;
    KSmb(:,2) = predicted_probsKS;
    Hmb(:,2) = predicted_probsH;
    PGmb(:,2) = predicted_probsPG;
    AUCmb(:,2) = predicted_probsAUC;
    BSmb(:,2) = predicted_probsBS;

end

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
predicted_probsPCC = netPCC(Xvalid');
predicted_probsPCC = predicted_probsPCC(1,:)';

predicted_probsKS = netKS(Xvalid');
predicted_probsKS = predicted_probsKS(1,:)';

predicted_probsH = netH(Xvalid');
predicted_probsH = predicted_probsH(1,:)';

predicted_probsPG = netPG(Xvalid');
predicted_probsPG = predicted_probsPG(1,:)';

predicted_probsAUC = netAUC(Xvalid');
predicted_probsAUC = predicted_probsAUC(1,:)';

predicted_probsBS = netBS(Xvalid');
predicted_probsBS = predicted_probsBS(1,:)';

% Compute the threshold for classifications.
sortedProbs = sort(predicted_probsPCC,'descend'); %sort probabilities
t = sortedProbs(round(mean(y1)*size(predicted_probsPCC,1)));

YhatPCC = predicted_probsPCC > t;

% Compute the PCC, requires real y-values, predicted_y
% values.
beta_PCC(3) =  sum( (Yvalid == YhatPCC) )/numel(Yvalid);

prior1 = mean(y1); prior0 = 1 - prior1;

[beta_AUC(3),~, ~ ] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsAUC, prior1, prior0);

[~,beta_PG(3), ~ ] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsPG, prior1, prior0);

[~,~, beta_H(3)] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsH, prior1, prior0);

beta_BS(3) = mean( (predicted_probsBS - Yvalid).^2);

beta_KS(3) = computeKSvalue(Yvalid,predicted_probsKS);

PCCm(:,3) = predicted_probsPCC;
KSm(:,3) = predicted_probsKS;
Hm(:,3) = predicted_probsH;
PGm(:,3) = predicted_probsPG;
AUCm(:,3) = predicted_probsAUC;
BSm(:,3) = predicted_probsBS;

% If we use AdaSyn+Undersampling then, we construct probability estimates
% with balanced validation set. Use these estimates for training the meta
% classifier on balanced validation set.
if useAdaSyn == 1
    predicted_probsPCC = netPCC([Xada;Xvalid]');
    predicted_probsPCC = predicted_probsPCC(1,:)';

    predicted_probsKS = netKS([Xada;Xvalid]');
    predicted_probsKS = predicted_probsKS(1,:)';

    predicted_probsH = netH([Xada;Xvalid]');
    predicted_probsH = predicted_probsH(1,:)';

    predicted_probsPG = netPG([Xada;Xvalid]');
    predicted_probsPG = predicted_probsPG(1,:)';

    predicted_probsAUC = netAUC([Xada;Xvalid]');
    predicted_probsAUC = predicted_probsAUC(1,:)';

    predicted_probsBS = netBS([Xada;Xvalid]');
    predicted_probsBS = predicted_probsBS(1,:)';
   
    PCCmb(:,3) = predicted_probsPCC;
    KSmb(:,3) = predicted_probsKS;
    Hmb(:,3) = predicted_probsH;
    PGmb(:,3) = predicted_probsPG;
    AUCmb(:,3) = predicted_probsAUC;
    BSmb(:,3) = predicted_probsBS;

end

%% Linear SVM #4

% Train the linear SVM models with optimal penalty C value and compute the
% performance measures on test data.
trainedlinSVMModelPCC = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_PCC,'KernelFunction','linear','ClassNames',[0,1]);
trainedlinSVMModelKS = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_KS,'KernelFunction','linear','ClassNames',[0,1]);
trainedlinSVMModelAUC = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_AUC,'KernelFunction','linear','ClassNames',[0,1]);
trainedlinSVMModelPG = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_PG,'KernelFunction','linear','ClassNames',[0,1]);
trainedlinSVMModelH = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_H,'KernelFunction','linear','ClassNames',[0,1]);
trainedlinSVMModelBS = fitcsvm(Xtrain,ytrain,'BoxConstraint',C_lin_optimal_BS,'KernelFunction','linear','ClassNames',[0,1]);

trainedlinSVMModelPCC = fitPosterior(trainedlinSVMModelPCC, Xtrain,ytrain);
trainedlinSVMModelKS = fitPosterior(trainedlinSVMModelKS, Xtrain,ytrain);
trainedlinSVMModelAUC = fitPosterior(trainedlinSVMModelAUC, Xtrain,ytrain);
trainedlinSVMModelPG = fitPosterior(trainedlinSVMModelPG, Xtrain,ytrain);
trainedlinSVMModelH = fitPosterior(trainedlinSVMModelH, Xtrain,ytrain);
trainedlinSVMModelBS = fitPosterior(trainedlinSVMModelBS, Xtrain,ytrain);

[~,predicted_probsPCC] = predict(trainedlinSVMModelPCC, Xvalid);
[~,predicted_probsKS] = predict(trainedlinSVMModelKS, Xvalid);
[~,predicted_probsAUC] = predict(trainedlinSVMModelAUC, Xvalid);
[~,predicted_probsPG] = predict(trainedlinSVMModelPG, Xvalid);
[~,predicted_probsH] = predict(trainedlinSVMModelH, Xvalid); 
[~,predicted_probsBS] = predict(trainedlinSVMModelBS, Xvalid); 

sortedProbs = sort(predicted_probsPCC(:,2),'descend'); %sort probabilities
t = sortedProbs(round(mean(y1)*size(predicted_probsPCC,1)));

YhatPCC = predicted_probsPCC(:,2) > t;
% function that computes the PCC, requires real y-values, predicted_y
% values.
beta_PCC(4) =  sum( (Yvalid == YhatPCC) )/numel(Yvalid);

prior1 = mean(y1); prior0 = 1 - prior1;

[beta_AUC(4),~, ~ ] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsAUC(:,2), prior1, prior0);

[~,beta_PG(4), ~ ] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsPG(:,2), prior1, prior0);

[~,~, beta_H(4)] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsH(:,2), prior1, prior0);

beta_BS(4) = mean( (predicted_probsBS(:,2) - Yvalid).^2);

beta_KS(4) = computeKSvalue(Yvalid,predicted_probsKS(:,2));

PCCm(:,4) = predicted_probsPCC(:,2);
KSm(:,4) = predicted_probsKS(:,2);
AUCm(:,4) = predicted_probsAUC(:,2);
PGm(:,4) = predicted_probsPG(:,2);
Hm(:,4) = predicted_probsH(:,2);
BSm(:,4) = predicted_probsBS(:,2);

% If we use AdaSyn+Undersampling then, we construct probability estimates
% with balanced validation set. Use these estimates for training the meta
% classifier on balanced validation set.
if useAdaSyn == 1
    
    [~,predicted_probsPCC] = predict(trainedlinSVMModelPCC, [Xada;Xvalid]);
    [~,predicted_probsKS] = predict(trainedlinSVMModelKS, [Xada;Xvalid]);
    [~,predicted_probsAUC] = predict(trainedlinSVMModelAUC, [Xada;Xvalid]);
    [~,predicted_probsPG] = predict(trainedlinSVMModelPG, [Xada;Xvalid]);
    [~,predicted_probsH] = predict(trainedlinSVMModelH, [Xada;Xvalid]); 
    [~,predicted_probsBS] = predict(trainedlinSVMModelBS, [Xada;Xvalid]);
   
    PCCmb(:,4) = predicted_probsPCC(:,2);
    KSmb(:,4) = predicted_probsKS(:,2);
    AUCmb(:,4) = predicted_probsAUC(:,2);
    PGmb(:,4) = predicted_probsPG(:,2);
    Hmb(:,4) = predicted_probsH(:,2);
    BSmb(:,4) = predicted_probsBS(:,2);

end


%% SVM with Rbf kernel #5

% Now fit SVMs with rbf kernel with the optimal hyperparameters for each performance
% measures and compute the performance values on validation data.
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

[~,predicted_probsPCC] = predict(trainedSVMModelPCC, Xvalid);
[~,predicted_probsKS] = predict(trainedSVMModelKS, Xvalid);
[~,predicted_probsAUC] = predict(trainedSVMModelAUC, Xvalid);
[~,predicted_probsPG] = predict(trainedSVMModelPG, Xvalid);
[~,predicted_probsH] = predict(trainedSVMModelH, Xvalid); 
[~,predicted_probsBS] = predict(trainedSVMModelBS, Xvalid); 

sortedProbs = sort(predicted_probsPCC(:,2),'descend'); %sort probabilities
t = sortedProbs(round(mean(y1)*size(predicted_probsPCC,1)));

YhatPCC = predicted_probsPCC(:,2) > t;

% function that computes the PCC, requires real y-values, predicted_y values.
beta_PCC(5) =  sum( (Yvalid == YhatPCC) )/numel(Yvalid);

prior1 = mean(y1); prior0 = 1 - prior1;

[beta_AUC(5),~, ~ ] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsAUC(:,2), prior1, prior0);

[~,beta_PG(5), ~ ] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsPG(:,2), prior1, prior0);

[~,~, beta_H(5)] = computeAUC_PGindex_Hvalue(Yvalid, predicted_probsH(:,2), prior1, prior0);

beta_BS(5) = mean( (predicted_probsBS(:,2) - Yvalid).^2);

beta_KS(5) = computeKSvalue(Yvalid,predicted_probsKS(:,2));

PCCm(:,5) = predicted_probsPCC(:,2);
KSm(:,5) = predicted_probsKS(:,2);
AUCm(:,5) = predicted_probsAUC(:,2);
PGm(:,5) = predicted_probsPG(:,2);
Hm(:,5) = predicted_probsH(:,2);
BSm(:,5) = predicted_probsBS(:,2);

% If we use AdaSyn+Undersampling then, we construct probability estimates
% with balanced validation set. Use these estimates for training the meta
% classifier on balanced validation set.
if useAdaSyn == 1
    
    [~,predicted_probsPCC] = predict(trainedSVMModelPCC, [Xada;Xvalid]);
    [~,predicted_probsKS] = predict(trainedSVMModelKS, [Xada;Xvalid]);
    [~,predicted_probsAUC] = predict(trainedSVMModelAUC, [Xada;Xvalid]);
    [~,predicted_probsPG] = predict(trainedSVMModelPG, [Xada;Xvalid]);
    [~,predicted_probsH] = predict(trainedSVMModelH, [Xada;Xvalid]); 
    [~,predicted_probsBS] = predict(trainedSVMModelBS, [Xada;Xvalid]);
   
    PCCmb(:,5) = predicted_probsPCC(:,2);
    KSmb(:,5) = predicted_probsKS(:,2);
    AUCmb(:,5) = predicted_probsAUC(:,2);
    PGmb(:,5) = predicted_probsPG(:,2);
    Hmb(:,5) = predicted_probsH(:,2);
    BSmb(:,5) = predicted_probsBS(:,2);

end

%% Compute the normalized weights for the avgW ensemble.

beta_PCC = beta_PCC/sum(beta_PCC);
beta_AUC = beta_AUC/sum(beta_AUC);
beta_PG = beta_PG/sum(beta_PG);
beta_BS = beta_BS/sum(beta_BS);
beta_KS = beta_KS/sum(beta_KS);
beta_H = beta_H/sum(beta_H);

%% make PD predictions on validation set 2 with LR

predicted_probs2 =  ( 1./(  1  + exp( -w'*Xvalid2' - b    ) ) )';

PCCm2(:,1) = predicted_probs2;
KSm2(:,1) = predicted_probs2;
AUCm2(:,1) = predicted_probs2;
PGm2(:,1) = predicted_probs2;
Hm2(:,1) = predicted_probs2;
BSm2(:,1) = predicted_probs2;
    
%% make PD predictions on validation set 2 with DTs

[~, predicted_probsPCC,~,~] = predict(TreePCC,Xvalid2);
predicted_probsPCC = predicted_probsPCC(:,2);

[~, predicted_probsKS,~,~] = predict(TreeKS,Xvalid2);
predicted_probsKS = predicted_probsKS(:,2);

[~, predicted_probsH,~,~] = predict(TreeH,Xvalid2);
predicted_probsH = predicted_probsH(:,2);

[~, predicted_probsPG,~,~] = predict(TreePG,Xvalid2);
predicted_probsPG = predicted_probsPG(:,2);

[~, predicted_probsAUC,~,~] = predict(TreeAUC,Xvalid2);
predicted_probsAUC = predicted_probsAUC(:,2);

[~, predicted_probsBS,~,~] = predict(TreeBS,Xvalid2);
predicted_probsBS = predicted_probsBS(:,2);

PCCm2(:,2) = predicted_probsPCC;
KSm2(:,2) = predicted_probsKS;
Hm2(:,2) = predicted_probsH;
PGm2(:,2) = predicted_probsPG;
AUCm2(:,2) = predicted_probsAUC;
BSm2(:,2) = predicted_probsBS;

%% make PD predictions on validation set 2 with FNN

predicted_probsPCC = netPCC(Xvalid2');
predicted_probsPCC = predicted_probsPCC(1,:)';

predicted_probsKS = netKS(Xvalid2');
predicted_probsKS = predicted_probsKS(1,:)';

predicted_probsH = netH(Xvalid2');
predicted_probsH = predicted_probsH(1,:)';

predicted_probsPG = netPG(Xvalid2');
predicted_probsPG = predicted_probsPG(1,:)';

predicted_probsAUC = netAUC(Xvalid2');
predicted_probsAUC = predicted_probsAUC(1,:)';

predicted_probsBS = netBS(Xvalid2');
predicted_probsBS = predicted_probsBS(1,:)';

PCCm2(:,3) = predicted_probsPCC;
KSm2(:,3) = predicted_probsKS;
Hm2(:,3) = predicted_probsH;
PGm2(:,3) = predicted_probsPG;
AUCm2(:,3) = predicted_probsAUC;
BSm2(:,3) = predicted_probsBS;

%% make PD predictions on validation set 2 with linear SVM

[~,predicted_probsPCC] = predict(trainedlinSVMModelPCC, Xvalid2);
[~,predicted_probsKS] = predict(trainedlinSVMModelKS, Xvalid2);
[~,predicted_probsAUC] = predict(trainedlinSVMModelAUC, Xvalid2);
[~,predicted_probsPG] = predict(trainedlinSVMModelPG, Xvalid2);
[~,predicted_probsH] = predict(trainedlinSVMModelH, Xvalid2); 
[~,predicted_probsBS] = predict(trainedlinSVMModelBS, Xvalid2); 

PCCm2(:,4) = predicted_probsPCC(:,2);
KSm2(:,4) = predicted_probsKS(:,2);
AUCm2(:,4) = predicted_probsAUC(:,2);
PGm2(:,4) = predicted_probsPG(:,2);
Hm2(:,4) = predicted_probsH(:,2);
BSm2(:,4) = predicted_probsBS(:,2);

%% make PD predictions on validation set 2 with SVM with Rbf kernel

[~,predicted_probsPCC] = predict(trainedSVMModelPCC, Xvalid2);
[~,predicted_probsKS] = predict(trainedSVMModelKS, Xvalid2);
[~,predicted_probsAUC] = predict(trainedSVMModelAUC, Xvalid2);
[~,predicted_probsPG] = predict(trainedSVMModelPG, Xvalid2);
[~,predicted_probsH] = predict(trainedSVMModelH, Xvalid2); 
[~,predicted_probsBS] = predict(trainedSVMModelBS, Xvalid2); 

PCCm2(:,5) = predicted_probsPCC(:,2);
KSm2(:,5) = predicted_probsKS(:,2);
AUCm2(:,5) = predicted_probsAUC(:,2);
PGm2(:,5) = predicted_probsPG(:,2);
Hm2(:,5) = predicted_probsH(:,2);
BSm2(:,5) = predicted_probsBS(:,2);
    
%% Determine optimal regularization parameter in LR-R:

% the rows are the 5 IV variables weights and the columns are the performance
% measures: PCC, AUC, PG, BS, KS, H.
optim_w_LRR = zeros(NumBaseLearners, 6);
optim_b_LRR = zeros(1,6);

% default is 100, but that takes a bit longer. Here eta is the
% LASSO regularization parameter of the regularized logistic regression
% model. The range of eta values differs per evaluation metric and is
% determined with a geometric sequence such that the last eta value
% generates zero coefficients.
numEtas = 50;

if useAdaSyn == 1
  Yvalid = [yada;Yvalid];   
  PCCm = PCCmb;
  AUCm = AUCmb;
  PGm = PGmb;
  BSm = BSmb;
  KSm = KSmb;
  Hm = Hmb;  
end

% Train the meta classifier on the validation set 1 with prob. prediction
% of the base learners.
[B_PCC,FitInfoPCC] = lassoglm(PCCm,Yvalid,'binomial', 'NumLambda', numEtas);
[B_AUC,FitInfoAUC] = lassoglm(AUCm,Yvalid,'binomial', 'NumLambda', numEtas);
[B_PG,FitInfoPG] = lassoglm(PGm,Yvalid,'binomial', 'NumLambda', numEtas);
[B_BS,FitInfoBS] = lassoglm(BSm,Yvalid,'binomial', 'NumLambda', numEtas);
[B_KS,FitInfoKS] = lassoglm(KSm,Yvalid,'binomial', 'NumLambda', numEtas);
[B_H,FitInfoH] = lassoglm(Hm,Yvalid,'binomial', 'NumLambda', numEtas);

% PCC, KS, AUC, H, PG, BS (in this order).
PerformanceMeasuresMatrix =  zeros(numEtas,6);

% Make predictions on the second validation for determining the optimal
% model/hyper parameters of the regularized LR meta classifier.
for eta = 1:numEtas
    eta
    
    tempPCC =  ( 1./(  1  + exp( -B_PCC(:,eta)'*PCCm2' - FitInfoPCC.Intercept(eta)  ) ) )';
    tempAUC =  ( 1./(  1  + exp( -B_AUC(:,eta)'*AUCm2' - FitInfoAUC.Intercept(eta)  ) ) )';
    tempPG =  ( 1./(  1  + exp( -B_PG(:,eta)'*PGm2' - FitInfoPG.Intercept(eta)  ) ) )';
    tempBS =  ( 1./(  1  + exp( -B_BS(:,eta)'*BSm2' - FitInfoBS.Intercept(eta)  ) ) )';
    tempKS =  ( 1./(  1  + exp( -B_KS(:,eta)'*KSm2' - FitInfoKS.Intercept(eta)  ) ) )';
    tempH =  ( 1./(  1  + exp( -B_H(:,eta)'*Hm2' - FitInfoH.Intercept(eta)  ) ) )';

    sortedProbs = sort(tempPCC,'descend'); %sort probabilities
    t = sortedProbs(round(mean(y1)*size(sortedProbs,1)));
    YhatPCC = tempPCC > t;
   
    % function that computes the PCC, requires true y-values and predicted y-values.
    PCC =  sum( (Yvalid2 == YhatPCC) )/numel(Yvalid2);
    PerformanceMeasuresMatrix(eta,1) = PCC;

    prior1 = mean(y1); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Yvalid2, tempAUC, prior1, prior0);
    PerformanceMeasuresMatrix(eta,3) = AUC;


    [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Yvalid2, tempPG, prior1, prior0);
    PerformanceMeasuresMatrix(eta,5) = PG_index;


    [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Yvalid2, tempH, prior1, prior0);
    PerformanceMeasuresMatrix(eta,4) = H_measure ;


    BScore = mean( (tempBS - Yvalid2).^2);
    PerformanceMeasuresMatrix(eta,6) = BScore;

    KS_value = computeKSvalue(Yvalid2,tempKS);
    PerformanceMeasuresMatrix(eta,2) = KS_value ;

end

% extract the indices of the corresponding optimal eta regularization parameter for each
% measure:
[~, ind] = max(PerformanceMeasuresMatrix(:,1)); % PCC
optim_w_LRR(:,1) =  B_PCC(:,ind);
optim_b_LRR(1,1) =  FitInfoPCC.Intercept(ind);

[~, ind] = max(PerformanceMeasuresMatrix(:,2)); % KS
optim_w_LRR(:,5) =  B_KS(:,ind);
optim_b_LRR(1,5) =  FitInfoKS.Intercept(ind);

[~, ind] = max(PerformanceMeasuresMatrix(:,3)); % AUC
optim_w_LRR(:,2) =  B_AUC(:,ind);
optim_b_LRR(1,2) =  FitInfoAUC.Intercept(ind);

[~, ind] = max(PerformanceMeasuresMatrix(:,4)); % Hmeasure
optim_w_LRR(:,6) =  B_H(:,ind);
optim_b_LRR(1,6) =  FitInfoH.Intercept(ind);

[~, ind] = max(PerformanceMeasuresMatrix(:,5)); % PG_index
optim_w_LRR(:,3) =  B_PG(:,ind);
optim_b_LRR(1,3) =  FitInfoPG.Intercept(ind);

[~, ind] = min(PerformanceMeasuresMatrix(:,6)); % BS
optim_w_LRR(:,4) =  B_BS(:,ind);
optim_b_LRR(1,4) =  FitInfoBS.Intercept(ind);


end