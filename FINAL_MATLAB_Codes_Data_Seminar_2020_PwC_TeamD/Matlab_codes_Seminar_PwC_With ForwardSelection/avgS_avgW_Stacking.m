function [PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS,...
          PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW,...
          PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking] = avgS_avgW_Stacking( X1,X2,y1,y2,...
                                                                                            Optimal_HP_FNN, Optimal_HP_DT, Optimal_HP_RbfSVM, Optimal_HP_LinSVM )
%% This code implements three heterogeneous ensemble methods to make classifications on the final test sample.
% avgS:     Simply average the PD estimates of all the optimally trained base learners
%           to get the average PD prediction.
% avgW:     use a weighted average of the PD estimates obtained from all
%           optimally trained base learners, where the weights are based on the
%           relative performances of the base learners in terms of one of six
%           performance measures, for example: we use PCC as the performance of
%           the base learners to compute the weight for base learner i as
%           w_i = PCC_i / sum(PCC_i), where PCC_i is computed with the
%           generated validation data. This PCC criterion is only used for
%           comparison with the final PCC criterion on the test data. This is
%           done similarly for every other criteria: AUC, PG, BS, KS , H.
% Stacking: Use a meta-learner (here it is the regularized LR classifier)
%           to use the PD estimates from the optimally trained base
%           learners to train the meta-learner on the validation data set.
%           Use this validation set to find the optimal regularization
%           parameter (of the LR-R model). Finally use the trained
%           meta-learner to make predictions on the test data with the
%           trained base learners.
% The following five base learners are used: LR, FNN, SVM-L, SVM-Rbf, DT.

% extract the optimal  Hyperparameters for FNN given as input:
lambda_and_hidNodes_optimal_PCC = Optimal_HP_FNN(1,:); 
lambda_and_hidNodes_optimal_AUC = Optimal_HP_FNN(2,:);
lambda_and_hidNodes_optimal_PG  = Optimal_HP_FNN(3,:); 
lambda_and_hidNodes_optimal_BS  = Optimal_HP_FNN(4,:); 
lambda_and_hidNodes_optimal_KS  = Optimal_HP_FNN(5,:); 
lambda_and_hidNodes_optimal_H   = Optimal_HP_FNN(6,:); 

% extract the optimal Hyperparameters for Decision trees given as input:
Prune_and_MinLeafSize_optimal_PCC = Optimal_HP_DT(1,:); 
Prune_and_MinLeafSize_optimal_AUC = Optimal_HP_DT(2,:);
Prune_and_MinLeafSize_optimal_PG = Optimal_HP_DT(3,:); 
Prune_and_MinLeafSize_optimal_BS = Optimal_HP_DT(4,:); 
Prune_and_MinLeafSize_optimal_KS = Optimal_HP_DT(5,:); 
Prune_and_MinLeafSize_optimal_H = Optimal_HP_DT(6,:); 

% extract the optimal Hyperparameters for SVM with Rbf kernel given as input:
C_optimal_PCC = Optimal_HP_RbfSVM(1,1); 
S_optimal_PCC = Optimal_HP_RbfSVM(1,2);

C_optimal_AUC = Optimal_HP_RbfSVM(2,1); 
S_optimal_AUC = Optimal_HP_RbfSVM(2,2);

C_optimal_PG = Optimal_HP_RbfSVM(3,1); 
S_optimal_PG = Optimal_HP_RbfSVM(3,2);

C_optimal_BS = Optimal_HP_RbfSVM(4,1); 
S_optimal_BS = Optimal_HP_RbfSVM(4,2);

C_optimal_KS = Optimal_HP_RbfSVM(5,1); 
S_optimal_KS = Optimal_HP_RbfSVM(5,2);

C_optimal_H = Optimal_HP_RbfSVM(6,1); 
S_optimal_H = Optimal_HP_RbfSVM(6,2);

% extract the optimal Hyperparameters for Linear SVM given as input:
C_lin_optimal_PCC = Optimal_HP_LinSVM(1,:);
C_lin_optimal_AUC = Optimal_HP_LinSVM(2,:);
C_lin_optimal_PG = Optimal_HP_LinSVM(3,:);
C_lin_optimal_BS = Optimal_HP_LinSVM(4,:);
C_lin_optimal_KS = Optimal_HP_LinSVM(5,:);
C_lin_optimal_H = Optimal_HP_LinSVM(6,:);

% If we want to use the combined resampling (undersampling+AdaSyn) we set
% useAdaSyn to 1, else 0.
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

% Generate first two validation sets Xvalid,Yvalid, Xvalid2,Yvalid2 from the first quarter data
% set. Use the remaining observation for generating the training sets
% Xtrain and ytrain dependent on whether we use AdaSyn + undersampling or
% not.
if useAdaSyn == 1
    number = 0;
    number2 = 0;
    while number <5 || number2 <5
        [temp, ind] = datasample([X1 y1],10000, 'Replace',false);
        Xvalid = temp(:,1:end-1); 
        Yvalid = temp(:,end);
        
        missIndex  = setdiff(1:size(X1,1),ind);
        X1remaining = X1(missIndex,:);
        y1remaining = y1(missIndex);
        
        [temp, ind] = datasample([X1remaining y1remaining],10000, 'Replace',false);
        Xvalid2 = temp(:,1:end-1); 
        Yvalid2 = temp(:,end);
        
        missIndex  = setdiff(1:size(X1remaining,1),ind);
        X1remaining = X1remaining(missIndex,:);
        y1remaining = y1remaining(missIndex);
        
        % Use all the minority class examples for the training set
        % (undersampling) and if #minority examples is < 5000, fill the
        % remaining training sets with ADASYNed minority class
        % observations such that final Xtrain and ytrain has APPROXIMATELY 50% has 1s and 50% 0s
        % Xtrain and ytrain have 10000 samples.
        [Xtrain,ytrain] =   BalancedData(X1remaining,y1remaining); 
        
        number = sum(Yvalid);
        number2 = sum(Yvalid2);
    end
else
    number1 = 0;
    number2 = 0;
    number3 = 0;
    while number1 <5 || number2 <5 || number3 <5
        [temp, ind] = datasample([X1 y1],10000, 'Replace',false);
        Xvalid = temp(:,1:end-1); 
        Yvalid = temp(:,end);
        
        missIndex  = setdiff(1:size(X1,1),ind);
        X1remaining = X1(missIndex,:);
        y1remaining = y1(missIndex);
        
        [temp, ind] = datasample([X1remaining y1remaining],10000, 'Replace',false);
        Xvalid2 = temp(:,1:end-1); 
        Yvalid2 = temp(:,end);
        
        missIndex  = setdiff(1:size(X1remaining,1),ind);
        X1remaining = X1remaining(missIndex,:);
        y1remaining = y1remaining(missIndex);
        
        % Use the remaining observations of the original training data X1 and y1
        % to generate 10000 random examples: Xtrain and ytrain
        temp = datasample([X1remaining y1remaining],10000, 'Replace',false);
        Xtrain = temp(:,1:end-1); 
        ytrain = temp(:,end);
             
        number1 = sum(ytrain);
        number2 = sum(Yvalid);
        number3 = sum(Yvalid2);
    end
end
  
% Use the validation data set to compute the performance measure based
% weights beta_i for each optimally trained base Learner i, in the avgW ensemble.
% For the BS measure, we use the reciproque (i.e., 1/BS) as a lower BS
% value means more weight is given to the base learner.
% The outputs are 5x1 vectors for each performance measures.
[beta_PCC, beta_AUC, beta_PG, beta_BS, beta_KS, beta_H, optim_w_LRR, optim_b_LRR ]  = computeEnsembleWeights( Xtrain,ytrain,Xvalid, Yvalid,Xvalid2,Yvalid2,y1,...
                                Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG, Prune_and_MinLeafSize_optimal_BS,...
                                Prune_and_MinLeafSize_optimal_H, Prune_and_MinLeafSize_optimal_KS,...
                                lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
                                lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS,...
                                C_lin_optimal_PCC, C_lin_optimal_AUC, C_lin_optimal_PG, C_lin_optimal_BS, C_lin_optimal_H, C_lin_optimal_KS,...
                                C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS,...
                                S_optimal_PCC, S_optimal_AUC, S_optimal_PG, S_optimal_BS, S_optimal_H, S_optimal_KS , useAdaSyn );

% Use the previously obtained optimal model and Hyperparameters of the avgS, avgW and Stacking method
%   to compute the six performance measures for each of the three methods.
% The avgS averages the PD estimates of the different base
%   learners and use the final PD estimates for computing the performance
%   measures.
% The avgW uses the weighted average of the PD estimates of the base
%   learners for final performance values.
% Stacking uses the regularized LR method as meta classifier to aggregate the base learners
%   predictions on the test set to obtain final PD estimates for performance
%   values.
[PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS,...
 PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW,...
 PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking]    = compute_AvgS_AvgW_Stacking( Xtrain,ytrain,Xtest,Ytest,y1,...
                                Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG, Prune_and_MinLeafSize_optimal_BS,...
                                Prune_and_MinLeafSize_optimal_H, Prune_and_MinLeafSize_optimal_KS,...
                                lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
                                lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS,...
                                C_lin_optimal_PCC, C_lin_optimal_AUC, C_lin_optimal_PG, C_lin_optimal_BS, C_lin_optimal_H, C_lin_optimal_KS,...
                                C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS,...
                                S_optimal_PCC, S_optimal_AUC, S_optimal_PG, S_optimal_BS, S_optimal_H, S_optimal_KS,...
                                beta_PCC, beta_AUC, beta_PG, beta_BS, beta_KS, beta_H, optim_w_LRR, optim_b_LRR );

end
