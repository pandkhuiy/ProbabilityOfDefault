function [PCC, AUC, PG,BS, KS, H] = rbfSVM(X,y)
%% This code uses support vector machines with polynomial kernel functions to make classifications for the test sample
% where we make use of Px2 cross validation, that is, we do P times 2 fold
% cross validation for the total data set X and y.

P = 3;

seed = 1;
rng('default');
rng(seed);

% This vector is (P*2 by 1) and represents the PCC values for each cv
% iteration. Same for the other five performance measures.
PCC_vector = zeros(P*2,1);
AUC_vector = zeros(P*2,1);
PGini_vector = zeros(P*2,1);
BScore_vector = zeros(P*2,1);
Hmeasure_vector = zeros(P*2,1);
KSvalue_vector = zeros(P*2,1);


 for  l = 1:P
  l
     
  k = randperm(size(X,1));
 
  % X1 X2 Y1 and Y2 are training and test sets that can interchanged for
  % our 2 fold cross validation for each outerloop of index l.
  X1 = X(k(1:size(X,1)/2),:);
  X2 = X(k( ((size(X,1)/2)+1): end), :);
 
  Y1 = y(k(1:size(X,1)/2));
  Y2 = y(k(((size(X,1)/2)+1): end));

  
  % Start with X1 and Y1 as training sets, and X2 and Y2 as test sets.
  % initialization
  %[N,numW] = size(X1);
  
  % Determine the optimal penalty constant C and optimal rbd kernel scale parameter s, with k fold cross validation
  % with k = 5 on the training data.
  [C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS, ...
      S_optimal_PCC, S_optimal_AUC, S_optimal_PG, S_optimal_BS, S_optimal_H, S_optimal_KS] =  optimizeRbfSVM(X1, Y1);
  
  % Now fit a SVM with the optimal hyperparameter for each performance
  % measure
  trainedSVMModelPCC = fitcsvm(X1,Y1,'BoxConstraint',C_optimal_PCC,'KernelFunction','rbf','KernelScale',S_optimal_PCC,'ClassNames',[0,1]);
  trainedSVMModelKS = fitcsvm(X1,Y1,'BoxConstraint',C_optimal_KS,'KernelFunction','rbf','KernelScale',S_optimal_KS,'ClassNames',[0,1]);
  trainedSVMModelAUC = fitcsvm(X1,Y1,'BoxConstraint',C_optimal_AUC,'KernelFunction','rbf','KernelScale',S_optimal_AUC,'ClassNames',[0,1]);
  trainedSVMModelPG = fitcsvm(X1,Y1,'BoxConstraint',C_optimal_PG,'KernelFunction','rbf','KernelScale',S_optimal_H,'ClassNames',[0,1]);
  trainedSVMModelH = fitcsvm(X1,Y1,'BoxConstraint',C_optimal_H,'KernelFunction','rbf','KernelScale',S_optimal_PG,'ClassNames',[0,1]);
  trainedSVMModelBS = fitcsvm(X1,Y1,'BoxConstraint',C_optimal_BS,'KernelFunction','rbf','KernelScale',S_optimal_BS,'ClassNames',[0,1]);
  
  trainedSVMModelPCC = fitPosterior(trainedSVMModelPCC, X1, Y1);
  trainedSVMModelKS = fitPosterior(trainedSVMModelKS, X1, Y1);
  trainedSVMModelAUC = fitPosterior(trainedSVMModelAUC, X1, Y1);
  trainedSVMModelPG = fitPosterior(trainedSVMModelPG, X1, Y1);
  trainedSVMModelH = fitPosterior(trainedSVMModelH, X1, Y1);
  trainedSVMModelBS = fitPosterior(trainedSVMModelBS, X1, Y1);
  
  [YhatPCC,~] = predict(trainedSVMModelPCC, X2);
  [~,predicted_probsKS] = predict(trainedSVMModelKS, X2);
  [~,predicted_probsAUC] = predict(trainedSVMModelAUC, X2);
  [~,predicted_probsPG] = predict(trainedSVMModelPG, X2);
  [~,predicted_probsH] = predict(trainedSVMModelH, X2); 
  [~,predicted_probsBS] = predict(trainedSVMModelBS, X2); 
  
  
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Y2 == YhatPCC) )/numel(Y2);
  PCC_vector(l) = PCC;
  
  prior1 = mean(Y1); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Y2, predicted_probsAUC(:,2), prior1, prior0);
  AUC_vector(l) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Y2, predicted_probsPG(:,2), prior1, prior0);
  PGini_vector(l) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Y2, predicted_probsH(:,2), prior1, prior0);
  Hmeasure_vector(l) = H_measure;
  
  
  BScore = mean( (predicted_probsBS(:,2) - Y2).^2);
  BScore_vector(l) = BScore;
 
  KS_value = computeKSvalue(Y2,predicted_probsKS(:,2));
  KSvalue_vector(l) = KS_value;
  
  %%
  % Reverse the roles: use X2 and Y2 as training sets, and X1 and Y1 as test sets.  
  % Start with X2 and Y2 as training sets, and X1 and Y1 as test sets.
  
  % Determine the optimal penalty constant C and optimal rbd kernel scale parameter s, with k fold cross validation
  % with k = 5 on the training data.
  [C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS, ...
      S_optimal_PCC, S_optimal_AUC, S_optimal_PG, S_optimal_BS, S_optimal_H, S_optimal_KS] =  optimizeRbfSVM(X2, Y2);
  
  % Now fit a SVM with the optimal hyperparameter for each performance
  % measure
  trainedSVMModelPCC = fitcsvm(X2,Y2,'BoxConstraint',C_optimal_PCC,'KernelFunction','rbf','KernelScale',S_optimal_PCC,'ClassNames',[0,1]);
  trainedSVMModelKS = fitcsvm(X2,Y2,'BoxConstraint',C_optimal_KS,'KernelFunction','rbf','KernelScale',S_optimal_KS,'ClassNames',[0,1]);
  trainedSVMModelAUC = fitcsvm(X2,Y2,'BoxConstraint',C_optimal_AUC,'KernelFunction','rbf','KernelScale',S_optimal_AUC,'ClassNames',[0,1]);
  trainedSVMModelPG = fitcsvm(X2,Y2,'BoxConstraint',C_optimal_PG,'KernelFunction','rbf','KernelScale',S_optimal_H,'ClassNames',[0,1]);
  trainedSVMModelH = fitcsvm(X2,Y2,'BoxConstraint',C_optimal_H,'KernelFunction','rbf','KernelScale',S_optimal_PG,'ClassNames',[0,1]);
  trainedSVMModelBS = fitcsvm(X2,Y2,'BoxConstraint',C_optimal_BS,'KernelFunction','rbf','KernelScale',S_optimal_BS,'ClassNames',[0,1]);
  
  trainedSVMModelPCC = fitPosterior(trainedSVMModelPCC, X2, Y2);
  trainedSVMModelKS = fitPosterior(trainedSVMModelKS, X2, Y2);
  trainedSVMModelAUC = fitPosterior(trainedSVMModelAUC, X2, Y2);
  trainedSVMModelPG = fitPosterior(trainedSVMModelPG, X2, Y2);
  trainedSVMModelH = fitPosterior(trainedSVMModelH, X2, Y2);
  trainedSVMModelBS = fitPosterior(trainedSVMModelBS, X2, Y2);
  
  [YhatPCC,~] = predict(trainedSVMModelPCC, X1);
  [~,predicted_probsKS] = predict(trainedSVMModelKS, X1);
  [~,predicted_probsAUC] = predict(trainedSVMModelAUC, X1);
  [~,predicted_probsPG] = predict(trainedSVMModelPG, X1);
  [~,predicted_probsH] = predict(trainedSVMModelH, X1); 
  [~,predicted_probsBS] = predict(trainedSVMModelBS, X1); 
  
  
  % We computes the PCC, and require real y-values and predicted_y
  % values.
  PCC =  sum( (Y1 == YhatPCC) )/numel(Y1);
  PCC_vector(l+3) = PCC;
  
  prior1 = mean(Y2); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Y1, predicted_probsAUC(:,2), prior1, prior0);
  AUC_vector(l+3) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Y1, predicted_probsPG(:,2), prior1, prior0);
  PGini_vector(l+3) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Y1, predicted_probsH(:,2), prior1, prior0);
  Hmeasure_vector(l+3) = H_measure;
  
  
  BScore = mean( (predicted_probsBS(:,2) - Y1).^2);
  BScore_vector(l+3) = BScore;
 
  KS_value = computeKSvalue(Y1,predicted_probsKS(:,2));
  KSvalue_vector(l+3) = KS_value;
  
 end
 
PCC = mean(PCC_vector);
AUC = mean(AUC_vector);
PG  = mean(PGini_vector);
BS  = mean(BScore_vector);
H   = mean(Hmeasure_vector);
KS  = mean(KSvalue_vector);

end