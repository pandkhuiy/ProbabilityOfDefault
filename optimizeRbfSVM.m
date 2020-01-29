function [C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS,...
    S_optimal_PCC, S_optimal_AUC, S_optimal_PG, S_optimal_BS, S_optimal_H, S_optimal_KS] =  optimizeRbfSVM(Xtrain, Ytrain)
%% Perform k fold cross validation to determine the optimal penalty parameter C and optimal rbf kernel scale s for the rbf SVM method
% in order to get probability score, we use Platt's method to callibrate
% the classifications in the test set to get probability scores through the
% use of the in-built MATLAB function fitPosterior(SVMModel, X,Y) where
% SVMModel is the trained SVM model and X and Y are the training data.

% [N,numVariables] = size(X);

k = 5;

C = [0.00001;0.0001;0.001;0.01;0.1;1;10;100];
S = [0.0001;0.001;0.01;0.1;10;100;1000]; 

% This kx6 matrix contains the average performance values computed after k
% fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
PerformanceMeasuresMatrixPCC =  zeros(numel(C),numel(S));
PerformanceMeasuresMatrixKS =  zeros(numel(C),numel(S));
PerformanceMeasuresMatrixAUC =  zeros(numel(C),numel(S));
PerformanceMeasuresMatrixH =  zeros(numel(C),numel(S));
PerformanceMeasuresMatrixPG =  zeros(numel(C),numel(S));
PerformanceMeasuresMatrixBS =  zeros(numel(C),numel(S));


for  c = 1:numel(C)
 c 
for  s = 1:numel(S)
 s
 
[indices] = DoKfoldCrossValid(Xtrain,k);
TrainingIndices = logical(indices);  
TestIndices = logical(1-indices);  

% This kx6 matrix contains the average performance values computed after k
% fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
kfoldPerformanceMeasuresMatrix =  zeros(k,6);

for j = 1:k
    j
  % train the linear SVM model and use them for predicting whether the loan corresponding
  % to the features will be granted credit or reject (will default or not).
  % Use these classifcations and predicted probs for computing the six performance measures. 
  trainedSVMModel = fitcsvm(Xtrain(TrainingIndices(:,j),:),Ytrain(TrainingIndices(:,j)),'BoxConstraint',C(c),'KernelFunction','rbf','KernelScale',S(s),'ClassNames',[0,1]);
  trainedSVMModel = fitPosterior(trainedSVMModel, Xtrain(TrainingIndices(:,j),:), Ytrain(TrainingIndices(:,j)) );
  
  [Yhat,predicted_probs] = predict(trainedSVMModel, Xtrain(TestIndices(:,j),:)    );
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Ytrain(TestIndices(:,j))  == Yhat) )/numel(Ytrain(TestIndices(:,j)));
  kfoldPerformanceMeasuresMatrix(j,1) = PCC;
  
  prior1 = mean(Ytrain(TrainingIndices(:,j))); 
  prior0 = 1 - prior1;
  
  [AUC,PG_index, H_measure ] = computeAUC_PGindex_Hvalue(Ytrain(TestIndices(:,j)), predicted_probs(:,2), prior1, prior0);
  
  kfoldPerformanceMeasuresMatrix(j,3) = AUC;
  kfoldPerformanceMeasuresMatrix(j,4) = H_measure;
  kfoldPerformanceMeasuresMatrix(j,5) = PG_index;
  
  [KS_value] = computeKSvalue(Ytrain(TestIndices(:,j)),predicted_probs(:,2));
  
  kfoldPerformanceMeasuresMatrix(j,2) = KS_value;
  
  BScore = mean( (predicted_probs(:,2) - Ytrain(TestIndices(:,j)) ).^2);
  kfoldPerformanceMeasuresMatrix(j,6) = BScore;
  
end

MeanVector = mean(kfoldPerformanceMeasuresMatrix);

PerformanceMeasuresMatrixPCC(c,s) = MeanVector(1);
PerformanceMeasuresMatrixKS(c,s) = MeanVector(2);
PerformanceMeasuresMatrixAUC(c,s) = MeanVector(3);
PerformanceMeasuresMatrixH(c,s) = MeanVector(4);
PerformanceMeasuresMatrixPG(c,s) = MeanVector(5);
PerformanceMeasuresMatrixBS(c,s) = MeanVector(6);

end
end

% extract the indices of the corresponding optimal parameter C for each
% measure:
%PCC
[MaxRow, ind] = max(PerformanceMeasuresMatrixPCC); 
[~, s_index] = max(MaxRow);
c_index = ind(s_index);
C_optimal_PCC =  C(c_index);
S_optimal_PCC =  S(s_index);

%KS
[MaxRow, ind] = max(PerformanceMeasuresMatrixKS); 
[~, s_index] = max(MaxRow);
c_index = ind(s_index);
C_optimal_KS =  C(c_index);
S_optimal_KS =  S(s_index);

%AUC
[MaxRow, ind] = max(PerformanceMeasuresMatrixAUC); 
[~, s_index] = max(MaxRow);
c_index = ind(s_index);
C_optimal_AUC =  C(c_index);
S_optimal_AUC =  S(s_index);

%H measure
[MaxRow, ind] = max(PerformanceMeasuresMatrixH); 
[~, s_index] = max(MaxRow);
c_index = ind(s_index);
C_optimal_H =  C(c_index);
S_optimal_H =  S(s_index);

%PG
[MaxRow, ind] = max(PerformanceMeasuresMatrixPG); 
[~, s_index] = max(MaxRow);
c_index = ind(s_index);
C_optimal_PG =  C(c_index);
S_optimal_PG =  S(s_index);

%BS
[MaxRow, ind] = max(PerformanceMeasuresMatrixBS); 
[~, s_index] = max(MaxRow);
c_index = ind(s_index);
C_optimal_BS =  C(c_index);
S_optimal_BS =  S(s_index);

end