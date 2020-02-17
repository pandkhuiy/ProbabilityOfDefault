function [C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS] =  optimizeLinearSVM(Xtrain, Ytrain, useAdaSyn)
%% Perform k fold cross validation to determine the optimal penalty parameter C for the linear SVM method
% in order to get probability score, we use Platt's method to callibrate
% the classifications in the test set to get probability scores through the
% use of the in-built MATLAB function fitPosterior(SVMModel, X,Y) where
% SVMModel is the trained SVM model and X and Y are the training data.

% [N,numVariables] = size(X);

k = 5;

C = [0.0001;0.0005;0.001;0.005;0.01;0.05;0.1;1];

% This kx6 matrix contains the average performance values computed after k
% fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
PerformanceMeasuresMatrix =  zeros(numel(C),6);

for  c = 1:numel(C)
 c 
   [TrainingIndices, TestIndices] = DoKfoldCrossValid(Ytrain,k);
%TrainingIndices = logical(indices);  
%TestIndices = logical(1-indices);  

% This kx6 matrix contains the average performance values computed after k
% fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
kfoldPerformanceMeasuresMatrix =  zeros(k,6);

for j = 1:k
    j
  % train the linear SVM model and use them for predicting whether the loan corresponding
  % to the features will be granted credit or reject (will default or not).
  % Use these classifcations and predicted probs for computing the six performance measures. 
  % if useAdaSyn == 1,  we use AdaSyn sampling method to create synthetic balanced training
  % data set, else we just use the original unbalanced training set.
  if useAdaSyn == 1
      
      [XAdaSyn, yAda] = ADASYN(Xtrain(TrainingIndices(:,j),:), Ytrain(TrainingIndices(:,j)), 1, [], [], false); 
      
      trainedSVMModel = fitcsvm([XAdaSyn;Xtrain(TrainingIndices(:,j),:)], [double(yAda);Ytrain(TrainingIndices(:,j))],'BoxConstraint',C(c),'KernelFunction','linear','ClassNames',[0,1]);
      trainedSVMModel = fitPosterior(trainedSVMModel, [XAdaSyn;Xtrain(TrainingIndices(:,j),:)], [double(yAda);Ytrain(TrainingIndices(:,j))] );

      [Yhat,predicted_probs] = predict(trainedSVMModel, Xtrain(TestIndices(:,j),:)    );
      % function that computes the PCC, requires real y-values, predicted_y
      % values.
      PCC =  sum( (Ytrain(TestIndices(:,j))  == Yhat) )/numel(Ytrain(TestIndices(:,j)));
      kfoldPerformanceMeasuresMatrix(j,1) = PCC;

      prior1 = mean([double(yAda);Ytrain(TrainingIndices(:,j))]); 
      prior0 = 1 - prior1;

      [AUC,PG_index, H_measure ] = computeAUC_PGindex_Hvalue(Ytrain(TestIndices(:,j)), predicted_probs(:,2), prior1, prior0);

      kfoldPerformanceMeasuresMatrix(j,3) = AUC;
      kfoldPerformanceMeasuresMatrix(j,4) = H_measure;
      kfoldPerformanceMeasuresMatrix(j,5) = PG_index;

      [KS_value] = computeKSvalue(Ytrain(TestIndices(:,j)),predicted_probs(:,2));

      kfoldPerformanceMeasuresMatrix(j,2) = KS_value;

      BScore = mean( (predicted_probs(:,2) - Ytrain(TestIndices(:,j)) ).^2);
      kfoldPerformanceMeasuresMatrix(j,6) = BScore;
      
  else
      trainedSVMModel = fitcsvm(Xtrain(TrainingIndices(:,j),:),Ytrain(TrainingIndices(:,j)),'BoxConstraint',C(c),'KernelFunction','linear','ClassNames',[0,1]);
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
  
end

PerformanceMeasuresMatrix(c,:) = mean(kfoldPerformanceMeasuresMatrix);


end

% extract the indices of the corresponding optimal parameter C for each
% measure:
[~, ind] = max(PerformanceMeasuresMatrix(:,1)); % PCC
C_optimal_PCC =  C(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,2)); % KS
C_optimal_KS =  C(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,3)); % AUC
C_optimal_AUC =  C(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,4)); % Hmeasure
C_optimal_H =  C(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,5)); % PG_index
C_optimal_PG =  C(ind);
[~, ind] = min(PerformanceMeasuresMatrix(:,6)); % BS
C_optimal_BS =  C(ind);

end