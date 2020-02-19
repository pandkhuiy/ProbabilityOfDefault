function [C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS,PCC,AUC, PG,BS, KS, H] =  gridSearchHyperParamLinearSVM(Xtrain, Ytrain,Xtest,Ytest)
%% Perform k fold cross validation to determine the optimal penalty parameter C for the linear SVM method
% in order to get probability score, we use Platt's method to callibrate
% the classifications in the test set to get probability scores through the
% use of the in-built MATLAB function fitPosterior(SVMModel, X,Y) where
% SVMModel is the trained SVM model and X and Y are the training data.


C = [0.0001;0.0005;0.001;0.005;0.01;0.05;0.1;1];

% This kx6 matrix contains the average performance values computed after k
% fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
PerformanceMeasuresMatrix =  zeros(numel(C),6);

for  c = 1:numel(C)
    c 

    % train the linear SVM model and use them for predicting whether the loan corresponding
    % to the features will be granted credit or reject (will default or not).
    % Use these classifcations and predicted probs for computing the six performance measures. 
    % if useAdaSyn == 1,  we use AdaSyn sampling method to create synthetic balanced training
    % data set, else we just use the original unbalanced training set.

    trainedSVMModel = fitcsvm(Xtrain,Ytrain,'BoxConstraint',C(c),'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModel = fitPosterior(trainedSVMModel, Xtrain, Ytrain);

    [Yhat,predicted_probs] = predict(trainedSVMModel, Xtest   );
    % function that computes the PCC, requires real y-values, predicted_y
    % values.
    PCC =  sum( (Ytest  == Yhat) )/numel(Ytest);
    PerformanceMeasuresMatrix(c,1) = PCC;

    prior1 = mean(Ytrain); 
    prior0 = 1 - prior1;

    [AUC,PG_index, H_measure ] = computeAUC_PGindex_Hvalue(Ytest, predicted_probs(:,2), prior1, prior0);

    PerformanceMeasuresMatrix(c,3) = AUC;
    PerformanceMeasuresMatrix(c,4) = H_measure;
    PerformanceMeasuresMatrix(c,5) = PG_index;

    [KS_value] = computeKSvalue(Ytest,predicted_probs(:,2));

    PerformanceMeasuresMatrix(c,2) = KS_value;

    BScore = mean( (predicted_probs(:,2) - Ytest ).^2);
    PerformanceMeasuresMatrix(c,6) = BScore;
  
end

% extract the indices of the corresponding optimal parameter C for each
% measure:
[PCC, ind] = max(PerformanceMeasuresMatrix(:,1)); % PCC
C_optimal_PCC =  C(ind);
[KS, ind] = max(PerformanceMeasuresMatrix(:,2)); % KS
C_optimal_KS =  C(ind);
[AUC, ind] = max(PerformanceMeasuresMatrix(:,3)); % AUC
C_optimal_AUC =  C(ind);
[H, ind] = max(PerformanceMeasuresMatrix(:,4)); % Hmeasure
C_optimal_H =  C(ind);
[PG, ind] = max(PerformanceMeasuresMatrix(:,5)); % PG_index
C_optimal_PG =  C(ind);
[BS, ind] = min(PerformanceMeasuresMatrix(:,6)); % BS
C_optimal_BS =  C(ind);

end