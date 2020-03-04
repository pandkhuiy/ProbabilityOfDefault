function [C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS,...
    S_optimal_PCC, S_optimal_AUC, S_optimal_PG, S_optimal_BS, S_optimal_H, S_optimal_KS] =  gridSearchHyperParamRbfSVM(Xtrain, Ytrain,Xvalid,Yvalid,y1)
%% Perform a 2d grid search to determine the optimal penalty parameter C and optimal rbf kernel scale s for the rbf SVM method
% in order to get probability score, we use Platt's method to callibrate
% the classifications in the test set to get probability scores through the
% use of the in-built MATLAB function fitPosterior(SVMModel, X,Y) where
% SVMModel is the trained SVM model and X and Y are the training data.


C = [0.0001;0.0005;0.001;0.01;0.1;1];
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
        %s

        % train the linear SVM model and use them for predicting whether the loan corresponding
        % to the features will be granted credit or reject (will default or not).
        % Use these classifcations and predicted probs for computing the six performance measures.
        trainedSVMModel = fitcsvm(Xtrain,Ytrain,'BoxConstraint',C(c),'KernelFunction','rbf','KernelScale',S(s),'ClassNames',[0,1]);
        trainedSVMModel = fitPosterior(trainedSVMModel, Xtrain, Ytrain );

        [~,Prob_scores] = predict(trainedSVMModel, Xvalid  );
        Prob_scores = Prob_scores(:,2);
        % function that computes the PCC, requires real y-values, predicted_y
        % values.
        predsort = sort(Prob_scores,'descend'); %sort probabilities
        t = predsort(round(mean(y1)*size(Prob_scores,1)));
        classifications_test = Prob_scores > t;
        
        tempPCC =  sum( (Yvalid)  == (classifications_test))/numel(Yvalid);
        PerformanceMeasuresMatrixPCC(c,s) = tempPCC;

        prior1 = mean(y1); 
        prior0 = 1 - prior1;

        [tempAUC,tempPG_index, tempH_measure ] = computeAUC_PGindex_Hvalue(Yvalid, Prob_scores, prior1, prior0);

        PerformanceMeasuresMatrixAUC(c,s) = tempAUC;
        PerformanceMeasuresMatrixH(c,s) = tempH_measure;
        PerformanceMeasuresMatrixPG(c,s) = tempPG_index;

        [tempKS_value] = computeKSvalue(Yvalid,Prob_scores);
        PerformanceMeasuresMatrixKS(c,s) = tempKS_value;

        tempBScore = mean( (Prob_scores - Yvalid ).^2);
        PerformanceMeasuresMatrixBS(c,s) = tempBScore;
       
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
[MinRow, ind] = min(PerformanceMeasuresMatrixBS); 
[~, s_index] = min(MinRow);
c_index = ind(s_index);
C_optimal_BS =  C(c_index);
S_optimal_BS =  S(s_index);

end