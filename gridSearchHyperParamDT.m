function [Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG, Prune_and_MinLeafSize_optimal_BS,...
          Prune_and_MinLeafSize_optimal_H, Prune_and_MinLeafSize_optimal_KS,PCC,AUC, PG,BS, KS, H] =  gridSearchHyperParamDT(Xtrain, Ytrain,Xtest,Ytest)
%% Determine the optimal hyper parameters of Decision Trees,which is MinLeafSize and Prune, with a 2d grid search.
% Prune_and_MinLeafSize_optimal_PCC, 
% Prune_and_MinLeafSize_optimal_AUC, ..., Prune_and_MinLeafSize_optimal_KS are
% 2x1 cell arrays containing the optimal hyperparameters for each measure.
% that is, pruning: on/of  and mininum leaf size.


Prune =  cell(1,2);
Prune(1) = {'on'};
Prune(2) = {'off'};

MinLeafSizeVector = [1;10; 100;500;1000;2500;5000]';

% This kx6 matrix contains the average performance values computed after k
% fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
PerformanceMeasuresMatrixPCC =  zeros(numel(Prune),numel(MinLeafSizeVector));
PerformanceMeasuresMatrixKS =  zeros(numel(Prune),numel(MinLeafSizeVector));
PerformanceMeasuresMatrixAUC =  zeros(numel(Prune),numel(MinLeafSizeVector));
PerformanceMeasuresMatrixH =  zeros(numel(Prune),numel(MinLeafSizeVector));
PerformanceMeasuresMatrixPG =  zeros(numel(Prune),numel(MinLeafSizeVector));
PerformanceMeasuresMatrixBS =  zeros(numel(Prune),numel(MinLeafSizeVector));


for  l = 1:numel(Prune)
    l 
    for  h = 1:numel(MinLeafSizeVector)
        h


        Tree = fitctree(Xtrain,Ytrain ,'MinLeafSize', round( MinLeafSizeVector(h) ),'Prune', Prune{l} );

        % Construct new probability scores
        [~, Prob_scores,~,~] = predict(Tree,Xtest);
        Prob_scores = Prob_scores(:,2);

        % function that computes the PCC, requires real y-values, predicted_y
        % values.
        t = mean(Ytrain);
        classifications_test = Prob_scores >= t;

        tempPCC =  sum( (Ytest)  == (classifications_test))/numel(Ytest);
        PerformanceMeasuresMatrixPCC(l,h) = tempPCC;

        prior1 = mean(Ytrain); 
        prior0 = 1 - prior1;

        [tempAUC,tempPG_index, tempH_measure ] = computeAUC_PGindex_Hvalue(Ytest, Prob_scores, prior1, prior0);

        PerformanceMeasuresMatrixAUC(l,h) = tempAUC;
        PerformanceMeasuresMatrixH(l,h) = tempH_measure;
        PerformanceMeasuresMatrixPG(l,h) = tempPG_index;

        [tempKS_value] = computeKSvalue(Ytest,Prob_scores);
        PerformanceMeasuresMatrixKS(l,h) = tempKS_value;

        tempBScore = mean( (Prob_scores - Ytest ).^2);
        PerformanceMeasuresMatrixBS(l,h) = tempBScore;
        
    end
end

% extract the indices of the corresponding optimal parameter C for each
% measure:
%PCC
[MaxRow, ind] = max(PerformanceMeasuresMatrixPCC); 
[PCC, h_index] = max(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_PCC = cell(1,2);
Prune_and_MinLeafSize_optimal_PCC(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_PCC(2) =  {MinLeafSizeVector(h_index)};

%KS
[MaxRow, ind] = max(PerformanceMeasuresMatrixKS); 
[KS, h_index] = max(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_KS = cell(1,2);
Prune_and_MinLeafSize_optimal_KS(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_KS(2) =  {MinLeafSizeVector(h_index)};

%AUC
[MaxRow, ind] = max(PerformanceMeasuresMatrixAUC); 
[AUC, h_index] = max(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_AUC = cell(1,2);
Prune_and_MinLeafSize_optimal_AUC(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_AUC(2) =  {MinLeafSizeVector(h_index)};

%H measure
[MaxRow, ind] = max(PerformanceMeasuresMatrixH); 
[H, h_index] = max(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_H = cell(1,2);
Prune_and_MinLeafSize_optimal_H(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_H(2) =  {MinLeafSizeVector(h_index)};

%PG
[MaxRow, ind] = max(PerformanceMeasuresMatrixPG); 
[PG, h_index] = max(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_PG = cell(1,2);
Prune_and_MinLeafSize_optimal_PG(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_PG(2) =  {MinLeafSizeVector(h_index)};

%BS
[MaxRow, ind] = min(PerformanceMeasuresMatrixBS); 
[BS, h_index] = min(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_BS = cell(1,2);
Prune_and_MinLeafSize_optimal_BS(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_BS(2) =  {MinLeafSizeVector(h_index)};

end