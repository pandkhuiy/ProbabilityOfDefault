function [Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG, Prune_and_MinLeafSize_optimal_BS,...
          Prune_and_MinLeafSize_optimal_H, Prune_and_MinLeafSize_optimal_KS] =  optimizeHyperParamDT(Xtrain, Ytrain)
%% Determine the optimal hyper parameters of Decision Trees,which is MinLeafSize and Prune, with 5 fold cross validation.
% Prune_and_MinLeafSize_optimal_PCC,
% Prune_and_MinLeafSize_optimal_AUC, ..., Prune_and_MinLeafSize_optimal_KS are
% 2x1 cell arrays containing the optimal hyperparameters for each measure.
% that is, pruning: on/of  and mininum leaf size.
k = 5;

Prune =  cell(1,2);
Prune(1) = {'on'};
Prune(2) = {'off'};

MinLeafSizeVector = [0.01;0.025; 0.05;0.1;0.25;0.5]';

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

        [TrainingIndices, TestIndices] = DoKfoldCrossValid(Ytrain,k);
        %TrainingIndices = logical(indices);  
        %TestIndices = logical(1-indices);  

        % This kx6 matrix contains the average performance values computed after k
        % fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
        kfoldPerformanceMeasuresMatrix =  zeros(k,6);

        for j = 1:k
            %j
            Tree = fitctree(Xtrain(TrainingIndices(:,j),:),Ytrain(TrainingIndices(:,j)) ,'Prune', Prune{l} ,'MinLeafSize', round( MinLeafSizeVector(h)*size(Xtrain(TrainingIndices(:,j),:),1) ) );

            % Construct new probability scores
            [~, Prob_scores,~,~] = predict(Tree,Xtrain(TestIndices(:,j),:));
            Prob_scores = Prob_scores(:,2);

            % function that computes the PCC, requires real y-values, predicted_y
            % values.
            t = mean(Ytrain(TrainingIndices(:,j)));
            classifications_test = Prob_scores > t;

            PCC =  sum( (Ytrain(TestIndices(:,j))  == (classifications_test)) )/numel(Ytrain(TestIndices(:,j)));
            kfoldPerformanceMeasuresMatrix(j,1) = PCC;

            prior1 = mean(Ytrain(TrainingIndices(:,j))); 
            prior0 = 1 - prior1;

            [AUC,PG_index, H_measure ] = computeAUC_PGindex_Hvalue(Ytrain(TestIndices(:,j)), Prob_scores, prior1, prior0);

            kfoldPerformanceMeasuresMatrix(j,3) = AUC;
            kfoldPerformanceMeasuresMatrix(j,4) = H_measure;
            kfoldPerformanceMeasuresMatrix(j,5) = PG_index;

            [KS_value] = computeKSvalue(Ytrain(TestIndices(:,j)),Prob_scores);

            kfoldPerformanceMeasuresMatrix(j,2) = KS_value;

            BScore = mean( (Prob_scores - Ytrain(TestIndices(:,j)) ).^2);
            kfoldPerformanceMeasuresMatrix(j,6) = BScore;

        end

        MeanVector = mean(kfoldPerformanceMeasuresMatrix);

        PerformanceMeasuresMatrixPCC(l,h) = MeanVector(1);
        PerformanceMeasuresMatrixKS(l,h) = MeanVector(2);
        PerformanceMeasuresMatrixAUC(l,h) = MeanVector(3);
        PerformanceMeasuresMatrixH(l,h) = MeanVector(4);
        PerformanceMeasuresMatrixPG(l,h) = MeanVector(5);
        PerformanceMeasuresMatrixBS(l,h) = MeanVector(6);

    end
end

% extract the indices of the corresponding optimal parameter C for each
% measure:
%PCC
[MaxRow, ind] = max(PerformanceMeasuresMatrixPCC); 
[~, h_index] = max(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_PCC = cell(1,2);
Prune_and_MinLeafSize_optimal_PCC(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_PCC(2) =  {MinLeafSizeVector(h_index)};

%KS
[MaxRow, ind] = max(PerformanceMeasuresMatrixKS); 
[~, h_index] = max(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_KS = cell(1,2);
Prune_and_MinLeafSize_optimal_KS(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_KS(2) =  {MinLeafSizeVector(h_index)};

%AUC
[MaxRow, ind] = max(PerformanceMeasuresMatrixAUC); 
[~, h_index] = max(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_AUC = cell(1,2);
Prune_and_MinLeafSize_optimal_AUC(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_AUC(2) =  {MinLeafSizeVector(h_index)};

%H measure
[MaxRow, ind] = max(PerformanceMeasuresMatrixH); 
[~, h_index] = max(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_H = cell(1,2);
Prune_and_MinLeafSize_optimal_H(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_H(2) =  {MinLeafSizeVector(h_index)};

%PG
[MaxRow, ind] = max(PerformanceMeasuresMatrixPG); 
[~, h_index] = max(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_PG = cell(1,2);
Prune_and_MinLeafSize_optimal_PG(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_PG(2) =  {MinLeafSizeVector(h_index)};

%BS
[MaxRow, ind] = min(PerformanceMeasuresMatrixBS); 
[~, h_index] = min(MaxRow);
l_index = ind(h_index);
Prune_and_MinLeafSize_optimal_BS = cell(1,2);
Prune_and_MinLeafSize_optimal_BS(1) =  Prune(l_index);
Prune_and_MinLeafSize_optimal_BS(2) =  {MinLeafSizeVector(h_index)};

end