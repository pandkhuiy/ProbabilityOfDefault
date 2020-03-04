function [cycles_optimal_PCC, cycles_optimal_AUC, cycles_optimal_PG, cycles_optimal_BS, cycles_optimal_H, cycles_optimal_KS] =  gridSearchHyperParamBoostingDT(Xtrain,ytrain, Xvalid,Yvalid,y1,...
                   Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG,...
                   Prune_and_MinLeafSize_optimal_BS, Prune_and_MinLeafSize_optimal_KS, Prune_and_MinLeafSize_optimal_H)
%% This function uses a linear grid search to find the optimal number of cycles in the boosting algorithm on Decision trees for each performance measure.
% The decision trees are boosted with previously obtained optimal hyper
% parameters: Prune yes or not, AND MinLeafSize  (these are given as input)

% Boosting cycles vector: range of values:
NumberCyclesVector = [10 50 100 250 500 1000];

% columns are the six performance metrics and the rows are the six values
% of the hyperparameter No. of cycles.
% PCC, KS ----- AUC, Hmeasure, PG index ----- BS (in this order).
PerformanceMeasuresMatrix =  zeros(numel(NumberCyclesVector),6);

for c = 1:length(NumberCyclesVector)
    c
        
    % Use the boosting method for Decision trees with previously obtained
    % optimal hyperparameters for each performance measure.
    templateTreePCC = templateTree('Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PCC{2} );
    templateTreeAUC = templateTree('Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_AUC{2} );
    templateTreePG = templateTree('Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PG{2} );
    templateTreeBS = templateTree('Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_BS{2} );
    templateTreeKS = templateTree('Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_KS{2} );
    templateTreeH = templateTree('Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_H{2} );

    MdlPCC = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreePCC,'NumLearningCycles',NumberCyclesVector(c),'ScoreTransform','logit');
    MdlAUC = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreeAUC,'NumLearningCycles',NumberCyclesVector(c),'ScoreTransform','logit');
    MdlPG = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreePG,'NumLearningCycles',NumberCyclesVector(c),'ScoreTransform','logit');
    MdlBS = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreeBS,'NumLearningCycles',NumberCyclesVector(c),'ScoreTransform','logit');
    MdlKS = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreeKS,'NumLearningCycles',NumberCyclesVector(c),'ScoreTransform','logit');
    MdlH = fitcensemble(Xtrain,ytrain,'Method','AdaBoostM1','Learners',templateTreeH,'NumLearningCycles',NumberCyclesVector(c),'ScoreTransform','logit');

    % make predictions with the boosted ensembles for each measure.
    [~,predictedProbsPCC] = predict(MdlPCC,Xvalid);
    tempPCC = predictedProbsPCC(:,2);

    [~,predictedProbsAUC] = predict(MdlAUC,Xvalid);
    tempAUC = predictedProbsAUC(:,2);

    [~,predictedProbsPG] = predict(MdlPG,Xvalid);
    tempPG = predictedProbsPG(:,2);

    [~,predictedProbsBS] = predict(MdlBS,Xvalid);
    tempBS = predictedProbsBS(:,2);

    [~,predictedProbsKS] = predict(MdlKS,Xvalid);
    tempKS = predictedProbsKS(:,2);

    [~,predictedProbsH] = predict(MdlH,Xvalid);
    tempH = predictedProbsH(:,2);

    sortedProbs = sort(tempPCC,'descend'); %sort probabilities
    t = sortedProbs(round(mean(y1)*size(sortedProbs,1)));
    YhatPCC = tempPCC > t;
   
    % function that computes the PCC, requires true y-values and predicted y-values.
    PCC =  sum( (Yvalid == YhatPCC) )/numel(Yvalid);
    PerformanceMeasuresMatrix(c,1) = PCC;

    prior1 = mean(y1); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Yvalid, tempAUC, prior1, prior0);
    PerformanceMeasuresMatrix(c,3) = AUC;


    [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Yvalid, tempPG, prior1, prior0);
    PerformanceMeasuresMatrix(c,5) = PG_index;


    [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Yvalid, tempH, prior1, prior0);
    PerformanceMeasuresMatrix(c,4) = H_measure ;


    BScore = mean( (tempBS - Yvalid).^2);
    PerformanceMeasuresMatrix(c,6) = BScore;

    KS_value = computeKSvalue(Yvalid,tempKS);
    PerformanceMeasuresMatrix(c,2) = KS_value ;

end

% extract the indices of the corresponding optimal parameter C for each
% measure:
[~, ind] = max(PerformanceMeasuresMatrix(:,1)); % PCC
cycles_optimal_PCC =  NumberCyclesVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,2)); % KS
cycles_optimal_KS =  NumberCyclesVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,3)); % AUC
cycles_optimal_AUC =  NumberCyclesVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,4)); % Hmeasure
cycles_optimal_H =  NumberCyclesVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,5)); % PG_index
cycles_optimal_PG =  NumberCyclesVector(ind);
[~, ind] = min(PerformanceMeasuresMatrix(:,6)); % BS
cycles_optimal_BS =  NumberCyclesVector(ind);

end