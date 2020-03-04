function [Bags_optimal_PCC, Bags_optimal_AUC, Bags_optimal_PG, Bags_optimal_BS, Bags_optimal_H, Bags_optimal_KS] =  gridSearchHyperParamBaggingDT(Xtrain,ytrain, Xvalid,Yvalid,y1,...
    Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG,...
    Prune_and_MinLeafSize_optimal_BS, Prune_and_MinLeafSize_optimal_KS, Prune_and_MinLeafSize_optimal_H)
%% This function determines the optimal number of DT bags for baggin DTs with optimal hyperparameters (given as input).
% A linear grid search is done for each performance measure.

% Bag the DTs with a range of hyperparameters:
NumberBagsVector = [5 10 25 50 100 1000];

% columns are the six performance metrics and the rows are the six values
% of the hyperparameter No. of bags.
% PCC, KS ----- AUC, Hmeasure, PG index ----- BS (in this order).
PerformanceMeasuresMatrix =  zeros(numel(NumberBagsVector),6);

for c = 1:length(NumberBagsVector)
    c

    PCCm = zeros(size(Xvalid,1),NumberBagsVector(c) );
    KSm  = zeros(size(Xvalid,1),NumberBagsVector(c));
    AUCm = zeros(size(Xvalid,1),NumberBagsVector(c));
    PGm  = zeros(size(Xvalid,1),NumberBagsVector(c));
    Hm   = zeros(size(Xvalid,1),NumberBagsVector(c));
    BSm  = zeros(size(Xvalid,1),NumberBagsVector(c));

    for b = 1:NumberBagsVector(c)
        % Construct a bootstrap sample with replacement.
        TrainMatrix=[Xtrain,ytrain];
        Lengthdata=size(TrainMatrix,1);
        number = 0;
        while number <5 
            temp1=ceil(Lengthdata*rand(Lengthdata,1));
            bsdata=TrainMatrix(temp1,:);
            bsx=bsdata(:,1:end-1);
            bsy=bsdata(:,end);
            number = sum(bsy);
        end

        TreePCC = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PCC{2} );
        TreeAUC = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_AUC{2} );
        TreePG = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize',   Prune_and_MinLeafSize_optimal_PG{2} );
        TreeBS = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize',   Prune_and_MinLeafSize_optimal_BS{2} );
        TreeH = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize',     Prune_and_MinLeafSize_optimal_H{2} );
        TreeKS = fitctree(bsx,bsy,'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize',   Prune_and_MinLeafSize_optimal_KS{2} );


        % Construct new probability scores on test data.
        [~, predicted_probsPCC,~,~] = predict(TreePCC,Xvalid);
        predicted_probsPCC = predicted_probsPCC(:,2);
        PCCm(:,b) = predicted_probsPCC;

        [~, predicted_probsAUC,~,~] = predict(TreeAUC,Xvalid);
        predicted_probsAUC = predicted_probsAUC(:,2);
        AUCm(:,b) = predicted_probsAUC;

        [~, predicted_probsKS,~,~] = predict(TreeKS,Xvalid);
        predicted_probsKS = predicted_probsKS(:,2);
        KSm(:,b) = predicted_probsKS;

        [~, predicted_probsBS,~,~] = predict(TreeBS,Xvalid);
        predicted_probsBS = predicted_probsBS(:,2);
        BSm(:,b) = predicted_probsBS;

        [~, predicted_probsPG,~,~] = predict(TreePG,Xvalid);
        predicted_probsPG = predicted_probsPG(:,2);
        PGm(:,b) = predicted_probsPG;

        [~, predicted_probsH,~,~] = predict(TreeH,Xvalid);
        predicted_probsH = predicted_probsH(:,2);
        Hm(:,b) = predicted_probsH;

    end

    tempPCC = mean(PCCm,2);
    tempKS=mean(KSm,2);
    tempH=mean(Hm,2);
    tempPG=mean(PGm,2);
    tempAUC=mean(AUCm,2);
    tempBS=mean(BSm,2);
    
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
Bags_optimal_PCC =  NumberBagsVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,2)); % KS
Bags_optimal_KS =  NumberBagsVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,3)); % AUC
Bags_optimal_AUC =  NumberBagsVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,4)); % Hmeasure
Bags_optimal_H =  NumberBagsVector(ind);
[~, ind] = max(PerformanceMeasuresMatrix(:,5)); % PG_index
Bags_optimal_PG =  NumberBagsVector(ind);
[~, ind] = min(PerformanceMeasuresMatrix(:,6)); % BS
Bags_optimal_BS =  NumberBagsVector(ind);

end