function [PCC, AUC, PG,BS, KS, H] = LinearSVM(X1,X2,X3,X4,y1,y2,y3,y4)
%% This code uses support vector machines with linear kernel functions to make classifications for the test sample
% where we make use of 4 fold cross validation, that is, we do 4 iterations
% where in each iteration we train the SVM model on three quarter data and
% test it on one test quarter data.
useAdaSyn = 0;
seed = 1;
rng('default');
rng(seed);

%X = normalize(X);

% This vector is (4 by 1) and represents the PCC values for each cv
% iteration. Same for the other five performance measures.
PCC_vector = zeros(4,1);
AUC_vector = zeros(4,1);
PGini_vector = zeros(4,1);
BScore_vector = zeros(4,1);
Hmeasure_vector = zeros(4,1);
KSvalue_vector = zeros(4,1);


if useAdaSyn == 1
    number = 0;
    while number <5
    temp = datasample([X1 y1;X2 y2;X3 y3],5000, 'Replace',false);
    Xtrain123 = temp(:,1:end-1); 
    ytrain123 = temp(:,end);
    number = sum(ytrain123);
    end
else
    number = 0;
    while number <5
    temp = datasample([X1 y1;X2 y2;X3 y3],10000, 'Replace',false);
    Xtrain123 = temp(:,1:end-1); 
    ytrain123 = temp(:,end);
    number = sum(ytrain123);
    end
end
number = 0;
while number <5 
temp = datasample([X4 y4],5000, 'Replace',false);
Xtest4 = temp(:,1:end-1);
Ytest4 = temp(:,end);
number = sum(Ytest4);
end

% Determine the optimal penalty constant C with k fold cross validation
% with k = 5 on the training data.
[C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS] =  optimizeLinearSVM(Xtrain123, ytrain123, useAdaSyn);

if useAdaSyn == 1
    [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
    
    % Now fit a SVM with the optimal hyperparameter for each performance
    % measure
    trainedSVMModelPCC = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_PCC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelKS = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_KS,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelAUC = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_AUC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelPG = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_PG,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelH = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_H,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelBS = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_BS,'KernelFunction','linear','ClassNames',[0,1]);
    
    trainedSVMModelPCC = fitPosterior(trainedSVMModelPCC, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelKS = fitPosterior(trainedSVMModelKS, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelAUC = fitPosterior(trainedSVMModelAUC, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelPG = fitPosterior(trainedSVMModelPG, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelH = fitPosterior(trainedSVMModelH, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelBS = fitPosterior(trainedSVMModelBS, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    
    [YhatPCC,~] = predict(trainedSVMModelPCC, Xtest4);
    [~,predicted_probsKS] = predict(trainedSVMModelKS, Xtest4);
    [~,predicted_probsAUC] = predict(trainedSVMModelAUC, Xtest4);
    [~,predicted_probsPG] = predict(trainedSVMModelPG, Xtest4);
    [~,predicted_probsH] = predict(trainedSVMModelH, Xtest4); 
    [~,predicted_probsBS] = predict(trainedSVMModelBS, Xtest4);
    
    % function that computes the PCC, requires real y-values, predicted_y
    % values.
    PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
    PCC_vector(1) = PCC;

    prior1 = mean([double(yAda);ytrain123]); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC(:,2), prior1, prior0);
    AUC_vector(1) = AUC;

    [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG(:,2), prior1, prior0);
    PGini_vector(1) = PG_index;

    [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH(:,2), prior1, prior0);
    Hmeasure_vector(1) = H_measure;


    BScore = mean( (predicted_probsBS(:,2) - Ytest4).^2);
    BScore_vector(1) = BScore;

    KS_value = computeKSvalue(Ytest4,predicted_probsKS(:,2));
    KSvalue_vector(1) = KS_value;
    
else
    
    % Now fit a SVM with the optimal hyperparameter for each performance
    % measure
    trainedSVMModelPCC = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_PCC,'KernelFunction','linear','ClassNames',[0,1]);1
    trainedSVMModelKS = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_KS,'KernelFunction','linear','ClassNames',[0,1]);2
    trainedSVMModelAUC = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_AUC,'KernelFunction','linear','ClassNames',[0,1]);3
    trainedSVMModelPG = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_PG,'KernelFunction','linear','ClassNames',[0,1]);4
    trainedSVMModelH = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_H,'KernelFunction','linear','ClassNames',[0,1]);5
    trainedSVMModelBS = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_BS,'KernelFunction','linear','ClassNames',[0,1]);6

    trainedSVMModelPCC = fitPosterior(trainedSVMModelPCC, Xtrain123,ytrain123);7
    trainedSVMModelKS = fitPosterior(trainedSVMModelKS, Xtrain123,ytrain123);8
    trainedSVMModelAUC = fitPosterior(trainedSVMModelAUC, Xtrain123,ytrain123);9
    trainedSVMModelPG = fitPosterior(trainedSVMModelPG, Xtrain123,ytrain123);10
    trainedSVMModelH = fitPosterior(trainedSVMModelH, Xtrain123,ytrain123);11
    trainedSVMModelBS = fitPosterior(trainedSVMModelBS, Xtrain123,ytrain123);12

    [YhatPCC,~] = predict(trainedSVMModelPCC, Xtest4);
    [~,predicted_probsKS] = predict(trainedSVMModelKS, Xtest4);
    [~,predicted_probsAUC] = predict(trainedSVMModelAUC, Xtest4);
    [~,predicted_probsPG] = predict(trainedSVMModelPG, Xtest4);
    [~,predicted_probsH] = predict(trainedSVMModelH, Xtest4); 
    [~,predicted_probsBS] = predict(trainedSVMModelBS, Xtest4); 


    % function that computes the PCC, requires real y-values, predicted_y
    % values.
    PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
    PCC_vector(1) = PCC;

    prior1 = mean(ytrain123); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC(:,2), prior1, prior0);
    AUC_vector(1) = AUC;

    [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG(:,2), prior1, prior0);
    PGini_vector(1) = PG_index;

    [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH(:,2), prior1, prior0);
    Hmeasure_vector(1) = H_measure;


    BScore = mean( (predicted_probsBS(:,2) - Ytest4).^2);
    BScore_vector(1) = BScore;

    KS_value = computeKSvalue(Ytest4,predicted_probsKS(:,2));
    KSvalue_vector(1) = KS_value;
end
%%
% Start with X124 and Y124 as training sets, and X3 and Y3 as test sets.
if useAdaSyn == 1
    number = 0;
    while number <5
    temp = datasample([X1 y1;X2 y2;X4 y4],5000, 'Replace',false);
    Xtrain123 = temp(:,1:end-1); 
    ytrain123 = temp(:,end);
    number = sum(ytrain123);
    end
else
    number = 0;
    while number <5
    temp = datasample([X1 y1;X2 y2;X4 y4],10000, 'Replace',false);
    Xtrain123 = temp(:,1:end-1); 
    ytrain123 = temp(:,end);
    number = sum(ytrain123);
    end
end

number = 0;
while number <5 
temp = datasample([X3 y3],5000, 'Replace',false);
Xtest4 = temp(:,1:end-1);
Ytest4 = temp(:,end);
number = sum(Ytest4);
end

% Determine the optimal penalty constant C with k fold cross validation
% with k = 5 on the training data.
[C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS] =  optimizeLinearSVM(Xtrain123, ytrain123, useAdaSyn);

if useAdaSyn == 1
    [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
    
    % Now fit a SVM with the optimal hyperparameter for each performance
    % measure
    trainedSVMModelPCC = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_PCC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelKS = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_KS,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelAUC = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_AUC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelPG = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_PG,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelH = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_H,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelBS = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_BS,'KernelFunction','linear','ClassNames',[0,1]);
    
    trainedSVMModelPCC = fitPosterior(trainedSVMModelPCC, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelKS = fitPosterior(trainedSVMModelKS, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelAUC = fitPosterior(trainedSVMModelAUC, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelPG = fitPosterior(trainedSVMModelPG, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelH = fitPosterior(trainedSVMModelH, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelBS = fitPosterior(trainedSVMModelBS, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    
    [YhatPCC,~] = predict(trainedSVMModelPCC, Xtest4);
    [~,predicted_probsKS] = predict(trainedSVMModelKS, Xtest4);
    [~,predicted_probsAUC] = predict(trainedSVMModelAUC, Xtest4);
    [~,predicted_probsPG] = predict(trainedSVMModelPG, Xtest4);
    [~,predicted_probsH] = predict(trainedSVMModelH, Xtest4); 
    [~,predicted_probsBS] = predict(trainedSVMModelBS, Xtest4);
    
    % function that computes the PCC, requires real y-values, predicted_y
    % values.
    PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
    PCC_vector(2) = PCC;

    prior1 = mean([double(yAda);ytrain123]); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC(:,2), prior1, prior0);
    AUC_vector(2) = AUC;

    [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG(:,2), prior1, prior0);
    PGini_vector(2) = PG_index;

    [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH(:,2), prior1, prior0);
    Hmeasure_vector(2) = H_measure;


    BScore = mean( (predicted_probsBS(:,2) - Ytest4).^2);
    BScore_vector(2) = BScore;

    KS_value = computeKSvalue(Ytest4,predicted_probsKS(:,2));
    KSvalue_vector(2) = KS_value;
    
else
    
    % Now fit a SVM with the optimal hyperparameter for each performance
    % measure
    trainedSVMModelPCC = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_PCC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelKS = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_KS,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelAUC = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_AUC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelPG = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_PG,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelH = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_H,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelBS = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_BS,'KernelFunction','linear','ClassNames',[0,1]);

    trainedSVMModelPCC = fitPosterior(trainedSVMModelPCC, Xtrain123,ytrain123);
    trainedSVMModelKS = fitPosterior(trainedSVMModelKS, Xtrain123,ytrain123);
    trainedSVMModelAUC = fitPosterior(trainedSVMModelAUC, Xtrain123,ytrain123);
    trainedSVMModelPG = fitPosterior(trainedSVMModelPG, Xtrain123,ytrain123);
    trainedSVMModelH = fitPosterior(trainedSVMModelH, Xtrain123,ytrain123);
    trainedSVMModelBS = fitPosterior(trainedSVMModelBS, Xtrain123,ytrain123);

    [YhatPCC,~] = predict(trainedSVMModelPCC, Xtest4);
    [~,predicted_probsKS] = predict(trainedSVMModelKS, Xtest4);
    [~,predicted_probsAUC] = predict(trainedSVMModelAUC, Xtest4);
    [~,predicted_probsPG] = predict(trainedSVMModelPG, Xtest4);
    [~,predicted_probsH] = predict(trainedSVMModelH, Xtest4); 
    [~,predicted_probsBS] = predict(trainedSVMModelBS, Xtest4); 


    % function that computes the PCC, requires real y-values, predicted_y
    % values.
    PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
    PCC_vector(2) = PCC;

    prior1 = mean(ytrain123); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC(:,2), prior1, prior0);
    AUC_vector(2) = AUC;

    [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG(:,2), prior1, prior0);
    PGini_vector(2) = PG_index;

    [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH(:,2), prior1, prior0);
    Hmeasure_vector(2) = H_measure;


    BScore = mean( (predicted_probsBS(:,2) - Ytest4).^2);
    BScore_vector(2) = BScore;

    KS_value = computeKSvalue(Ytest4,predicted_probsKS(:,2));
    KSvalue_vector(2) = KS_value;
end
%%
% Start with X134 and Y134 as training sets, and X2 and Y2 as test sets.

if useAdaSyn == 1
    number = 0;
    while number <5
    temp = datasample([X1 y1;X3 y3;X4 y4],5000, 'Replace',false);
    Xtrain123 = temp(:,1:end-1); 
    ytrain123 = temp(:,end);
    number = sum(ytrain123);
    end
else
    number = 0;
    while number <5
    temp = datasample([X1 y1;X3 y3;X4 y4],10000, 'Replace',false);
    Xtrain123 = temp(:,1:end-1); 
    ytrain123 = temp(:,end);
    number = sum(ytrain123);
    end
end
 number = 0;
 while number <5 
  temp = datasample([X2 y2],5000, 'Replace',false);
  Xtest4 = temp(:,1:end-1);
  Ytest4 = temp(:,end);
  number = sum(Ytest4);
 end


% Determine the optimal penalty constant C with k fold cross validation
% with k = 5 on the training data.
[C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS] =  optimizeLinearSVM(Xtrain123, ytrain123, useAdaSyn);

if useAdaSyn == 1
    [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
    
    % Now fit a SVM with the optimal hyperparameter for each performance
    % measure
    trainedSVMModelPCC = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_PCC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelKS = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_KS,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelAUC = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_AUC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelPG = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_PG,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelH = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_H,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelBS = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_BS,'KernelFunction','linear','ClassNames',[0,1]);
    
    trainedSVMModelPCC = fitPosterior(trainedSVMModelPCC, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelKS = fitPosterior(trainedSVMModelKS, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelAUC = fitPosterior(trainedSVMModelAUC, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelPG = fitPosterior(trainedSVMModelPG, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelH = fitPosterior(trainedSVMModelH, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelBS = fitPosterior(trainedSVMModelBS, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    
    [YhatPCC,~] = predict(trainedSVMModelPCC, Xtest4);
    [~,predicted_probsKS] = predict(trainedSVMModelKS, Xtest4);
    [~,predicted_probsAUC] = predict(trainedSVMModelAUC, Xtest4);
    [~,predicted_probsPG] = predict(trainedSVMModelPG, Xtest4);
    [~,predicted_probsH] = predict(trainedSVMModelH, Xtest4); 
    [~,predicted_probsBS] = predict(trainedSVMModelBS, Xtest4);
    
    % function that computes the PCC, requires real y-values, predicted_y
    % values.
    PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
    PCC_vector(3) = PCC;

    prior1 = mean([double(yAda);ytrain123]); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC(:,2), prior1, prior0);
    AUC_vector(3) = AUC;

    [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG(:,2), prior1, prior0);
    PGini_vector(3) = PG_index;

    [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH(:,2), prior1, prior0);
    Hmeasure_vector(3) = H_measure;


    BScore = mean( (predicted_probsBS(:,2) - Ytest4).^2);
    BScore_vector(3) = BScore;

    KS_value = computeKSvalue(Ytest4,predicted_probsKS(:,2));
    KSvalue_vector(3) = KS_value;
    
else
    
    % Now fit a SVM with the optimal hyperparameter for each performance
    % measure
    trainedSVMModelPCC = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_PCC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelKS = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_KS,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelAUC = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_AUC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelPG = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_PG,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelH = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_H,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelBS = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_BS,'KernelFunction','linear','ClassNames',[0,1]);

    trainedSVMModelPCC = fitPosterior(trainedSVMModelPCC, Xtrain123,ytrain123);
    trainedSVMModelKS = fitPosterior(trainedSVMModelKS, Xtrain123,ytrain123);
    trainedSVMModelAUC = fitPosterior(trainedSVMModelAUC, Xtrain123,ytrain123);
    trainedSVMModelPG = fitPosterior(trainedSVMModelPG, Xtrain123,ytrain123);
    trainedSVMModelH = fitPosterior(trainedSVMModelH, Xtrain123,ytrain123);
    trainedSVMModelBS = fitPosterior(trainedSVMModelBS, Xtrain123,ytrain123);

    [YhatPCC,~] = predict(trainedSVMModelPCC, Xtest4);
    [~,predicted_probsKS] = predict(trainedSVMModelKS, Xtest4);
    [~,predicted_probsAUC] = predict(trainedSVMModelAUC, Xtest4);
    [~,predicted_probsPG] = predict(trainedSVMModelPG, Xtest4);
    [~,predicted_probsH] = predict(trainedSVMModelH, Xtest4); 
    [~,predicted_probsBS] = predict(trainedSVMModelBS, Xtest4); 


    % function that computes the PCC, requires real y-values, predicted_y
    % values.
    PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
    PCC_vector(3) = PCC;

    prior1 = mean(ytrain123); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC(:,2), prior1, prior0);
    AUC_vector(3) = AUC;

    [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG(:,2), prior1, prior0);
    PGini_vector(3) = PG_index;

    [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH(:,2), prior1, prior0);
    Hmeasure_vector(3) = H_measure;


    BScore = mean( (predicted_probsBS(:,2) - Ytest4).^2);
    BScore_vector(3) = BScore;

    KS_value = computeKSvalue(Ytest4,predicted_probsKS(:,2));
    KSvalue_vector(3) = KS_value;
end
%%
% Start with X234 and Y234 as training sets, and X1 and Y1 as test sets.

if useAdaSyn == 1
    number = 0;
    while number <5
    temp = datasample([X2 y2;X3 y3;X4 y4],5000, 'Replace',false);
    Xtrain123 = temp(:,1:end-1); 
    ytrain123 = temp(:,end);
    number = sum(ytrain123);
    end
else
    number = 0;
    while number <5
    temp = datasample([X2 y2;X3 y3;X4 y4],10000, 'Replace',false);
    Xtrain123 = temp(:,1:end-1); 
    ytrain123 = temp(:,end);
    number = sum(ytrain123);
    end
end

number = 0;
while number <5 
temp = datasample([X1 y1],5000, 'Replace',false);
Xtest4 = temp(:,1:end-1);
Ytest4 = temp(:,end);
number = sum(Ytest4);
end

% Determine the optimal penalty constant C with k fold cross validation
% with k = 5 on the training data.
[C_optimal_PCC, C_optimal_AUC, C_optimal_PG, C_optimal_BS, C_optimal_H, C_optimal_KS] =  optimizeLinearSVM(Xtrain123, ytrain123, useAdaSyn);

if useAdaSyn == 1
    [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
    
    % Now fit a SVM with the optimal hyperparameter for each performance
    % measure
    trainedSVMModelPCC = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_PCC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelKS = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_KS,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelAUC = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_AUC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelPG = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_PG,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelH = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_H,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelBS = fitcsvm([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'BoxConstraint',C_optimal_BS,'KernelFunction','linear','ClassNames',[0,1]);
    
    trainedSVMModelPCC = fitPosterior(trainedSVMModelPCC, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelKS = fitPosterior(trainedSVMModelKS, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelAUC = fitPosterior(trainedSVMModelAUC, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelPG = fitPosterior(trainedSVMModelPG, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelH = fitPosterior(trainedSVMModelH, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    trainedSVMModelBS = fitPosterior(trainedSVMModelBS, [XAdaSyn;Xtrain123],[double(yAda);ytrain123]);
    
    [YhatPCC,~] = predict(trainedSVMModelPCC, Xtest4);
    [~,predicted_probsKS] = predict(trainedSVMModelKS, Xtest4);
    [~,predicted_probsAUC] = predict(trainedSVMModelAUC, Xtest4);
    [~,predicted_probsPG] = predict(trainedSVMModelPG, Xtest4);
    [~,predicted_probsH] = predict(trainedSVMModelH, Xtest4); 
    [~,predicted_probsBS] = predict(trainedSVMModelBS, Xtest4);
    
    % function that computes the PCC, requires real y-values, predicted_y
    % values.
    PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
    PCC_vector(4) = PCC;

    prior1 = mean([double(yAda);ytrain123]); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC(:,2), prior1, prior0);
    AUC_vector(4) = AUC;

    [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG(:,2), prior1, prior0);
    PGini_vector(4) = PG_index;

    [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH(:,2), prior1, prior0);
    Hmeasure_vector(4) = H_measure;


    BScore = mean( (predicted_probsBS(:,2) - Ytest4).^2);
    BScore_vector(4) = BScore;

    KS_value = computeKSvalue(Ytest4,predicted_probsKS(:,2));
    KSvalue_vector(4) = KS_value;
    
else
    
    % Now fit a SVM with the optimal hyperparameter for each performance
    % measure
    trainedSVMModelPCC = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_PCC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelKS = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_KS,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelAUC = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_AUC,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelPG = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_PG,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelH = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_H,'KernelFunction','linear','ClassNames',[0,1]);
    trainedSVMModelBS = fitcsvm(Xtrain123,ytrain123,'BoxConstraint',C_optimal_BS,'KernelFunction','linear','ClassNames',[0,1]);

    trainedSVMModelPCC = fitPosterior(trainedSVMModelPCC, Xtrain123,ytrain123);
    trainedSVMModelKS = fitPosterior(trainedSVMModelKS, Xtrain123,ytrain123);
    trainedSVMModelAUC = fitPosterior(trainedSVMModelAUC, Xtrain123,ytrain123);
    trainedSVMModelPG = fitPosterior(trainedSVMModelPG, Xtrain123,ytrain123);
    trainedSVMModelH = fitPosterior(trainedSVMModelH, Xtrain123,ytrain123);
    trainedSVMModelBS = fitPosterior(trainedSVMModelBS, Xtrain123,ytrain123);

    [YhatPCC,~] = predict(trainedSVMModelPCC, Xtest4);
    [~,predicted_probsKS] = predict(trainedSVMModelKS, Xtest4);
    [~,predicted_probsAUC] = predict(trainedSVMModelAUC, Xtest4);
    [~,predicted_probsPG] = predict(trainedSVMModelPG, Xtest4);
    [~,predicted_probsH] = predict(trainedSVMModelH, Xtest4); 
    [~,predicted_probsBS] = predict(trainedSVMModelBS, Xtest4); 


    % function that computes the PCC, requires real y-values, predicted_y
    % values.
    PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
    PCC_vector(4) = PCC;

    prior1 = mean(ytrain123); prior0 = 1 - prior1;

    [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC(:,2), prior1, prior0);
    AUC_vector(4) = AUC;

    [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG(:,2), prior1, prior0);
    PGini_vector(4) = PG_index;

    [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH(:,2), prior1, prior0);
    Hmeasure_vector(4) = H_measure;


    BScore = mean( (predicted_probsBS(:,2) - Ytest4).^2);
    BScore_vector(4) = BScore;

    KS_value = computeKSvalue(Ytest4,predicted_probsKS(:,2));
    KSvalue_vector(4) = KS_value;
end
  
PCC = mean(PCC_vector);
AUC = mean(AUC_vector);
PG  = mean(PGini_vector);
BS  = mean(BScore_vector);
H   = mean(Hmeasure_vector);
KS  = mean(KSvalue_vector);

end