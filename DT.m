function [PCC, AUC, PG,BS, KS, H] = DT(X1,X2,X3,X4,y1,y2,y3,y4)
%% This code use the Decisions Tree function fitctree to make classifications for the test sample
% we make use of 4 fold cross validation (fold for each quarter of each
% year). Also, we will use 5 fold cross validation to fine tune the
% hyperparameters a the validation set which

useAdaSyn = 1;
seed = 1;
rng('default');
rng(seed);

%X = normalize(X);

% This vector is (P*2 by 1) and represents the PCC values for each cv
% iteration. Same for the other five performance measures.
PCC_vector = zeros(4,1);
AUC_vector = zeros(4,1);
PGini_vector = zeros(4,1);
BScore_vector = zeros(4,1);
Hmeasure_vector = zeros(4,1);
KSvalue_vector = zeros(4,1);


  % Start with X123 and Y123 as training sets, and X4 and Y4 as test sets.
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
  
  % Determine the optimal regularization parameter lambda (to prevent overfitting of the neural networks)
  % with k fold cross validation. 
  % k = 5 on the training data.
  [Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG, Prune_and_MinLeafSize_optimal_BS,...
          Prune_and_MinLeafSize_optimal_H, Prune_and_MinLeafSize_optimal_KS] =  optimizeDT(Xtrain123,ytrain123, useAdaSyn);
 
  % Train the models with optimal hyperparameters.
  if useAdaSyn == 1
  [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
  
  TreePCC = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_PCC{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeAUC = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_AUC{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreePG = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_PG{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeBS = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_BS{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeH = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize',     round( Prune_and_MinLeafSize_optimal_H{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeKS = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_KS{2}*size([XAdaSyn;Xtrain123],1) ) );
  
  [~, predicted_probsPCC,~,~] = predict(TreePCC,Xtest4);
  predicted_probsPCC = predicted_probsPCC(:,2);
  
  [~, predicted_probsKS,~,~] = predict(TreeKS,Xtest4);
  predicted_probsKS = predicted_probsKS(:,2);
  
  [~, predicted_probsH,~,~] = predict(TreeH,Xtest4);
  predicted_probsH = predicted_probsH(:,2);
  
  [~, predicted_probsPG,~,~] = predict(TreePG,Xtest4);
  predicted_probsPG = predicted_probsPG(:,2);
 
  [~, predicted_probsAUC,~,~] = predict(TreeAUC,Xtest4);
  predicted_probsAUC = predicted_probsAUC(:,2);
  
  [~, predicted_probsBS,~,~] = predict(TreeBS,Xtest4);
  predicted_probsBS = predicted_probsBS(:,2);
  
  
   t = mean([double(yAda);ytrain123]);
   YhatPCC = predicted_probsPCC > t;
  % mean(predicted_probsPCC)
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
  PCC_vector(1) = PCC;
  
  prior1 = mean([double(yAda);ytrain123]); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC, prior1, prior0);
  AUC_vector(1) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG, prior1, prior0);
  PGini_vector(1) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH, prior1, prior0);
  Hmeasure_vector(1) = H_measure;
  
  BScore = mean( (predicted_probsBS - Ytest4).^2);
  BScore_vector(1) = BScore;
 
  KS_value = computeKSvalue(Ytest4,predicted_probsKS);
  KSvalue_vector(1) = KS_value;
   
  else
      
  TreePCC = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PCC{2}*size(Xtrain123,1));
  TreeAUC = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_AUC{2}*size(Xtrain123,1));
  TreePG = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PG{2}*size(Xtrain123,1));
  TreeBS = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_BS{2}*size(Xtrain123,1));
  TreeH = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_H{2}*size(Xtrain123,1));
  TreeKS = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_KS{2}*size(Xtrain123,1));
  
  % Construct new probability score  on test data.
   [~, predicted_probsPCC,~,~] = predict(TreePCC,Xtest4);
  predicted_probsPCC = predicted_probsPCC(:,2);
  
  [~, predicted_probsKS,~,~] = predict(TreeKS,Xtest4);
  predicted_probsKS = predicted_probsKS(:,2);
  
  [~, predicted_probsH,~,~] = predict(TreeH,Xtest4);
  predicted_probsH = predicted_probsH(:,2);
  
  [~, predicted_probsPG,~,~] = predict(TreePG,Xtest4);
  predicted_probsPG = predicted_probsPG(:,2);
 
  [~, predicted_probsAUC,~,~] = predict(TreeAUC,Xtest4);
  predicted_probsAUC = predicted_probsAUC(:,2);
  
  [~, predicted_probsBS,~,~] = predict(TreeBS,Xtest4);
  predicted_probsBS = predicted_probsBS(:,2);
  
  t = mean(ytrain123);
  YhatPCC = predicted_probsPCC > t;
  % mean(predicted_probsPCC)
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
  PCC_vector(1) = PCC;
  
  prior1 = mean(ytrain123); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC, prior1, prior0);
  AUC_vector(1) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG, prior1, prior0);
  PGini_vector(1) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH, prior1, prior0);
  Hmeasure_vector(1) = H_measure;
    
  BScore = mean( (predicted_probsBS - Ytest4).^2);
  BScore_vector(1) = BScore;
 
  KS_value = computeKSvalue(Ytest4,predicted_probsKS);
  KSvalue_vector(1) = KS_value;
  end
   
   
  %%
  % Repeat the above process but with the second fold of the four fold
  % cross validation.
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

  % Determine the optimal regularization parameter lambda (to prevent overfitting of the neural networks)
  % with k fold cross validation. 
  % k = 5 on the training data.
  [Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG, Prune_and_MinLeafSize_optimal_BS,...
          Prune_and_MinLeafSize_optimal_H, Prune_and_MinLeafSize_optimal_KS] =  optimizeDT(Xtrain123,ytrain123, useAdaSyn);
  
  % Train the models where we split the training set Xtrain123 in to a train and
  % validation set which has the purpose to determine the appropriate
  % early stopping. Input [Xtrain123 XaDaSyn  is 100xN1 matrix and input y is 2XN1 (one row
  % for each class label, here first is defaulted loans, second is
  % non-defaulted loans).
  if useAdaSyn == 1
  
  [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
  
  TreePCC = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_PCC{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeAUC = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_AUC{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreePG = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_PG{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeBS = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_BS{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeH = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize',     round( Prune_and_MinLeafSize_optimal_H{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeKS = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_KS{2}*size([XAdaSyn;Xtrain123],1) ) );
  
    % Construct new probability score (using default softmax activation function in the output node) on test data.
  [~, predicted_probsPCC,~,~] = predict(TreePCC,Xtest4);
  predicted_probsPCC = predicted_probsPCC(:,2);
  
  [~, predicted_probsKS,~,~] = predict(TreeKS,Xtest4);
  predicted_probsKS = predicted_probsKS(:,2);
  
  [~, predicted_probsH,~,~] = predict(TreeH,Xtest4);
  predicted_probsH = predicted_probsH(:,2);
  
  [~, predicted_probsPG,~,~] = predict(TreePG,Xtest4);
  predicted_probsPG = predicted_probsPG(:,2);
 
  [~, predicted_probsAUC,~,~] = predict(TreeAUC,Xtest4);
  predicted_probsAUC = predicted_probsAUC(:,2);
  
  [~, predicted_probsBS,~,~] = predict(TreeBS,Xtest4);
  predicted_probsBS = predicted_probsBS(:,2);
  
  
   t = mean([double(yAda);ytrain123]);
   YhatPCC = predicted_probsPCC > t;
  % mean(predicted_probsPCC)
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
  PCC_vector(2) = PCC;
  
  prior1 = mean([double(yAda);ytrain123]); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC, prior1, prior0);
  AUC_vector(2) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG, prior1, prior0);
  PGini_vector(2) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH, prior1, prior0);
  Hmeasure_vector(2) = H_measure;
  
  BScore = mean( (predicted_probsBS - Ytest4).^2);
  BScore_vector(2) = BScore;
 
  KS_value = computeKSvalue(Ytest4,predicted_probsKS);
  KSvalue_vector(2) = KS_value;
  
  else
      
  TreePCC = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PCC{2}*size(Xtrain123,1));
  TreeAUC = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_AUC{2}*size(Xtrain123,1));
  TreePG = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PG{2}*size(Xtrain123,1));
  TreeBS = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_BS{2}*size(Xtrain123,1));
  TreeH = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_H{2}*size(Xtrain123,1));
  TreeKS = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_KS{2}*size(Xtrain123,1));
  
  % Construct new probability score  on test data.
   [~, predicted_probsPCC,~,~] = predict(TreePCC,Xtest4);
  predicted_probsPCC = predicted_probsPCC(:,2);
  
  [~, predicted_probsKS,~,~] = predict(TreeKS,Xtest4);
  predicted_probsKS = predicted_probsKS(:,2);
  
  [~, predicted_probsH,~,~] = predict(TreeH,Xtest4);
  predicted_probsH = predicted_probsH(:,2);
  
  [~, predicted_probsPG,~,~] = predict(TreePG,Xtest4);
  predicted_probsPG = predicted_probsPG(:,2);
 
  [~, predicted_probsAUC,~,~] = predict(TreeAUC,Xtest4);
  predicted_probsAUC = predicted_probsAUC(:,2);
  
  [~, predicted_probsBS,~,~] = predict(TreeBS,Xtest4);
  predicted_probsBS = predicted_probsBS(:,2);
  
  t = mean(ytrain123);
  YhatPCC = predicted_probsPCC > t;
  % mean(predicted_probsPCC)
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
  PCC_vector(2) = PCC;
  
  prior1 = mean(ytrain123); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC, prior1, prior0);
  AUC_vector(2) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG, prior1, prior0);
  PGini_vector(2) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH, prior1, prior0);
  Hmeasure_vector(2) = H_measure;
    
  BScore = mean( (predicted_probsBS - Ytest4).^2);
  BScore_vector(2) = BScore;
 
  KS_value = computeKSvalue(Ytest4,predicted_probsKS);
  KSvalue_vector(2) = KS_value;
  
  end
%%
  % Repeat the above process but with the third fold of the four fold
  % cross validation.
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
    
 % Determine the optimal regularization parameter lambda (to prevent overfitting of the neural networks)
  % with k fold cross validation. 
  % k = 5 on the training data.
  [Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG, Prune_and_MinLeafSize_optimal_BS,...
          Prune_and_MinLeafSize_optimal_H, Prune_and_MinLeafSize_optimal_KS] =  optimizeDT(Xtrain123,ytrain123, useAdaSyn);
   
  if useAdaSyn == 1
  
  [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
  
  TreePCC = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_PCC{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeAUC = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_AUC{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreePG = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_PG{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeBS = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_BS{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeH = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize',     round( Prune_and_MinLeafSize_optimal_H{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeKS = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_KS{2}*size([XAdaSyn;Xtrain123],1) ) );
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  [~, predicted_probsPCC,~,~] = predict(TreePCC,Xtest4);
  predicted_probsPCC = predicted_probsPCC(:,2);
  
  [~, predicted_probsKS,~,~] = predict(TreeKS,Xtest4);
  predicted_probsKS = predicted_probsKS(:,2);
  
  [~, predicted_probsH,~,~] = predict(TreeH,Xtest4);
  predicted_probsH = predicted_probsH(:,2);
  
  [~, predicted_probsPG,~,~] = predict(TreePG,Xtest4);
  predicted_probsPG = predicted_probsPG(:,2);
 
  [~, predicted_probsAUC,~,~] = predict(TreeAUC,Xtest4);
  predicted_probsAUC = predicted_probsAUC(:,2);
  
  [~, predicted_probsBS,~,~] = predict(TreeBS,Xtest4);
  predicted_probsBS = predicted_probsBS(:,2);
  
  
   t = mean([double(yAda);ytrain123]);
   YhatPCC = predicted_probsPCC > t;
  % mean(predicted_probsPCC)
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
  PCC_vector(3) = PCC;
  
  prior1 = mean([double(yAda);ytrain123]); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC, prior1, prior0);
  AUC_vector(3) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG, prior1, prior0);
  PGini_vector(3) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH, prior1, prior0);
  Hmeasure_vector(3) = H_measure;
  
  BScore = mean( (predicted_probsBS - Ytest4).^2);
  BScore_vector(3) = BScore;
 
  KS_value = computeKSvalue(Ytest4,predicted_probsKS);
  KSvalue_vector(3) = KS_value;
     
  else
      
  TreePCC = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PCC{2}*size(Xtrain123,1));
  TreeAUC = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_AUC{2}*size(Xtrain123,1));
  TreePG = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PG{2}*size(Xtrain123,1));
  TreeBS = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_BS{2}*size(Xtrain123,1));
  TreeH = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_H{2}*size(Xtrain123,1));
  TreeKS = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_KS{2}*size(Xtrain123,1));
  
  % Construct new probability score  on test data.
   [~, predicted_probsPCC,~,~] = predict(TreePCC,Xtest4);
  predicted_probsPCC = predicted_probsPCC(:,2);
  
  [~, predicted_probsKS,~,~] = predict(TreeKS,Xtest4);
  predicted_probsKS = predicted_probsKS(:,2);
  
  [~, predicted_probsH,~,~] = predict(TreeH,Xtest4);
  predicted_probsH = predicted_probsH(:,2);
  
  [~, predicted_probsPG,~,~] = predict(TreePG,Xtest4);
  predicted_probsPG = predicted_probsPG(:,2);
 
  [~, predicted_probsAUC,~,~] = predict(TreeAUC,Xtest4);
  predicted_probsAUC = predicted_probsAUC(:,2);
  
  [~, predicted_probsBS,~,~] = predict(TreeBS,Xtest4);
  predicted_probsBS = predicted_probsBS(:,2);
  
  t = mean(ytrain123);
  YhatPCC = predicted_probsPCC > t;
  % mean(predicted_probsPCC)
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
  PCC_vector(3) = PCC;
  
  prior1 = mean(ytrain123); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC, prior1, prior0);
  AUC_vector(3) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG, prior1, prior0);
  PGini_vector(3) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH, prior1, prior0);
  Hmeasure_vector(3) = H_measure;
    
  BScore = mean( (predicted_probsBS - Ytest4).^2);
  BScore_vector(3) = BScore;
 
  KS_value = computeKSvalue(Ytest4,predicted_probsKS);
  KSvalue_vector(3) = KS_value;
  end
  
 
  %%
  % Repeat the above process but with the third fold of the four fold
  % cross validation.
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
 
 % Determine the optimal regularization parameter lambda (to prevent overfitting of the neural networks)
  % with k fold cross validation. 
  % k = 5 on the training data.
  [Prune_and_MinLeafSize_optimal_PCC, Prune_and_MinLeafSize_optimal_AUC, Prune_and_MinLeafSize_optimal_PG, Prune_and_MinLeafSize_optimal_BS,...
          Prune_and_MinLeafSize_optimal_H, Prune_and_MinLeafSize_optimal_KS] =  optimizeDT(Xtrain123,ytrain123, useAdaSyn);
  
  if useAdaSyn == 1
  
  [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
  
  TreePCC = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_PCC{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeAUC = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', round( Prune_and_MinLeafSize_optimal_AUC{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreePG = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_PG{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeBS = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_BS{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeH = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize',     round( Prune_and_MinLeafSize_optimal_H{2}*size([XAdaSyn;Xtrain123],1) ) );
  TreeKS = fitctree([XAdaSyn;Xtrain123],[double(yAda);ytrain123],'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize',   round( Prune_and_MinLeafSize_optimal_KS{2}*size([XAdaSyn;Xtrain123],1) ) );
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  [~, predicted_probsPCC,~,~] = predict(TreePCC,Xtest4);
  predicted_probsPCC = predicted_probsPCC(:,2);
  
  [~, predicted_probsKS,~,~] = predict(TreeKS,Xtest4);
  predicted_probsKS = predicted_probsKS(:,2);
  
  [~, predicted_probsH,~,~] = predict(TreeH,Xtest4);
  predicted_probsH = predicted_probsH(:,2);
  
  [~, predicted_probsPG,~,~] = predict(TreePG,Xtest4);
  predicted_probsPG = predicted_probsPG(:,2);
 
  [~, predicted_probsAUC,~,~] = predict(TreeAUC,Xtest4);
  predicted_probsAUC = predicted_probsAUC(:,2);
  
  [~, predicted_probsBS,~,~] = predict(TreeBS,Xtest4);
  predicted_probsBS = predicted_probsBS(:,2);
  
  
   t = mean([double(yAda);ytrain123]);
   YhatPCC = predicted_probsPCC > t;
  % mean(predicted_probsPCC)
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
  PCC_vector(4) = PCC;
  
  prior1 = mean([double(yAda);ytrain123]); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC, prior1, prior0);
  AUC_vector(4) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG, prior1, prior0);
  PGini_vector(4) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH, prior1, prior0);
  Hmeasure_vector(4) = H_measure;
  
  BScore = mean( (predicted_probsBS - Ytest4).^2);
  BScore_vector(4) = BScore;
 
  KS_value = computeKSvalue(Ytest4,predicted_probsKS);
  KSvalue_vector(4) = KS_value;
     
  else
      
  TreePCC = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_PCC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PCC{2}*size(Xtrain123,1));
  TreeAUC = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_AUC{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_AUC{2}*size(Xtrain123,1));
  TreePG = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_PG{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_PG{2}*size(Xtrain123,1));
  TreeBS = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_BS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_BS{2}*size(Xtrain123,1));
  TreeH = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_H{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_H{2}*size(Xtrain123,1));
  TreeKS = fitctree(Xtrain123,ytrain123,'Prune', Prune_and_MinLeafSize_optimal_KS{1} ,'MinLeafSize', Prune_and_MinLeafSize_optimal_KS{2}*size(Xtrain123,1));
  
  % Construct new probability score  on test data.
   [~, predicted_probsPCC,~,~] = predict(TreePCC,Xtest4);
  predicted_probsPCC = predicted_probsPCC(:,2);
  
  [~, predicted_probsKS,~,~] = predict(TreeKS,Xtest4);
  predicted_probsKS = predicted_probsKS(:,2);
  
  [~, predicted_probsH,~,~] = predict(TreeH,Xtest4);
  predicted_probsH = predicted_probsH(:,2);
  
  [~, predicted_probsPG,~,~] = predict(TreePG,Xtest4);
  predicted_probsPG = predicted_probsPG(:,2);
 
  [~, predicted_probsAUC,~,~] = predict(TreeAUC,Xtest4);
  predicted_probsAUC = predicted_probsAUC(:,2);
  
  [~, predicted_probsBS,~,~] = predict(TreeBS,Xtest4);
  predicted_probsBS = predicted_probsBS(:,2);
  
  t = mean(ytrain123);
  YhatPCC = predicted_probsPCC > t;
  % mean(predicted_probsPCC)
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Ytest4 == YhatPCC) )/numel(Ytest4);
  PCC_vector(4) = PCC;
  
  prior1 = mean(ytrain123); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsAUC, prior1, prior0);
  AUC_vector(4) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsPG, prior1, prior0);
  PGini_vector(4) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Ytest4, predicted_probsH, prior1, prior0);
  Hmeasure_vector(4) = H_measure;
    
  BScore = mean( (predicted_probsBS - Ytest4).^2);
  BScore_vector(4) = BScore;
 
  KS_value = computeKSvalue(Ytest4,predicted_probsKS);
  KSvalue_vector(4) = KS_value;
  end
%%
PCC = mean(PCC_vector);
AUC = mean(AUC_vector);
PG  = mean(PGini_vector);
BS  = mean(BScore_vector);
H   = mean(Hmeasure_vector);
KS  = mean(KSvalue_vector);

end
