function [PCC, AUC, PG,BS, KS, H] = FNNnew(X1,X2,X3,X4,y1,y2,y3,y4)
%% This code uses feedforward neural networks (estimated with backpropagation and mini-batch stochastic gradient descent) to make classifications for the test sample
% we make use of 4 fold cross validation (fold for each quarter of each
% year). Also, we will use 5 fold cross validation to fine tune the
% hyperparameters 

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
  useRegularization = 1;
  
  [lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
          lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS] =  optimizeFNNRevised(Xtrain123,ytrain123,useRegularization, useAdaSyn);
  
  % Now fit FNNs with the optimal hyperparameter(s) for each performance
  % measure OLD VERSION: see below for the REVISED way to use FNN.
  %[Wvector1PCC,Wvector2PCC,bvec1PCC ,bvec2PCC,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_PCC(1), lambda_and_hidNodes_optimal_PCC(2));
  %[WvectorKS,Wvector2KS,bvec1KS ,bvec2KS,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_KS(1), lambda_and_hidNodes_optimal_KS(2));
  %[Wvector1AUC,Wvector2AUC,bvec1AUC ,bvec2AUC,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_AUC(1), lambda_and_hidNodes_optimal_AUC(2));
  %[Wvector1PG,Wvector2PG,bvec1PG ,bvec2PG,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_PG(1), lambda_and_hidNodes_optimal_PG(2));
  %[Wvector1H,Wvector2H,bvec1H ,bvec2H,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_H(1), lambda_and_hidNodes_optimal_H(2));
  %[Wvector1BS,Wvector2BS,bvec1BS ,bvec2BS,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_BS(1), lambda_and_hidNodes_optimal_BS(2));
  
  
  netPCC = patternnet(lambda_and_hidNodes_optimal_PCC(2));
  netKS = patternnet(lambda_and_hidNodes_optimal_KS(2));
  netAUC = patternnet(lambda_and_hidNodes_optimal_AUC(2));
  netPG = patternnet(lambda_and_hidNodes_optimal_PG(2));
  netH = patternnet(lambda_and_hidNodes_optimal_H(2));
  netBS = patternnet(lambda_and_hidNodes_optimal_BS(2));
      
  % We note that the default loss function used in training the FNN is
  % the cross entropy and not the MSE.
  netPCC.divideFcn = 'divideblock'; % this function divides the training set into a training set and validation set.
  netPCC.divideParam.trainRatio = 85/100;
  netPCC.divideParam.valRatio = 15/100;
  netPCC.divideParam.testRatio = 0/100;
  netPCC.trainParam.showWindow = 0;
  netPCC.performParam.regularization = lambda_and_hidNodes_optimal_PCC(1); % set regularization parameter to prevent overfitting.
  
  netKS.divideFcn = 'divideblock'; 
  netKS.divideParam.trainRatio = 85/100;
  netKS.divideParam.valRatio = 15/100;
  netKS.divideParam.testRatio = 0/100;
  netKS.trainParam.showWindow = 0;
  netKS.performParam.regularization = lambda_and_hidNodes_optimal_KS(1); 
 
  netAUC.divideFcn = 'divideblock'; 
  netAUC.divideParam.trainRatio = 85/100;
  netAUC.divideParam.valRatio = 15/100;
  netAUC.divideParam.testRatio = 0/100;
  netAUC.trainParam.showWindow = 0;
  netAUC.performParam.regularization = lambda_and_hidNodes_optimal_AUC(1); 
  
  netPG.divideFcn = 'divideblock'; 
  netPG.divideParam.trainRatio = 85/100;
  netPG.divideParam.valRatio = 15/100;
  netPG.divideParam.testRatio = 0/100;
  netPG.trainParam.showWindow = 0;
  netPG.performParam.regularization = lambda_and_hidNodes_optimal_PG(1);

  netH.divideFcn = 'divideblock';
  netH.divideParam.trainRatio = 85/100;
  netH.divideParam.valRatio = 15/100;
  netH.divideParam.testRatio = 0/100;
  netH.trainParam.showWindow = 0;
  netH.performParam.regularization = lambda_and_hidNodes_optimal_H(1); 

  netBS.divideFcn = 'divideblock'; 
  netBS.divideParam.trainRatio = 85/100;
  netBS.divideParam.valRatio = 15/100;
  netBS.divideParam.testRatio = 0/100;
  netBS.trainParam.showWindow = 0;
  netBS.performParam.regularization = lambda_and_hidNodes_optimal_BS(1); 


  % Train the models where we split the training set Xtrain123 in to a train and
  % validation set which has the purpose to determine the appropriate
  % early stopping. Input [Xtrain123 XaDaSyn  is 100xN1 matrix and input y is 2XN1 (one row
  % for each class label, here first is defaulted loans, second is
  % non-defaulted loans).
  if useAdaSyn == 1
  [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
  
  [netPCC,~] = train(netPCC,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netKS,~] = train(netKS,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netAUC,~] = train(netAUC,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netPG,~] = train(netPG,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netH,~] = train(netH, [XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netBS,~] = train(netBS, [XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  predicted_probsPCC = netPCC(Xtest4');
  predicted_probsPCC = predicted_probsPCC(1,:)';
  
  predicted_probsKS = netKS(Xtest4');
  predicted_probsKS = predicted_probsKS(1,:)';

  predicted_probsH = netH(Xtest4');
  predicted_probsH = predicted_probsH(1,:)';
  
  predicted_probsPG = netPG(Xtest4');
  predicted_probsPG = predicted_probsPG(1,:)';
 
  predicted_probsAUC = netAUC(Xtest4');
  predicted_probsAUC = predicted_probsAUC(1,:)';
  
  predicted_probsBS = netBS(Xtest4');
  predicted_probsBS = predicted_probsBS(1,:)';
  
  
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
  [netPCC,~] = train(netPCC,Xtrain123', ytrain123');
  [netKS,~] = train(netKS,Xtrain123', ytrain123');
  [netAUC,~] = train(netAUC,Xtrain123', ytrain123');
  [netPG,~] = train(netPG,Xtrain123', ytrain123');
  [netH,~] = train(netH,Xtrain123', ytrain123');
  [netBS,~] = train(netBS,Xtrain123', ytrain123'); 
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  predicted_probsPCC = netPCC(Xtest4');
  predicted_probsPCC = predicted_probsPCC(1,:)';
  
  predicted_probsKS = netKS(Xtest4');
  predicted_probsKS = predicted_probsKS(1,:)';

  predicted_probsH = netH(Xtest4');
  predicted_probsH = predicted_probsH(1,:)';
  
  predicted_probsPG = netPG(Xtest4');
  predicted_probsPG = predicted_probsPG(1,:)';
 
  predicted_probsAUC = netAUC(Xtest4');
  predicted_probsAUC = predicted_probsAUC(1,:)';
  
  predicted_probsBS = netBS(Xtest4');
  predicted_probsBS = predicted_probsBS(1,:)';
  
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
   
 %  for i = 1:size( X2,1 )
 %   [yhat,~,~,~] = Feedforward(Wvector1PCC,Wvector2PCC,bvec1PCC ,bvec2PCC,X2(i,:)' );
 %   predicted_probsPCC(i) = yhat;
 %    [yhat,~,~,~] = Feedforward(WvectorKS,Wvector2KS,bvec1KS ,bvec2KS,X2(i,:)' );
 %   predicted_probsKS(i) = yhat;
 %    [yhat,~,~,~] = Feedforward(Wvector1AUC,Wvector2AUC,bvec1AUC ,bvec2AUC,X2(i,:)' );
 %   predicted_probsAUC(i) = yhat;
 %    [yhat,~,~,~] = Feedforward(Wvector1PG,Wvector2PG,bvec1PG ,bvec2PG,X2(i,:)' );
 %   predicted_probsPG(i) = yhat;
 %    [yhat,~,~,~] = Feedforward(Wvector1H,Wvector2H,bvec1H ,bvec2H,X2(i,:)' );
 %   predicted_probsH(i) = yhat;
 %    [yhat,~,~,~] = Feedforward(Wvector1BS,Wvector2BS,bvec1BS ,bvec2BS,X2(i,:)' );
 %   predicted_probsBS(i) = yhat;
 %  end
  
 
  
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
  useRegularization = 1;
  
  [lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
          lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS] =  optimizeFNNRevised(Xtrain123,ytrain123,useRegularization, useAdaSyn);
  
  % Now fit FNNs with the optimal hyperparameter(s) for each performance
  % measure OLD VERSION: see below for the REVISED way to use FNN.
  %[Wvector1PCC,Wvector2PCC,bvec1PCC ,bvec2PCC,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_PCC(1), lambda_and_hidNodes_optimal_PCC(2));
  %[WvectorKS,Wvector2KS,bvec1KS ,bvec2KS,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_KS(1), lambda_and_hidNodes_optimal_KS(2));
  %[Wvector1AUC,Wvector2AUC,bvec1AUC ,bvec2AUC,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_AUC(1), lambda_and_hidNodes_optimal_AUC(2));
  %[Wvector1PG,Wvector2PG,bvec1PG ,bvec2PG,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_PG(1), lambda_and_hidNodes_optimal_PG(2));
  %[Wvector1H,Wvector2H,bvec1H ,bvec2H,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_H(1), lambda_and_hidNodes_optimal_H(2));
  %[Wvector1BS,Wvector2BS,bvec1BS ,bvec2BS,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_BS(1), lambda_and_hidNodes_optimal_BS(2));
  
  
  netPCC = patternnet(lambda_and_hidNodes_optimal_PCC(2));
  netKS = patternnet(lambda_and_hidNodes_optimal_KS(2));
  netAUC = patternnet(lambda_and_hidNodes_optimal_AUC(2));
  netPG = patternnet(lambda_and_hidNodes_optimal_PG(2));
  netH = patternnet(lambda_and_hidNodes_optimal_H(2));
  netBS = patternnet(lambda_and_hidNodes_optimal_BS(2));
      
  % We note that the default loss function used in training the FNN is
  % the cross entropy and not the MSE.
  netPCC.divideFcn = 'divideblock'; % this function divides the training set into a training set and validation set.
  netPCC.divideParam.trainRatio = 85/100;
  netPCC.divideParam.valRatio = 15/100;
  netPCC.divideParam.testRatio = 0/100;
  netPCC.trainParam.showWindow = 0;
  netPCC.performParam.regularization = lambda_and_hidNodes_optimal_PCC(1); % set regularization parameter to prevent overfitting.
  
  netKS.divideFcn = 'divideblock'; 
  netKS.divideParam.trainRatio = 85/100;
  netKS.divideParam.valRatio = 15/100;
  netKS.divideParam.testRatio = 0/100;
  netKS.trainParam.showWindow = 0;
  netKS.performParam.regularization = lambda_and_hidNodes_optimal_KS(1); 
 
  netAUC.divideFcn = 'divideblock'; 
  netAUC.divideParam.trainRatio = 85/100;
  netAUC.divideParam.valRatio = 15/100;
  netAUC.divideParam.testRatio = 0/100;
  netAUC.trainParam.showWindow = 0;
  netAUC.performParam.regularization = lambda_and_hidNodes_optimal_AUC(1); 
  
  netPG.divideFcn = 'divideblock'; 
  netPG.divideParam.trainRatio = 85/100;
  netPG.divideParam.valRatio = 15/100;
  netPG.divideParam.testRatio = 0/100;
  netPG.trainParam.showWindow = 0;
  netPG.performParam.regularization = lambda_and_hidNodes_optimal_PG(1);

  netH.divideFcn = 'divideblock';
  netH.divideParam.trainRatio = 85/100;
  netH.divideParam.valRatio = 15/100;
  netH.divideParam.testRatio = 0/100;
  netH.trainParam.showWindow = 0;
  netH.performParam.regularization = lambda_and_hidNodes_optimal_H(1); 

  netBS.divideFcn = 'divideblock'; 
  netBS.divideParam.trainRatio = 85/100;
  netBS.divideParam.valRatio = 15/100;
  netBS.divideParam.testRatio = 0/100;
  netBS.trainParam.showWindow = 0;
  netBS.performParam.regularization = lambda_and_hidNodes_optimal_BS(1); 


  % Train the models where we split the training set Xtrain123 in to a train and
  % validation set which has the purpose to determine the appropriate
  % early stopping. Input [Xtrain123 XaDaSyn  is 100xN1 matrix and input y is 2XN1 (one row
  % for each class label, here first is defaulted loans, second is
  % non-defaulted loans).
  if useAdaSyn == 1
  [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
  
  [netPCC,~] = train(netPCC,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netKS,~] = train(netKS,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netAUC,~] = train(netAUC,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netPG,~] = train(netPG,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netH,~] = train(netH, [XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netBS,~] = train(netBS, [XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  predicted_probsPCC = netPCC(Xtest4');
  predicted_probsPCC = predicted_probsPCC(1,:)';
  
  predicted_probsKS = netKS(Xtest4');
  predicted_probsKS = predicted_probsKS(1,:)';

  predicted_probsH = netH(Xtest4');
  predicted_probsH = predicted_probsH(1,:)';
  
  predicted_probsPG = netPG(Xtest4');
  predicted_probsPG = predicted_probsPG(1,:)';
 
  predicted_probsAUC = netAUC(Xtest4');
  predicted_probsAUC = predicted_probsAUC(1,:)';
  
  predicted_probsBS = netBS(Xtest4');
  predicted_probsBS = predicted_probsBS(1,:)';
  
  
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
  [netPCC,~] = train(netPCC,Xtrain123', ytrain123');
  [netKS,~] = train(netKS,Xtrain123', ytrain123');
  [netAUC,~] = train(netAUC,Xtrain123', ytrain123');
  [netPG,~] = train(netPG,Xtrain123', ytrain123');
  [netH,~] = train(netH,Xtrain123', ytrain123');
  [netBS,~] = train(netBS,Xtrain123', ytrain123'); 
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  predicted_probsPCC = netPCC(Xtest4');
  predicted_probsPCC = predicted_probsPCC(1,:)';
  
  predicted_probsKS = netKS(Xtest4');
  predicted_probsKS = predicted_probsKS(1,:)';

  predicted_probsH = netH(Xtest4');
  predicted_probsH = predicted_probsH(1,:)';
  
  predicted_probsPG = netPG(Xtest4');
  predicted_probsPG = predicted_probsPG(1,:)';
 
  predicted_probsAUC = netAUC(Xtest4');
  predicted_probsAUC = predicted_probsAUC(1,:)';
  
  predicted_probsBS = netBS(Xtest4');
  predicted_probsBS = predicted_probsBS(1,:)';
  
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
  useRegularization = 1;
  
  [lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
          lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS] =  optimizeFNNRevised(Xtrain123,ytrain123,useRegularization, useAdaSyn);
  
  % Now fit FNNs with the optimal hyperparameter(s) for each performance
  % measure OLD VERSION: see below for the REVISED way to use FNN.
  %[Wvector1PCC,Wvector2PCC,bvec1PCC ,bvec2PCC,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_PCC(1), lambda_and_hidNodes_optimal_PCC(2));
  %[WvectorKS,Wvector2KS,bvec1KS ,bvec2KS,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_KS(1), lambda_and_hidNodes_optimal_KS(2));
  %[Wvector1AUC,Wvector2AUC,bvec1AUC ,bvec2AUC,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_AUC(1), lambda_and_hidNodes_optimal_AUC(2));
  %[Wvector1PG,Wvector2PG,bvec1PG ,bvec2PG,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_PG(1), lambda_and_hidNodes_optimal_PG(2));
  %[Wvector1H,Wvector2H,bvec1H ,bvec2H,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_H(1), lambda_and_hidNodes_optimal_H(2));
  %[Wvector1BS,Wvector2BS,bvec1BS ,bvec2BS,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_BS(1), lambda_and_hidNodes_optimal_BS(2));
  
  
  netPCC = patternnet(lambda_and_hidNodes_optimal_PCC(2));
  netKS = patternnet(lambda_and_hidNodes_optimal_KS(2));
  netAUC = patternnet(lambda_and_hidNodes_optimal_AUC(2));
  netPG = patternnet(lambda_and_hidNodes_optimal_PG(2));
  netH = patternnet(lambda_and_hidNodes_optimal_H(2));
  netBS = patternnet(lambda_and_hidNodes_optimal_BS(2));
      
  % We note that the default loss function used in training the FNN is
  % the cross entropy and not the MSE.
  netPCC.divideFcn = 'divideblock'; % this function divides the training set into a training set and validation set.
  netPCC.divideParam.trainRatio = 85/100;
  netPCC.divideParam.valRatio = 15/100;
  netPCC.divideParam.testRatio = 0/100;
  netPCC.trainParam.showWindow = 0;
  netPCC.performParam.regularization = lambda_and_hidNodes_optimal_PCC(1); % set regularization parameter to prevent overfitting.
  
  netKS.divideFcn = 'divideblock'; 
  netKS.divideParam.trainRatio = 85/100;
  netKS.divideParam.valRatio = 15/100;
  netKS.divideParam.testRatio = 0/100;
  netKS.trainParam.showWindow = 0;
  netKS.performParam.regularization = lambda_and_hidNodes_optimal_KS(1); 
 
  netAUC.divideFcn = 'divideblock'; 
  netAUC.divideParam.trainRatio = 85/100;
  netAUC.divideParam.valRatio = 15/100;
  netAUC.divideParam.testRatio = 0/100;
  netAUC.trainParam.showWindow = 0;
  netAUC.performParam.regularization = lambda_and_hidNodes_optimal_AUC(1); 
  
  netPG.divideFcn = 'divideblock'; 
  netPG.divideParam.trainRatio = 85/100;
  netPG.divideParam.valRatio = 15/100;
  netPG.divideParam.testRatio = 0/100;
  netPG.trainParam.showWindow = 0;
  netPG.performParam.regularization = lambda_and_hidNodes_optimal_PG(1);

  netH.divideFcn = 'divideblock';
  netH.divideParam.trainRatio = 85/100;
  netH.divideParam.valRatio = 15/100;
  netH.divideParam.testRatio = 0/100;
  netH.trainParam.showWindow = 0;
  netH.performParam.regularization = lambda_and_hidNodes_optimal_H(1); 

  netBS.divideFcn = 'divideblock'; 
  netBS.divideParam.trainRatio = 85/100;
  netBS.divideParam.valRatio = 15/100;
  netBS.divideParam.testRatio = 0/100;
  netBS.trainParam.showWindow = 0;
  netBS.performParam.regularization = lambda_and_hidNodes_optimal_BS(1); 


  % Train the models where we split the training set Xtrain123 in to a train and
  % validation set which has the purpose to determine the appropriate
  % early stopping. Input [Xtrain123 XaDaSyn  is 100xN1 matrix and input y is 2XN1 (one row
  % for each class label, here first is defaulted loans, second is
  % non-defaulted loans).
  if useAdaSyn == 1
  [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
  
  [netPCC,~] = train(netPCC,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netKS,~] = train(netKS,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netAUC,~] = train(netAUC,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netPG,~] = train(netPG,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netH,~] = train(netH, [XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netBS,~] = train(netBS, [XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  predicted_probsPCC = netPCC(Xtest4');
  predicted_probsPCC = predicted_probsPCC(1,:)';
  
  predicted_probsKS = netKS(Xtest4');
  predicted_probsKS = predicted_probsKS(1,:)';

  predicted_probsH = netH(Xtest4');
  predicted_probsH = predicted_probsH(1,:)';
  
  predicted_probsPG = netPG(Xtest4');
  predicted_probsPG = predicted_probsPG(1,:)';
 
  predicted_probsAUC = netAUC(Xtest4');
  predicted_probsAUC = predicted_probsAUC(1,:)';
  
  predicted_probsBS = netBS(Xtest4');
  predicted_probsBS = predicted_probsBS(1,:)';
  
  
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
  [netPCC,~] = train(netPCC,Xtrain123', ytrain123');
  [netKS,~] = train(netKS,Xtrain123', ytrain123');
  [netAUC,~] = train(netAUC,Xtrain123', ytrain123');
  [netPG,~] = train(netPG,Xtrain123', ytrain123');
  [netH,~] = train(netH,Xtrain123', ytrain123');
  [netBS,~] = train(netBS,Xtrain123', ytrain123'); 
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  predicted_probsPCC = netPCC(Xtest4');
  predicted_probsPCC = predicted_probsPCC(1,:)';
  
  predicted_probsKS = netKS(Xtest4');
  predicted_probsKS = predicted_probsKS(1,:)';

  predicted_probsH = netH(Xtest4');
  predicted_probsH = predicted_probsH(1,:)';
  
  predicted_probsPG = netPG(Xtest4');
  predicted_probsPG = predicted_probsPG(1,:)';
 
  predicted_probsAUC = netAUC(Xtest4');
  predicted_probsAUC = predicted_probsAUC(1,:)';
  
  predicted_probsBS = netBS(Xtest4');
  predicted_probsBS = predicted_probsBS(1,:)';
  
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
  useRegularization = 1;
  
  [lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
          lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS] =  optimizeFNNRevised(Xtrain123,ytrain123,useRegularization, useAdaSyn);
  
  % Now fit FNNs with the optimal hyperparameter(s) for each performance
  % measure OLD VERSION: see below for the REVISED way to use FNN.
  %[Wvector1PCC,Wvector2PCC,bvec1PCC ,bvec2PCC,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_PCC(1), lambda_and_hidNodes_optimal_PCC(2));
  %[WvectorKS,Wvector2KS,bvec1KS ,bvec2KS,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_KS(1), lambda_and_hidNodes_optimal_KS(2));
  %[Wvector1AUC,Wvector2AUC,bvec1AUC ,bvec2AUC,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_AUC(1), lambda_and_hidNodes_optimal_AUC(2));
  %[Wvector1PG,Wvector2PG,bvec1PG ,bvec2PG,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_PG(1), lambda_and_hidNodes_optimal_PG(2));
  %[Wvector1H,Wvector2H,bvec1H ,bvec2H,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_H(1), lambda_and_hidNodes_optimal_H(2));
  %[Wvector1BS,Wvector2BS,bvec1BS ,bvec2BS,~] = MLP(X1,Y1, useRegularization,lambda_and_hidNodes_optimal_BS(1), lambda_and_hidNodes_optimal_BS(2));
  
  
  netPCC = patternnet(lambda_and_hidNodes_optimal_PCC(2));
  netKS = patternnet(lambda_and_hidNodes_optimal_KS(2));
  netAUC = patternnet(lambda_and_hidNodes_optimal_AUC(2));
  netPG = patternnet(lambda_and_hidNodes_optimal_PG(2));
  netH = patternnet(lambda_and_hidNodes_optimal_H(2));
  netBS = patternnet(lambda_and_hidNodes_optimal_BS(2));
      
  % We note that the default loss function used in training the FNN is
  % the cross entropy and not the MSE.
  netPCC.divideFcn = 'divideblock'; % this function divides the training set into a training set and validation set.
  netPCC.divideParam.trainRatio = 85/100;
  netPCC.divideParam.valRatio = 15/100;
  netPCC.divideParam.testRatio = 0/100;
  netPCC.trainParam.showWindow = 0;
  netPCC.performParam.regularization = lambda_and_hidNodes_optimal_PCC(1); % set regularization parameter to prevent overfitting.
  
  netKS.divideFcn = 'divideblock'; 
  netKS.divideParam.trainRatio = 85/100;
  netKS.divideParam.valRatio = 15/100;
  netKS.divideParam.testRatio = 0/100;
  netKS.trainParam.showWindow = 0;
  netKS.performParam.regularization = lambda_and_hidNodes_optimal_KS(1); 
 
  netAUC.divideFcn = 'divideblock'; 
  netAUC.divideParam.trainRatio = 85/100;
  netAUC.divideParam.valRatio = 15/100;
  netAUC.divideParam.testRatio = 0/100;
  netAUC.trainParam.showWindow = 0;
  netAUC.performParam.regularization = lambda_and_hidNodes_optimal_AUC(1); 
  
  netPG.divideFcn = 'divideblock'; 
  netPG.divideParam.trainRatio = 85/100;
  netPG.divideParam.valRatio = 15/100;
  netPG.divideParam.testRatio = 0/100;
  netPG.trainParam.showWindow = 0;
  netPG.performParam.regularization = lambda_and_hidNodes_optimal_PG(1);

  netH.divideFcn = 'divideblock';
  netH.divideParam.trainRatio = 85/100;
  netH.divideParam.valRatio = 15/100;
  netH.divideParam.testRatio = 0/100;
  netH.trainParam.showWindow = 0;
  netH.performParam.regularization = lambda_and_hidNodes_optimal_H(1); 

  netBS.divideFcn = 'divideblock'; 
  netBS.divideParam.trainRatio = 85/100;
  netBS.divideParam.valRatio = 15/100;
  netBS.divideParam.testRatio = 0/100;
  netBS.trainParam.showWindow = 0;
  netBS.performParam.regularization = lambda_and_hidNodes_optimal_BS(1); 


  % Train the models where we split the training set Xtrain123 in to a train and
  % validation set which has the purpose to determine the appropriate
  % early stopping. Input [Xtrain123 XaDaSyn  is 100xN1 matrix and input y is 2XN1 (one row
  % for each class label, here first is defaulted loans, second is
  % non-defaulted loans).
  if useAdaSyn == 1
  [XAdaSyn, yAda] = ADASYN(Xtrain123, ytrain123, 1, [], [], false);
  
  [netPCC,~] = train(netPCC,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netKS,~] = train(netKS,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netAUC,~] = train(netAUC,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netPG,~] = train(netPG,[XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netH,~] = train(netH, [XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  [netBS,~] = train(netBS, [XAdaSyn;Xtrain123]', [double(yAda);ytrain123]');
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  predicted_probsPCC = netPCC(Xtest4');
  predicted_probsPCC = predicted_probsPCC(1,:)';
  
  predicted_probsKS = netKS(Xtest4');
  predicted_probsKS = predicted_probsKS(1,:)';

  predicted_probsH = netH(Xtest4');
  predicted_probsH = predicted_probsH(1,:)';
  
  predicted_probsPG = netPG(Xtest4');
  predicted_probsPG = predicted_probsPG(1,:)';
 
  predicted_probsAUC = netAUC(Xtest4');
  predicted_probsAUC = predicted_probsAUC(1,:)';
  
  predicted_probsBS = netBS(Xtest4');
  predicted_probsBS = predicted_probsBS(1,:)';
  
  
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
  [netPCC,~] = train(netPCC,Xtrain123', ytrain123');
  [netKS,~] = train(netKS,Xtrain123', ytrain123');
  [netAUC,~] = train(netAUC,Xtrain123', ytrain123');
  [netPG,~] = train(netPG,Xtrain123', ytrain123');
  [netH,~] = train(netH,Xtrain123', ytrain123');
  [netBS,~] = train(netBS,Xtrain123', ytrain123'); 
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  predicted_probsPCC = netPCC(Xtest4');
  predicted_probsPCC = predicted_probsPCC(1,:)';
  
  predicted_probsKS = netKS(Xtest4');
  predicted_probsKS = predicted_probsKS(1,:)';

  predicted_probsH = netH(Xtest4');
  predicted_probsH = predicted_probsH(1,:)';
  
  predicted_probsPG = netPG(Xtest4');
  predicted_probsPG = predicted_probsPG(1,:)';
 
  predicted_probsAUC = netAUC(Xtest4');
  predicted_probsAUC = predicted_probsAUC(1,:)';
  
  predicted_probsBS = netBS(Xtest4');
  predicted_probsBS = predicted_probsBS(1,:)';
  
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