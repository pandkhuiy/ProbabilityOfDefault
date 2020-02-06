function [PCC, AUC, PG,BS, KS, H] = FNN(X1,X2,X3,X4,y1,y2,y3,y4)
%% This code uses feedforward neural networks (estimated with backpropagation and mini-batch stochastic gradient descent) to make classifications for the test sample
% we make use of 4 fold cross validation (fold for each quarter of each
% year). Also, we will use 5 fold cross validation to fine tune the
% hyperparameter

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


 for  l = 1:P
  l
     
  k = randperm(size(X,1));
 
  % X1 X2 Y1 and Y2 are training and test sets that can interchanged for
  % our 2 fold cross validation for each outerloop of index l.
  X1 = X(k(1:size(X,1)/2),:);
  X2 = X(k( ((size(X,1)/2)+1): end), :);
 
  Y1 = y(k(1:size(X,1)/2));
  Y2 = y(k(((size(X,1)/2)+1): end));

  
  % Start with X1 and Y1 as training sets, and X2 and Y2 as test sets.
  
  % Determine the optimal regularization parameter lambda (to prevent overfitting of the neural networks)
  % with k fold cross validation. 
  % k = 5 on the training data.
  useRegularization = 1;
  [lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
          lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS] =  optimizeFNNRevised(X1,Y1,useRegularization );
  
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


  % Train the models where we split the training set in to a train and
  % validation set which has the purpose to determine the appropriate
  % early stopping. Input X1 is kxN1 matrix and input y is 2XN1 (one row
  % for each class label, here first is defaulted loans, second is
  % non-defaulted loans).
  [netPCC,~] = train(netPCC,X1', [Y1']);
  [netKS,~] = train(netKS,X1', [Y1']);
  [netAUC,~] = train(netAUC,X1', [Y1']);
  [netPG,~] = train(netPG,X1', [Y1']);
  [netH,~] = train(netH,X1', [Y1']);
  [netBS,~] = train(netBS,X1', [Y1']);
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  predicted_probsPCC = netPCC(X2');
  predicted_probsPCC = predicted_probsPCC(1,:)';
  
  predicted_probsKS = netKS(X2');
  predicted_probsKS = predicted_probsKS(1,:)';

  predicted_probsH = netH(X2');
  predicted_probsH = predicted_probsH(1,:)';
  
  predicted_probsPG = netPG(X2');
  predicted_probsPG = predicted_probsPG(1,:)';
 
  predicted_probsAUC = netAUC(X2');
  predicted_probsAUC = predicted_probsAUC(1,:)';
  
  predicted_probsBS = netBS(X2');
  predicted_probsBS = predicted_probsBS(1,:)';

   
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
  
   t = mean(Y1);
   YhatPCC = predicted_probsPCC > t;
   mean(predicted_probsPCC)
  % function that computes the PCC, requires real y-values, predicted_y
  % values.
  PCC =  sum( (Y2 == YhatPCC) )/numel(Y2);
  PCC_vector(l) = PCC;
  
  prior1 = mean(Y1); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Y2, predicted_probsAUC, prior1, prior0);
  AUC_vector(l) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Y2, predicted_probsPG, prior1, prior0);
  PGini_vector(l) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Y2, predicted_probsH, prior1, prior0);
  Hmeasure_vector(l) = H_measure;
  
  
  BScore = mean( (predicted_probsBS - Y2).^2);
  BScore_vector(l) = BScore;
 
  KS_value = computeKSvalue(Y2,predicted_probsKS);
  KSvalue_vector(l) = KS_value;
  
  %%
  % Reverse the roles: use X2 and Y2 as training sets, and X1 and Y1 as test sets.  
  % Start with X2 and Y2 as training sets, and X1 and Y1 as test sets.
  
 
   % Determine the optimal regularization parameter lambda (to prevent overfitting of the neural networks)
  % with k fold cross validation. 
  % k = 5 on the training data.
  useRegularization = 1;
  [lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
          lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS] =  optimizeFNNRevised(X2,Y2,useRegularization );
  
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


  % Train the models where we split the training set in to a train and
  % validation set which has the purpose to determine the appropriate
  % early stopping. Input X2 is kxN2 matrix and input Y2 is 2XN2 (one row
  % for each class label, here first is defaulted loans, second is
  % non-defaulted loans).
  [netPCC,~] = train(netPCC,X2', [Y2']);
  [netKS,~] = train(netKS,X2', [Y2']);
  [netAUC,~] = train(netAUC,X2', [Y2']);
  [netPG,~] = train(netPG,X2', [Y2']);
  [netH,~] = train(netH,X2', [Y2']);
  [netBS,~] = train(netBS,X2', [Y2']);
  
  % Construct new probability score (using default softmax activation function in the output node) on test data.
  predicted_probsPCC = netPCC(X1');
  predicted_probsPCC = predicted_probsPCC(1,:)';
  
  predicted_probsKS = netKS(X1');
  predicted_probsKS = predicted_probsKS(1,:)';

  predicted_probsH = netH(X1');
  predicted_probsH = predicted_probsH(1,:)';
  
  predicted_probsPG = netPG(X1');
  predicted_probsPG = predicted_probsPG(1,:)';
 
  predicted_probsAUC = netAUC(X1');
  predicted_probsAUC = predicted_probsAUC(1,:)';
  
  predicted_probsBS = netBS(X1');
  predicted_probsBS = predicted_probsBS(1,:)';
  
  t = mean(Y2);
  YhatPCC = predicted_probsPCC > t;
  
  
  % We computes the PCC, and require real y-values and predicted_y
  % values.
  PCC =  sum( (Y1 == YhatPCC) )/numel(Y1);
  PCC_vector(l+3) = PCC;
  
  prior1 = mean(Y2); prior0 = 1 - prior1;
  
  [AUC,~, ~ ] = computeAUC_PGindex_Hvalue(Y1, predicted_probsAUC, prior1, prior0);
  AUC_vector(l+3) = AUC;
  
  [~,PG_index, ~ ] = computeAUC_PGindex_Hvalue(Y1, predicted_probsPG, prior1, prior0);
  PGini_vector(l+3) = PG_index;
  
  [~,~, H_measure ] = computeAUC_PGindex_Hvalue(Y1, predicted_probsH, prior1, prior0);
  Hmeasure_vector(l+3) = H_measure;
  
  
  BScore = mean( (predicted_probsBS - Y1).^2);
  BScore_vector(l+3) = BScore;
 
  KS_value = computeKSvalue(Y1,predicted_probsKS);
  KSvalue_vector(l+3) = KS_value;
  
 end
 
PCC = mean(PCC_vector);
AUC = mean(AUC_vector);
PG  = mean(PGini_vector);
BS  = mean(BScore_vector);
H   = mean(Hmeasure_vector);
KS  = mean(KSvalue_vector);

end