function [lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
          lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS] =  optimizeFNNRevised(Xtrain, Ytrain,useRegularization, useAdaSyn)
%% Determine the optimal regularization parameter lambda and number of hidden nodes of the FNN with 5 fold cross validation.
% If we use regularization then lambda_and_hidNodes_optimal_PCC,
% lambda_and_hidNodes_optimal_AUC, ..., lambda_and_hidNodes_optimal_KS are
% 2x1 vectors containing the optimal hyperparameters for each measure.
% If we do not use regularization these output variables are 1x1
% scalars containing the optimal number of hidden layer nodes.
k = 5;

if useRegularization ==1
    lambdas =  [0;0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.99]';
    numHiddenNodes = [2;5;10;15;20]; 

    % This kx6 matrix contains the average performance values computed after k
    % fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
    PerformanceMeasuresMatrixPCC =  zeros(numel(lambdas),numel(numHiddenNodes));
    PerformanceMeasuresMatrixKS =  zeros(numel(lambdas),numel(numHiddenNodes));
    PerformanceMeasuresMatrixAUC =  zeros(numel(lambdas),numel(numHiddenNodes));
    PerformanceMeasuresMatrixH =  zeros(numel(lambdas),numel(numHiddenNodes));
    PerformanceMeasuresMatrixPG =  zeros(numel(lambdas),numel(numHiddenNodes));
    PerformanceMeasuresMatrixBS =  zeros(numel(lambdas),numel(numHiddenNodes));


    for  l = 1:numel(lambdas)
     l 
    for  h = 1:numel(numHiddenNodes)
     %h

    [TrainingIndices, TestIndices] = DoKfoldCrossValid(Ytrain,k);
    %TrainingIndices = logical(indices);  
    %TestIndices = logical(1-indices);  

    % This kx6 matrix contains the average performance values computed after k
    % fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
    kfoldPerformanceMeasuresMatrix =  zeros(k,6);

    for j = 1:k
        %j
      % train the FNNand use them for predicting whether the loan corresponding
      % to the features will be granted credit or reject (will default or not).
      % Use these probabilties and predicted probs for computing the six performance measures. 
      % We make use of one hidden layer.
      net = patternnet(numHiddenNodes(h));
      
      % We note that the default loss function used in training the FNN is
      % the cross entropy and not the MSE.
      net.divideFcn = 'divideblock'; % this function divides the training set into a training set and validation set.
      net.divideParam.trainRatio = 85/100;
      net.divideParam.valRatio = 15/100;
      net.divideParam.testRatio = 0/100;
      net.performParam.regularization = lambdas(l); % set regularization parameter to prevent overfitting.
      net.biasConnect(2)= 0;
      net.layers{1}.transferFcn = 'tansig'; %default activation function in hidden layer is tansig function.
      net.layers{2}.transferFcn = 'softmax';
      net.trainFcn = 'trainscg';  % default function is 'trainscg', that is stochastic conjugate gradient, we can change it to 'trainbfg'.
      net.performFcn = 'crossentropy'; % default criterion is 'crossentropy', we can change it to 'mse' to chech which is more suitable.
      net.trainParam.showWindow = 0;
      
      % Train the model where we split the training set in to a train and
      % validation set which has the purpose to determine the appropriate
      % early stopping. Input X is kxN matrix and input y is 2XN (one row
      % for each class label, here first is defaulted loans, second is
      % non-defaulted loans). if useAdaSyn == 1 then we use a enhanced synthetic
      % training set which has 50% percent 1s and 50% 0s (balanced dataset)
      % instead of the orginal data set which is very few 1s and alot of
      % 0s.
      
      if useAdaSyn == 1
      
      [XAdaSyn, yAda] = ADASYN(Xtrain(TrainingIndices(:,j),:), Ytrain(TrainingIndices(:,j)), 1, [], [], false);    
      
      [net,~] = train(net,[XAdaSyn;Xtrain(TrainingIndices(:,j),:)]', [double(yAda);Ytrain(TrainingIndices(:,j))]' );
      
      % Construct new probability score (using sigmoid) on test data.
      Xtest = Xtrain(TestIndices(:,j),:);
      Prob_scores = net(Xtest');
      Prob_scores = Prob_scores(1,:)';
      
      % function that computes the PCC, requires real y-values, predicted_y
      % values.
      t = mean([double(yAda);Ytrain(TrainingIndices(:,j))]);
      classifications_test = Prob_scores > t;
      
      PCC =  sum( (Ytrain(TestIndices(:,j))  == (classifications_test)) )/numel(Ytrain(TestIndices(:,j)));
      kfoldPerformanceMeasuresMatrix(j,1) = PCC;

      prior1 = mean([double(yAda);Ytrain(TrainingIndices(:,j))]); 
      prior0 = 1 - prior1;

      [AUC,PG_index, H_measure ] = computeAUC_PGindex_Hvalue(Ytrain(TestIndices(:,j)), Prob_scores, prior1, prior0);

      kfoldPerformanceMeasuresMatrix(j,3) = AUC;
      kfoldPerformanceMeasuresMatrix(j,4) = H_measure;
      kfoldPerformanceMeasuresMatrix(j,5) = PG_index;

      [KS_value] = computeKSvalue(Ytrain(TestIndices(:,j)),Prob_scores);

      kfoldPerformanceMeasuresMatrix(j,2) = KS_value;

      BScore = mean( (Prob_scores - Ytrain(TestIndices(:,j)) ).^2);
      kfoldPerformanceMeasuresMatrix(j,6) = BScore;   
          
      else
      [net,~] = train(net,Xtrain(TrainingIndices(:,j),:)', Ytrain(TrainingIndices(:,j))' );
      
      % Construct new probability score (using default softmax activation function) on test data.
      Xtest = Xtrain(TestIndices(:,j),:);
      Prob_scores = net(Xtest');
      Prob_scores = Prob_scores(1,:)';
      
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
    lambda_and_hidNodes_optimal_PCC(1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_PCC(2) =  numHiddenNodes(h_index);

    %KS
    [MaxRow, ind] = max(PerformanceMeasuresMatrixKS); 
    [~, h_index] = max(MaxRow);
    l_index = ind(h_index);
    lambda_and_hidNodes_optimal_KS(1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_KS(2) =  numHiddenNodes(h_index);

    %AUC
    [MaxRow, ind] = max(PerformanceMeasuresMatrixAUC); 
    [~, h_index] = max(MaxRow);
    l_index = ind(h_index);
    lambda_and_hidNodes_optimal_AUC(1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_AUC(2) =  numHiddenNodes(h_index);

    %H measure
    [MaxRow, ind] = max(PerformanceMeasuresMatrixH); 
    [~, h_index] = max(MaxRow);
    l_index = ind(h_index);
    lambda_and_hidNodes_optimal_H(1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_H(2) =  numHiddenNodes(h_index);

    %PG
    [MaxRow, ind] = max(PerformanceMeasuresMatrixPG); 
    [~, h_index] = max(MaxRow);
    l_index = ind(h_index);
    lambda_and_hidNodes_optimal_PG(1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_PG(2) =  numHiddenNodes(h_index);

    %BS
    [MaxRow, ind] = min(PerformanceMeasuresMatrixBS); 
    [~, h_index] = min(MaxRow);
    l_index = ind(h_index);
    lambda_and_hidNodes_optimal_BS(1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_BS(2) =  numHiddenNodes(h_index);

else
    numHiddenNodes = (2:20)';

    % This kx6 matrix contains the average performance values computed after k
    % fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
    PerformanceMeasuresMatrix =  zeros(numel(numHiddenNodes),6);

    for  h = 1:numel(numHiddenNodes)
     h 
    [indices] = DoKfoldCrossValid(Xtrain,k);
    TrainingIndices = logical(indices);  
    TestIndices = logical(1-indices);  

    % This kx6 matrix contains the average performance values computed after k
    % fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
    kfoldPerformanceMeasuresMatrix =  zeros(k,6);

    for j = 1:k
        j
     
      % train the FNNand use them for predicting whether the loan corresponding
      % to the features will be granted credit or reject (will default or not).
      % Use these probabilties and predicted probs for computing the six performance measures. 
      % We make use of one hidden layer.
      net = patternnet(numHiddenNodes(h));
      
      % We note that the default loss function used in training the FNN is
      % the cross entropy and not the MSE.
      net.divideFcn = 'divideblock'; % this function divides the training set into a training set and validation set.
      net.divideParam.trainRatio = 85/100;
      net.divideParam.valRatio = 15/100;
      net.divideParam.testRatio = 0/100;
      net.trainParam.showWindow = 0;
     
      % Train the model where we split the training set in to a train and
      % validation set which has the purpose to determine the appropriate
      % early stopping. Input X is kxN matrix and input y is 2XN (one row
      % for each class label, here first is defaulted loans, second is
      % non-defaulted loans).
      [net,~] = train(net,Xtrain(TrainingIndices(:,j),:)', [Ytrain(TrainingIndices(:,j))'] );
      
      % Construct new probability score (using sigmoid) on test data.
      Xtest = Xtrain(TestIndices(:,j),:);
      Prob_scores = net(Xtest');
      Prob_scores = Prob_scores(1,:)';
      
      
      % function that computes the PCC, requires real y-values, predicted_y
      % values.
      t = mean(Ytrain(TrainingIndices(:,j)));
      classifications_test = Prob_scores > t;
     
      PCC =  sum( (Ytrain(TestIndices(:,j))  == classifications_test) )/numel(Ytrain(TestIndices(:,j)));
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

    PerformanceMeasuresMatrix(h,:) = mean(kfoldPerformanceMeasuresMatrix);


    end

    % extract the indices of the corresponding optimal parameter C for each
    % measure:
    [~, ind] = max(PerformanceMeasuresMatrix(:,1)); % PCC
    lambda_and_hidNodes_optimal_PCC =  numHiddenNodes(ind);
    [~, ind] = max(PerformanceMeasuresMatrix(:,2)); % KS
    lambda_and_hidNodes_optimal_KS =  numHiddenNodes(ind);
    [~, ind] = max(PerformanceMeasuresMatrix(:,3)); % AUC
    lambda_and_hidNodes_optimal_AUC = numHiddenNodes(ind);
    [~, ind] = max(PerformanceMeasuresMatrix(:,4)); % Hmeasure
    lambda_and_hidNodes_optimal_H = numHiddenNodes(ind);
    [~, ind] = max(PerformanceMeasuresMatrix(:,5)); % PG_index
    lambda_and_hidNodes_optimal_PG =  numHiddenNodes(ind);
    [~, ind] = min(PerformanceMeasuresMatrix(:,6)); % BS
    lambda_and_hidNodes_optimal_BS =  numHiddenNodes(ind);   
     
end
end