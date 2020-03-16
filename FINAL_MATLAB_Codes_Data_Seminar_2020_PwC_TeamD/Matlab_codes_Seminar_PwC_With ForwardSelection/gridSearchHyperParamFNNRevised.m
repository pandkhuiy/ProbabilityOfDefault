function [lambda_and_hidNodes_optimal_PCC, lambda_and_hidNodes_optimal_AUC, lambda_and_hidNodes_optimal_PG, lambda_and_hidNodes_optimal_BS,...
          lambda_and_hidNodes_optimal_H, lambda_and_hidNodes_optimal_KS] =  gridSearchHyperParamFNNRevised(Xtrain, Ytrain,Xval, Yval, y1)
%% Determine the optimal regularization parameter lambda and number of hidden nodes of the FNN with an exhaustive grid search
% If we use regularization then lambda_and_hidNodes_optimal_PCC,
% lambda_and_hidNodes_optimal_AUC, ..., lambda_and_hidNodes_optimal_KS are
% 2x1 vectors containing the optimal hyperparameters for each measure.
% If we do not use regularization these output variables are 1x1
% scalars containing the optimal number of hidden layer nodes.

lambdas =  [0;0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.99]';
numHiddenNodes = [2;5;10;15;20]; 
Epochs = [50 60 65 70 75];


% This kx6 matrix contains the average performance values computed after k
% fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
PerformanceMeasuresMatrixPCC =  zeros(numel(lambdas),numel(numHiddenNodes), numel(Epochs) );
PerformanceMeasuresMatrixKS =  zeros(numel(lambdas),numel(numHiddenNodes), numel(Epochs));
PerformanceMeasuresMatrixAUC =  zeros(numel(lambdas),numel(numHiddenNodes), numel(Epochs));
PerformanceMeasuresMatrixH =  zeros(numel(lambdas),numel(numHiddenNodes), numel(Epochs));
PerformanceMeasuresMatrixPG =  zeros(numel(lambdas),numel(numHiddenNodes), numel(Epochs));
PerformanceMeasuresMatrixBS =  zeros(numel(lambdas),numel(numHiddenNodes), numel(Epochs));

for e = 1:numel(Epochs)
    e
    for  l = 1:numel(lambdas)
        l 
        for  h = 1:numel(numHiddenNodes)
            %h 

            % This kx6 matrix contains the average performance values computed after k
            % fold cv as PCC, KS ----- AUC, Hmeasure, PG index ----- BS
            % train the FNNand use them for predicting whether the loan corresponding
            % to the features will be granted credit or reject (will default or not).
            % Use these probabilties and predicted probs for computing the six performance measures. 
            % We make use of one hidden layer.
            net = patternnet(numHiddenNodes(h));

            % We note that the default loss function used in training the FNN is
            % the cross entropy and not the MSE.
            %net.divideFcn = 'divideblock'; % this function divides the training set into a training set and validation set.
            net.divideParam.trainRatio = 1;
            net.divideParam.valRatio = 0/100;
            net.divideParam.testRatio = 0/100;
            net.performParam.regularization = lambdas(l); % set regularization parameter to prevent overfitting.
            net.biasConnect(2)= 1;
            net.layers{1}.transferFcn = 'tansig'; %default activation function in hidden layer is tansig function.
            net.layers{2}.transferFcn = 'softmax';
            net.trainFcn = 'trainscg';  % default function is 'trainscg', that is stochastic conjugate gradient, we can change it to 'trainbfg'.
            net.performFcn = 'crossentropy'; % default criterion is 'crossentropy', we can change it to 'mse' to chech which is more suitable.
            net.trainParam.showWindow = 0;
            net.trainParam.epochs = Epochs(e);

            % Train the model where we split the training set in to a train and
            % validation set which has the purpose to determine the appropriate
            % early stopping. Input X is kxN matrix and input y is 2XN (one row
            % for each class label, here first is defaulted loans, second is
            % non-defaulted loans). if useAdaSyn == 1 then we use a enhanced synthetic
            % training set which has 50% percent 1s and 50% 0s (balanced dataset)
            % instead of the orginal data set which is very few 1s and alot of
            % 0s.

            [net,~] = train(net,Xtrain', Ytrain');

            % Construct new probability score (using default softmax activation function) on validation data.
            Prob_scores = net(Xval');
            Prob_scores = Prob_scores(1,:)';

            % function that computes the PCC, requires real y-values, predicted_y
            % values.
            predsort = sort(Prob_scores,'descend'); %sort probabilities
            t = predsort(round(mean(y1)*size(Prob_scores,1)));
            classifications_test = Prob_scores > t;

            tempPCC =  sum( (Yval)  == (classifications_test))/numel(Yval);
            PerformanceMeasuresMatrixPCC(l,h,e) = tempPCC;

            prior1 = mean(y1); 
            prior0 = 1 - prior1;

            [tempAUC,tempPG_index, tempH_measure ] = computeAUC_PGindex_Hvalue(Yval, Prob_scores, prior1, prior0);

            PerformanceMeasuresMatrixAUC(l,h,e) = tempAUC;
            PerformanceMeasuresMatrixH(l,h,e) = tempH_measure;
            PerformanceMeasuresMatrixPG(l,h,e) = tempPG_index;

            [tempKS_value] = computeKSvalue(Yval,Prob_scores);
            PerformanceMeasuresMatrixKS(l,h,e) = tempKS_value;

            tempBScore = mean( (Prob_scores - Yval ).^2);
            PerformanceMeasuresMatrixBS(l,h,e) = tempBScore;

        end
    end
end

lambda_and_hidNodes_optimal_PCC = zeros(1,2);
lambda_and_hidNodes_optimal_KS = zeros(1,2);
lambda_and_hidNodes_optimal_AUC = zeros(1,2);
lambda_and_hidNodes_optimal_PG = zeros(1,2);
lambda_and_hidNodes_optimal_H = zeros(1,2);
lambda_and_hidNodes_optimal_BS = zeros(1,2);

% PCC - KS-  AUC - H -PG -BS in this order.
Perfmatrix = zeros(numel(Epochs),6);

paramMatrixPCC = zeros(numel(Epochs),2);
paramMatrixKS = zeros(numel(Epochs),2);
paramMatrixAUC = zeros(numel(Epochs),2);
paramMatrixH = zeros(numel(Epochs),2);
paramMatrixPG = zeros(numel(Epochs),2);
paramMatrixBS = zeros(numel(Epochs),2);


for e = 1:numel(Epochs)
    % extract the indices of the corresponding optimal parameter C for each
    % measure:
    %PCC
    [MaxRow, ind] = max(PerformanceMeasuresMatrixPCC(:,:,e) ); 
    [PCC, h_index] = max(MaxRow);
    l_index = ind(h_index);
    lambda_and_hidNodes_optimal_PCC(1,1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_PCC(1,2) =  numHiddenNodes(h_index);
    
    paramMatrixPCC(e,:) = lambda_and_hidNodes_optimal_PCC;
    Perfmatrix(e,1) = PCC;
    
    %KS
    [MaxRow, ind] = max(PerformanceMeasuresMatrixKS(:,:,e) ); 
    [KS, h_index] = max(MaxRow);
    l_index = ind(h_index);
    lambda_and_hidNodes_optimal_KS(1,1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_KS(1,2) =  numHiddenNodes(h_index);
    
    paramMatrixKS(e,:) = lambda_and_hidNodes_optimal_KS;
    Perfmatrix(e,2) = KS;
       
    %AUC
    [MaxRow, ind] = max(PerformanceMeasuresMatrixAUC(:,:,e) ); 
    [AUC, h_index] = max(MaxRow);
    l_index = ind(h_index);
    lambda_and_hidNodes_optimal_AUC(1,1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_AUC(1,2) =  numHiddenNodes(h_index);
    
    paramMatrixAUC(e,:) = lambda_and_hidNodes_optimal_AUC;
    Perfmatrix(e,3) = AUC;

    %H measure
    [MaxRow, ind] = max(PerformanceMeasuresMatrixH(:,:,e) ); 
    [H, h_index] = max(MaxRow);
    l_index = ind(h_index);
    lambda_and_hidNodes_optimal_H(1,1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_H(1,2) =  numHiddenNodes(h_index);
    
    paramMatrixH(e,:) = lambda_and_hidNodes_optimal_H;
    Perfmatrix(e,4) = H;

    %PG
    [MaxRow, ind] = max(PerformanceMeasuresMatrixPG(:,:,e) ); 
    [PG, h_index] = max(MaxRow);
    l_index = ind(h_index);
    lambda_and_hidNodes_optimal_PG(1,1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_PG(1,2) =  numHiddenNodes(h_index);
    
    paramMatrixPG(e,:) = lambda_and_hidNodes_optimal_PG;
    Perfmatrix(e,5) = PG;

    %BS
    [MaxRow, ind] = min(PerformanceMeasuresMatrixBS(:,:,e) ); 
    [BS, h_index] = min(MaxRow);
    l_index = ind(h_index);
    lambda_and_hidNodes_optimal_BS(1,1) =  lambdas(l_index);
    lambda_and_hidNodes_optimal_BS(1,2) =  numHiddenNodes(h_index);
    
    paramMatrixBS(e,:) = lambda_and_hidNodes_optimal_BS;
    Perfmatrix(e,6) = BS;
end

% extract the indices of the corresponding optimal number of epochs for each
% measure:
[~, ind] = max(Perfmatrix(:,1)); % PCC
epochs_optimal_PCC =  Epochs(ind);
lambda_and_hidNodes_optimal_PCC = [paramMatrixPCC(ind,:) epochs_optimal_PCC];
[~, ind] = max(Perfmatrix(:,2)); % KS
epochs_optimal_KS =  Epochs(ind);
lambda_and_hidNodes_optimal_KS = [paramMatrixKS(ind,:) epochs_optimal_KS];
[~, ind] = max(Perfmatrix(:,3)); % AUC
epochs_optimal_AUC =  Epochs(ind);
lambda_and_hidNodes_optimal_AUC = [paramMatrixAUC(ind,:) epochs_optimal_AUC];
[~, ind] = max(Perfmatrix(:,4)); % Hmeasure
epochs_optimal_H =  Epochs(ind);
lambda_and_hidNodes_optimal_H = [paramMatrixH(ind,:) epochs_optimal_H];
[~, ind] = max(Perfmatrix(:,5)); % PG_index
epochs_optimal_PG =  Epochs(ind);
lambda_and_hidNodes_optimal_PG = [paramMatrixPG(ind,:) epochs_optimal_PG];
[~, ind] = min(Perfmatrix(:,6)); % BS
epochs_optimal_BS =  Epochs(ind);
lambda_and_hidNodes_optimal_BS = [paramMatrixBS(ind,:) epochs_optimal_BS];

end