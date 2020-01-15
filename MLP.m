function [Wvector1,Wvector2,bvec1 ,bvec2] = MLP(X,y)
%% This code implements the MLP neural network.
 
rng('default');
rng(1);
% X is has N observations and k variables.
[N,k] = size(X);

% initialization
numHiddenNodes = 2;
numInputs = k;
numOutputs = 1;
Wvector1 = rand(numHiddenNodes,k);
Wvector2 = rand(numOutputs,numHiddenNodes);
bvec1 = rand(numHiddenNodes,1);
bvec2 = rand(numOutputs,1);

% Do stochastic gradient descent
epoch = 50;
batchSize = 10;
eta = 0.01;
numBatches = 1000000/10;

for j = 1:epoch
    
ShuffledX = shuffleRow(X);

for i = 1:numBatches
  [Wvector1,Wvector2,bvec1 ,bvec2] = update_mini_batch(Wvector1, Wvector2, bvec1, bvec2, X( ((i-1)*batchSize+1 ):(i*batchSize),:),y(((i-1)*batchSize+1 ):(i*batchSize),1), eta);

end


end

end