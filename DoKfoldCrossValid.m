function [TrainingIndices, TestIndices] = DoKfoldCrossValid(y,k)
%% 
% [n,~] = size(X);

% size_block = n/k; % assuming that n/k results in integers.
% indices = ones(n,k);

cv = cvpartition(y,'KFold',k,'Stratify',true);
TrainingIndices = [cv.training(1) cv.training(2) cv.training(3) cv.training(4) cv.training(5)];
TestIndices = [cv.test(1) cv.test(2) cv.test(3) cv.test(4) cv.test(5)];
end
%alexander.kiel64@gmail.com
%Tomate123!