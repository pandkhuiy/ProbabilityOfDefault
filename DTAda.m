function[temp] = DTAdaboost(Xtrain,Ytrain,Xtest)

DTada = fitcensemble(Xtrain,Ytrain,'Method','AdaBoostM2','ScoreTransform','logit');

[~,b] = predict(DTada,Xtest);
temp=b(:,2);

end