function[temp] = DT(Xtrain,Ytrain,Xtest)

Tree = fitctree(Xtrain,Ytrain);

[~,b,~,~] = predict(Tree,Xtest);
temp=b(:,2);
end