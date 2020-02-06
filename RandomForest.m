function[temp] = RandomForest(Xtrain,Ytrain,Xtest,trees)

bs=ceil(size(Xtrain,1)*rand(size(Xtrain,1),trees));
predictions=zeros(size(Xtrain,1),trees);

for i=1:trees
    t = fitctree(Xtrain(bs(:,i),:),Ytrain(bs(:,i),:));
    [~,b,~,~] = predict(t,Xtest);
    predictions(:,i)=b(:,2);
end
temp=mean(predictions,2);   

end