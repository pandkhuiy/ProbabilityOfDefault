function [Ybagging]= bagging(X,y,bags,BL)


data=[X,y];                                     %put the X and y in 1 matrix
random_data = data(randperm(size(data, 1)), :); %shuffles the rows of the data
n=size(random_data,1);
trainingdata=random_data(1:ceil(n/2),:);        %first half is training data
testdata=random_data(ceil(n/2)+1,:);            %second half is test data
trainx=trainingdata(:,1:end-1);                 %split the x and y in the train and test data
trainy=trainingdata(:,end);
testx=testdata(:,1:end-1);
testy=testdata(:,end);
bs=ceil(ceil(n/2)*rand(ceil(n/2),bags));        %in this matrix the rows of the bootstraps are generates
defaultbags=zeros(size(testy,1),bags);          %create empty matrix to store the forcasted y


switch BL                                       %if BL == Case the next loop will be 
    case SVM
        for i=1:bags
            defaultbags(:,i)=SVM(trainx(bs(:,i)),trainy(bs(:,i)),testx); 
            %here should be the forecast with input trainX and trainY to train the model and input testX to forecast
            %output will be the forecasted y
        end
    case MLP
        for i=1:bags
            defaultbags(:,i)=MLP(trainx(bs(:,i)),trainy(bs(:,i)),testx);
        end
    case DT
        for i=1:bags
            defaultbags(:,i)=DT(trainx(bs(:,i)),trainy(bs(:,i)),testx);
        end
    case LR
        for i=1:bags
            defaultbags(:,i)=LR(trainx(bs(:,i)),trainy(bs(:,i)),testx);
        end
    otherwise
        for i=1:bags
            defaultbags(:,i)=LDA(trainx(bs(:,i)),trainy(bs(:,i)),testx);
        end
end
Ybagging = mean(defaultbags,2);
%now we should calculate the measures with the Ybagging compared to the
%testY
end