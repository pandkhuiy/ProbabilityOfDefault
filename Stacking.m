function [Ystacking]= Stacking(X,y,Ratio)

%amount of base learners
data=[X,y];                                                                 %put the X and y in 1 matrix
random_data = data(randperm(size(data, 1)), :);                             %shuffles the rows of the data
n=size(random_data,1);                                                      

TrainingSet=random_data(1:ceil(n*ratio(1)),:);                              %divide the data into 3 sets Training validation and test
TrainX=TrainingSet(:,1:end-1);
TrainY=TrainingSet(:,end);

ValidationSet=random_data(ceil(n*ratio(1))+1:ceil(n*(ratio(1)+ratio(2))),:);
ValiX=ValidationSet(:,1:end-1);
ValiY=ValidationSet(:,end);

TestSet=random_data(ceil(n*(ratio(2)+ratio(1)))+1:end,:);
TestX=TestSet(:,1:end-1);
TestY=TestSet(:,end);

MetaValidate=zeros(size(ValidationSet,1),5);                                %create 2 matrices to store the predictions for the meta learner
MetaTest=zeros(size(TestSet,1),5);
MetaValidate(:,end)=ValiY;
MetaTest(:,end)=TestY;

MetaValidate(:,1)=SVM(TrainX,TrainY,ValiX);                                 %use the training set to train baselearners and predict a vector using the testX and ValiX
MetaTest(:,1)=SVM(TrainX,TrainY,TestX);

MetaValidate(:,2)=LR(TrainX,TrainY,ValiX);
MetaTest(:,2)=LR(TrainX,TrainY,TestX);

MetaValidate(:,3)=MLP(TrainX,TrainY,ValiX);
MetaTest(:,3)=MLP(TrainX,TrainY,TestX);

MetaValidate(:,4)=LDA(TrainX,TrainY,ValiX);
MetaTest(:,4)=LDA(TrainX,TrainY,TestX);

Ystacking=LR(MetaValidate(:,end-1),MetaValidate(:,end),MetaTest(:,end-1));  %Use the predictions of the base learners and the ValiX to train the meta learner and uses TestX to predict TestY

%compare Ystacking and TestY and get the measures




end