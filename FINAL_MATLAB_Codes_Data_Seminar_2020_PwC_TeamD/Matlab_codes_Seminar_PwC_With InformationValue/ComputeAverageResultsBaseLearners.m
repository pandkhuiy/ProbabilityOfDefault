function  [AverageResultsMatrix] = ComputeAverageResultsBaseLearners()
%% Construct time series of the computed performance measures for each of the base learner from 2000 till 2018.

% The following two lines of codes are for the original predictive performance results where we
% chose features based information values and interpretation.
% comment out one of them (Ctrl + t) to get results in this program.

load('Results_BaseLearners_And_Ensembles_NoAdaSyn_MW.mat');
%load('Results_BaseLearners_And_Ensembles_WithAdaSyn_MW.mat');

% The following two lines of codes are for the predictive performance results where we
% chose features based on forward selection and (all) 20 features as input values.
% comment out one of them (Ctrl + t) to get results in this program.

%load('results_BaseLeaners_Ensembles_ForwardSelection_MW_NoAdasyn');
%load('results_BaseLeaners_Ensembles_ForwardSelection_MW_WithAdasyn');

%!!!!!!!! the above load function can only be used one at a time. !!!!!

% make time series of PCC values, AUC values,  PG values, BS values , KS
% values, and H-measure values for each of the four base learners.
% Then compute the average over the 18 years and plot time series for each
% measure.
TimeVector = (2000:2018)';

ResultsMatrixFNN = [results_FNN2000_NoAdaSyn_MW;results_FNN2001_NoAdaSyn_MW;results_FNN2002_NoAdaSyn_MW;...
                    results_FNN2003_NoAdaSyn_MW;results_FNN2004_NoAdaSyn_MW;results_FNN2005_NoAdaSyn_MW;...
                    results_FNN2006_NoAdaSyn_MW;results_FNN2007_NoAdaSyn_MW;results_FNN2008_NoAdaSyn_MW;...
                    results_FNN2009_NoAdaSyn_MW;results_FNN2010_NoAdaSyn_MW;results_FNN2011_NoAdaSyn_MW;...
                    results_FNN2012_NoAdaSyn_MW;results_FNN2013_NoAdaSyn_MW;results_FNN2014_NoAdaSyn_MW;...
                    results_FNN2015_NoAdaSyn_MW;results_FNN2016_NoAdaSyn_MW;results_FNN2017_NoAdaSyn_MW;results_FNN2018_NoAdaSyn_MW];
 
ResultsMatrixLR  = [results_LR2000_NoAdaSyn_MW;results_LR2001_NoAdaSyn_MW;results_LR2002_NoAdaSyn_MW;...
                    results_LR2003_NoAdaSyn_MW;results_LR2004_NoAdaSyn_MW;results_LR2005_NoAdaSyn_MW;...
                    results_LR2006_NoAdaSyn_MW;results_LR2007_NoAdaSyn_MW;results_LR2008_NoAdaSyn_MW;...
                    results_LR2009_NoAdaSyn_MW;results_LR2010_NoAdaSyn_MW;results_LR2011_NoAdaSyn_MW;...
                    results_LR2012_NoAdaSyn_MW;results_LR2013_NoAdaSyn_MW;results_LR2014_NoAdaSyn_MW;...
                    results_LR2015_NoAdaSyn_MW;results_LR2016_NoAdaSyn_MW;results_LR2017_NoAdaSyn_MW;results_LR2018_NoAdaSyn_MW];
 
ResultsMatrixlinearSVM = [results_LinearSVM2000_NoAdaSyn_MW;results_LinearSVM2001_NoAdaSyn_MW;results_LinearSVM2002_NoAdaSyn_MW;...
                    results_LinearSVM2003_NoAdaSyn_MW;results_LinearSVM2004_NoAdaSyn_MW;results_LinearSVM2005_NoAdaSyn_MW;...
                    results_LinearSVM2006_NoAdaSyn_MW;results_LinearSVM2007_NoAdaSyn_MW;results_LinearSVM2008_NoAdaSyn_MW;...
                    results_LinearSVM2009_NoAdaSyn_MW;results_LinearSVM2010_NoAdaSyn_MW;results_LinearSVM2011_NoAdaSyn_MW;...
                    results_LinearSVM2012_NoAdaSyn_MW;results_LinearSVM2013_NoAdaSyn_MW;results_LinearSVM2014_NoAdaSyn_MW;...
                    results_LinearSVM2015_NoAdaSyn_MW;results_LinearSVM2016_NoAdaSyn_MW;results_LinearSVM2017_NoAdaSyn_MW;results_LinearSVM2018_NoAdaSyn_MW];
 
ResultsMatrixRbfSVM = [results_rbfSVM2000_NoAdaSyn_MW;results_rbfSVM2001_NoAdaSyn_MW;results_rbfSVM2002_NoAdaSyn_MW;...
                    results_rbfSVM2003_NoAdaSyn_MW;results_rbfSVM2004_NoAdaSyn_MW;results_rbfSVM2005_NoAdaSyn_MW;...
                    results_rbfSVM2006_NoAdaSyn_MW;results_rbfSVM2007_NoAdaSyn_MW;results_rbfSVM2008_NoAdaSyn_MW;...
                    results_rbfSVM2009_NoAdaSyn_MW;results_rbfSVM2010_NoAdaSyn_MW;results_rbfSVM2011_NoAdaSyn_MW;...
                    results_rbfSVM2012_NoAdaSyn_MW;results_rbfSVM2013_NoAdaSyn_MW;results_rbfSVM2014_NoAdaSyn_MW;...
                    results_rbfSVM2015_NoAdaSyn_MW;results_rbfSVM2016_NoAdaSyn_MW;results_rbfSVM2017_NoAdaSyn_MW;results_rbfSVM2018_NoAdaSyn_MW];
 
ResultsMatrixDT  = [results_DT2000_NoAdaSyn_MW;results_DT2001_NoAdaSyn_MW;results_DT2002_NoAdaSyn_MW;...
                    results_DT2003_NoAdaSyn_MW;results_DT2004_NoAdaSyn_MW;results_DT2005_NoAdaSyn_MW;...
                    results_DT2006_NoAdaSyn_MW;results_DT2007_NoAdaSyn_MW;results_DT2008_NoAdaSyn_MW;...
                    results_DT2009_NoAdaSyn_MW;results_DT2010_NoAdaSyn_MW;results_DT2011_NoAdaSyn_MW;...
                    results_DT2012_NoAdaSyn_MW;results_DT2013_NoAdaSyn_MW;results_DT2014_NoAdaSyn_MW;...
                    results_DT2015_NoAdaSyn_MW;results_DT2016_NoAdaSyn_MW;results_DT2017_NoAdaSyn_MW;results_DT2018_NoAdaSyn_MW];

% PCC, AUC, PG, BS, KS, H
AverageResultsMatrix = zeros(10,6);


%%%%%%%%%%%%%%%%%%%% LR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansLR = zeros(1,6);
stddevsLR = zeros(1,6);
means = mean(ResultsMatrixLR);
stddevs = std(ResultsMatrixLR);
% PCC, KS, AUC, PG, H, BS
meansLR(1) = means(1); meansLR(2) = means(5); meansLR(3) = means(2); meansLR(4) = means(3); meansLR(5) = means(6); meansLR(6) = means(4);  
stddevsLR(1) = stddevs(1); stddevsLR(2) = stddevs(5); stddevsLR(3) = stddevs(2); stddevsLR(4) = stddevs(3); stddevsLR(5) = stddevs(6); stddevsLR(6) = stddevs(4);                

AverageResultsMatrix(1,:) = meansLR;
AverageResultsMatrix(2,:) = stddevsLR;

%%%%%%%%%%%%%%%%%%%% DT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansDT = zeros(1,6);
stddevsDT = zeros(1,6);
means = mean(ResultsMatrixDT);
stddevs = std(ResultsMatrixDT);
% PCC, KS, AUC, PG, H, BS
meansDT(1) = means(1); meansDT(2) = means(5); meansDT(3) = means(2); meansDT(4) = means(3); meansDT(5) = means(6); meansDT(6) = means(4);  
stddevsDT(1) = stddevs(1); stddevsDT(2) = stddevs(5); stddevsDT(3) = stddevs(2); stddevsDT(4) = stddevs(3); stddevsDT(5) = stddevs(6); stddevsDT(6) = stddevs(4);                
 
AverageResultsMatrix(9,:) = meansDT;
AverageResultsMatrix(10,:) = stddevsDT;
	

%%%%%%%%%%%%%%%%%%%% FNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansFNN = zeros(1,6);
stddevsFNN = zeros(1,6);
means = mean(ResultsMatrixFNN);
stddevs = std(ResultsMatrixFNN);
% PCC, KS, AUC, PG, H, BS
meansFNN(1) = means(1); meansFNN(2) = means(5); meansFNN(3) = means(2); meansFNN(4) = means(3); meansFNN(5) = means(6); meansFNN(6) = means(4);  
stddevsFNN(1) = stddevs(1); stddevsFNN(2) = stddevs(5); stddevsFNN(3) = stddevs(2); stddevsFNN(4) = stddevs(3); stddevsFNN(5) = stddevs(6); stddevsFNN(6) = stddevs(4);                
 
AverageResultsMatrix(7,:) = meansFNN;
AverageResultsMatrix(8,:) = stddevsFNN;

%%%%%%%%%%%%%%%%%%%%% linear SVM  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meanslinearSVM = zeros(1,6);
stddevslinearSVM = zeros(1,6);
means = mean(ResultsMatrixlinearSVM);
stddevs = std(ResultsMatrixlinearSVM);
% PCC, KS, AUC, PG, H, BS
meanslinearSVM(1) = means(1); meanslinearSVM(2) = means(5); meanslinearSVM(3) = means(2); meanslinearSVM(4) = means(3); meanslinearSVM(5) = means(6); meanslinearSVM(6) = means(4);  
stddevslinearSVM(1) = stddevs(1); stddevslinearSVM(2) = stddevs(5); stddevslinearSVM(3) = stddevs(2); stddevslinearSVM(4) = stddevs(3); stddevslinearSVM(5) = stddevs(6); stddevslinearSVM(6) = stddevs(4);                
 
AverageResultsMatrix(3,:) = meanslinearSVM;
AverageResultsMatrix(4,:) = stddevslinearSVM;


%%%%%%%%%%%%%%%%%%%%%% Rbf SVM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansRbfSVM = zeros(1,6);
stddevsRbfSVM = zeros(1,6);
means = mean(ResultsMatrixRbfSVM);
stddevs = std(ResultsMatrixRbfSVM);
% PCC, KS, AUC, PG, H, BS
meansRbfSVM(1) = means(1); meansRbfSVM(2) = means(5); meansRbfSVM(3) = means(2); meansRbfSVM(4) = means(3); meansRbfSVM(5) = means(6); meansRbfSVM(6) = means(4);  
stddevsRbfSVM(1) = stddevs(1); stddevsRbfSVM(2) = stddevs(5); stddevsRbfSVM(3) = stddevs(2); stddevsRbfSVM(4) = stddevs(3); stddevsRbfSVM(5) = stddevs(6); stddevsRbfSVM(6) = stddevs(4);                
 
AverageResultsMatrix(5,:) = meansRbfSVM;
AverageResultsMatrix(6,:) = stddevsRbfSVM;


%%%%%%%%%%%%%%%%%%%%%%%%  DT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PCC, AUC, PG, BS, KS, H
% Create plots

figure;
%PCC
subplot(3,2,1);
FNN = plot(TimeVector,ResultsMatrixFNN(:,1),'-','LineWidth',1.65);
hold on 
LR = plot(TimeVector,ResultsMatrixLR(:,1), '-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixlinearSVM(:,1), '-k','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixRbfSVM(:,1), '-m','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixDT(:,1), '-g','LineWidth',1.65);

xlabel('Year');
ylabel('PCC');
grid on;
legend([FNN LR],'FNN','LR');

%AUC
subplot(3,2,2);
plot(TimeVector,ResultsMatrixFNN(:,2),'-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixLR(:,2), '-','LineWidth',1.65);
hold on 
SVML = plot(TimeVector,ResultsMatrixlinearSVM(:,2), '-k','LineWidth',1.65);
hold on
SVMRbf = plot(TimeVector,ResultsMatrixRbfSVM(:,2), '-m','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixDT(:,2), '-g','LineWidth',1.65);
xlabel('Year');
ylabel('AUC');
grid on;
legend([SVML SVMRbf],'SVM-L','SVM-Rbf');

%PG
subplot(3,2,3);
plot(TimeVector,ResultsMatrixFNN(:,3),'-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixLR(:,3), '-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixlinearSVM(:,3), '-k','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixRbfSVM(:,3), '-m','LineWidth',1.65);
hold on
DT = plot(TimeVector,ResultsMatrixDT(:,3), '-g','LineWidth',1.65);
xlabel('Year');
ylabel('PG');
grid on;
legend(DT,'DT');

%BS
subplot(3,2,4);
plot(TimeVector,ResultsMatrixFNN(:,4),'-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixLR(:,4), '-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixlinearSVM(:,4), '-k','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixRbfSVM(:,4), '-m','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixDT(:,4), '-g','LineWidth',1.65);
xlabel('Year');
ylabel('BS');
grid on;

%KS
subplot(3,2,5);
plot(TimeVector,ResultsMatrixFNN(:,5),'-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixLR(:,5), '-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixlinearSVM(:,5), '-k','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixRbfSVM(:,5), '-m','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixDT(:,5), '-g','LineWidth',1.65);
xlabel('Year');
ylabel('KS');
grid on;

%H-measure
subplot(3,2,6);
plot(TimeVector,ResultsMatrixFNN(:,6),'-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixLR(:,6), '-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixlinearSVM(:,6), '-k','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixRbfSVM(:,6), '-m','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixDT(:,6), '-g','LineWidth',1.65);
xlabel('Year');
ylabel('H-measure');
grid on;




end