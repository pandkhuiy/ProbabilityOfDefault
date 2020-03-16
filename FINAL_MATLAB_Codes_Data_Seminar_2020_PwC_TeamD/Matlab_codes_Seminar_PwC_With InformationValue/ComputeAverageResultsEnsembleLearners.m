function  [AverageResultsMatrix] = ComputeAverageResultsEnsembleLearners()
%% Construct time series of the computed performance measures for each of the base learner from 2000 till 2018. Also compute average performance measures (returned as a matrix).

% The following two lines of codes are for the original predictive performance results where we
% chose features based information values and interpretation.
% comment out one of them (Ctrl + t) to get results in this program.

%load('Results_BaseLearners_And_Ensembles_NoAdaSyn_MW.mat');
%load('Results_BaseLearners_And_Ensembles_WithAdaSyn_MW.mat');

% The following two lines of codes are for the predictive performance results where we
% chose features based on forward selection and (all) 20 features as input values.
% comment out one of them (Ctrl + t) to get results in this program.

%load('results_BaseLeaners_Ensembles_ForwardSelection_MW_NoAdasyn');
load('results_BaseLeaners_Ensembles_ForwardSelection_MW_WithAdasyn');


% make time series of PCC values, AUC values,  PG values, BS values , KS
% values, and H-measure values for each of the four base learners.
% Then compute the average over the 18 years and plot time series for each
% measure.
TimeVector = (2000:2018)';

ResultsMatrixBaggingFNNnew = [results_BaggingFNNnew2000_NoAdaSyn_MW;results_BaggingFNNnew2001_NoAdaSyn_MW;results_BaggingFNNnew2002_NoAdaSyn_MW;...
                    results_BaggingFNNnew2003_NoAdaSyn_MW;results_BaggingFNNnew2004_NoAdaSyn_MW;results_BaggingFNNnew2005_NoAdaSyn_MW;...
                    results_BaggingFNNnew2006_NoAdaSyn_MW;results_BaggingFNNnew2007_NoAdaSyn_MW;results_BaggingFNNnew2008_NoAdaSyn_MW;...
                    results_BaggingFNNnew2009_NoAdaSyn_MW;results_BaggingFNNnew2010_NoAdaSyn_MW;results_BaggingFNNnew2011_NoAdaSyn_MW;...
                    results_BaggingFNNnew2012_NoAdaSyn_MW;results_BaggingFNNnew2013_NoAdaSyn_MW;results_BaggingFNNnew2014_NoAdaSyn_MW;...
                    results_BaggingFNNnew2015_NoAdaSyn_MW;results_BaggingFNNnew2016_NoAdaSyn_MW;results_BaggingFNNnew2017_NoAdaSyn_MW;results_BaggingFNNnew2018_NoAdaSyn_MW];
  
ResultsMatrixAdaBoostDT  = [results_AdaBoostDT2000_NoAdaSyn_MW;results_AdaBoostDT2001_NoAdaSyn_MW;results_AdaBoostDT2002_NoAdaSyn_MW;...
                    results_AdaBoostDT2003_NoAdaSyn_MW;results_AdaBoostDT2004_NoAdaSyn_MW;results_AdaBoostDT2005_NoAdaSyn_MW;...
                    results_AdaBoostDT2006_NoAdaSyn_MW;results_AdaBoostDT2007_NoAdaSyn_MW;results_AdaBoostDT2008_NoAdaSyn_MW;...
                    results_AdaBoostDT2009_NoAdaSyn_MW;results_AdaBoostDT2010_NoAdaSyn_MW;results_AdaBoostDT2011_NoAdaSyn_MW;...
                    results_AdaBoostDT2012_NoAdaSyn_MW;results_AdaBoostDT2013_NoAdaSyn_MW;results_AdaBoostDT2014_NoAdaSyn_MW;...
                    results_AdaBoostDT2015_NoAdaSyn_MW;results_AdaBoostDT2016_NoAdaSyn_MW;results_AdaBoostDT2017_NoAdaSyn_MW;results_AdaBoostDT2018_NoAdaSyn_MW];
 
ResultsMatrixBaggingDT  = [results_BaggingDT2000_NoAdaSyn_MW;results_BaggingDT2001_NoAdaSyn_MW;results_BaggingDT2002_NoAdaSyn_MW;...
                            results_BaggingDT2003_NoAdaSyn_MW;results_BaggingDT2004_NoAdaSyn_MW;results_BaggingDT2005_NoAdaSyn_MW;...
                            results_BaggingDT2006_NoAdaSyn_MW;results_BaggingDT2007_NoAdaSyn_MW;results_BaggingDT2008_NoAdaSyn_MW;...
                            results_BaggingDT2009_NoAdaSyn_MW;results_BaggingDT2010_NoAdaSyn_MW;results_BaggingDT2011_NoAdaSyn_MW;...
                            results_BaggingDT2012_NoAdaSyn_MW;results_BaggingDT2013_NoAdaSyn_MW;results_BaggingDT2014_NoAdaSyn_MW;...
                            results_BaggingDT2015_NoAdaSyn_MW;results_BaggingDT2016_NoAdaSyn_MW;results_BaggingDT2017_NoAdaSyn_MW;results_BaggingDT2018_NoAdaSyn_MW];

ResultsMatrixavgS  = [results_avgS2000_NoAdaSyn_MW;results_avgS2001_NoAdaSyn_MW;results_avgS2002_NoAdaSyn_MW;...
                    results_avgS2003_NoAdaSyn_MW;results_avgS2004_NoAdaSyn_MW;results_avgS2005_NoAdaSyn_MW;...
                    results_avgS2006_NoAdaSyn_MW;results_avgS2007_NoAdaSyn_MW;results_avgS2008_NoAdaSyn_MW;...
                    results_avgS2009_NoAdaSyn_MW;results_avgS2010_NoAdaSyn_MW;results_avgS2011_NoAdaSyn_MW;...
                    results_avgS2012_NoAdaSyn_MW;results_avgS2013_NoAdaSyn_MW;results_avgS2014_NoAdaSyn_MW;...
                    results_avgS2015_NoAdaSyn_MW;results_avgS2016_NoAdaSyn_MW;results_avgS2017_NoAdaSyn_MW;results_avgS2018_NoAdaSyn_MW];
                
ResultsMatrixStacking  = [results_Stacking2000_NoAdaSyn_MW;results_Stacking2001_NoAdaSyn_MW;results_Stacking2002_NoAdaSyn_MW;...
                    results_Stacking2003_NoAdaSyn_MW;results_Stacking2004_NoAdaSyn_MW;results_Stacking2005_NoAdaSyn_MW;...
                    results_Stacking2006_NoAdaSyn_MW;results_Stacking2007_NoAdaSyn_MW;results_Stacking2008_NoAdaSyn_MW;...
                    results_Stacking2009_NoAdaSyn_MW;results_Stacking2010_NoAdaSyn_MW;results_Stacking2011_NoAdaSyn_MW;...
                    results_Stacking2012_NoAdaSyn_MW;results_Stacking2013_NoAdaSyn_MW;results_Stacking2014_NoAdaSyn_MW;...
                    results_Stacking2015_NoAdaSyn_MW;results_Stacking2016_NoAdaSyn_MW;results_Stacking2017_NoAdaSyn_MW;results_Stacking2018_NoAdaSyn_MW];               
 
ResultsMatrixavgW  = [results_avgW2000_NoAdaSyn_MW;results_avgW2001_NoAdaSyn_MW;results_avgW2002_NoAdaSyn_MW;...
                    results_avgW2003_NoAdaSyn_MW;results_avgW2004_NoAdaSyn_MW;results_avgW2005_NoAdaSyn_MW;...
                    results_avgW2006_NoAdaSyn_MW;results_avgW2007_NoAdaSyn_MW;results_avgW2008_NoAdaSyn_MW;...
                    results_avgW2009_NoAdaSyn_MW;results_avgW2010_NoAdaSyn_MW;results_avgW2011_NoAdaSyn_MW;...
                    results_avgW2012_NoAdaSyn_MW;results_avgW2013_NoAdaSyn_MW;results_avgW2014_NoAdaSyn_MW;...
                    results_avgW2015_NoAdaSyn_MW;results_avgW2016_NoAdaSyn_MW;results_avgW2017_NoAdaSyn_MW;results_avgW2018_NoAdaSyn_MW];
                
ResultsMatrixLR  = [results_LR2000_NoAdaSyn_MW;results_LR2001_NoAdaSyn_MW;results_LR2002_NoAdaSyn_MW;...
                    results_LR2003_NoAdaSyn_MW;results_LR2004_NoAdaSyn_MW;results_LR2005_NoAdaSyn_MW;...
                    results_LR2006_NoAdaSyn_MW;results_LR2007_NoAdaSyn_MW;results_LR2008_NoAdaSyn_MW;...
                    results_LR2009_NoAdaSyn_MW;results_LR2010_NoAdaSyn_MW;results_LR2011_NoAdaSyn_MW;...
                    results_LR2012_NoAdaSyn_MW;results_LR2013_NoAdaSyn_MW;results_LR2014_NoAdaSyn_MW;...
                    results_LR2015_NoAdaSyn_MW;results_LR2016_NoAdaSyn_MW;results_LR2017_NoAdaSyn_MW;results_LR2018_NoAdaSyn_MW];

% PCC, AUC, PG, BS, KS, H
AverageResultsMatrix = zeros(13,6);


%%%%%%%%%%%%%%%%%%%% LR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansLR = zeros(1,6);
stddevsLR = zeros(1,6);
means = mean(ResultsMatrixLR);
stddevs = std(ResultsMatrixLR);
% PCC, KS, AUC, PG, H, BS
meansLR(1) = means(1); meansLR(2) = means(5); meansLR(3) = means(2); meansLR(4) = means(3); meansLR(5) = means(6); meansLR(6) = means(4);  
stddevsLR(1) = stddevs(1); stddevsLR(2) = stddevs(5); stddevsLR(3) = stddevs(2); stddevsLR(4) = stddevs(3); stddevsLR(5) = stddevs(6); stddevsLR(6) = stddevs(4);                

AverageResultsMatrix(1,:) = meansLR;


%%%%%%%%%%%%%%%%%%%% avgS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansavgS = zeros(1,6);
stddevsavgS = zeros(1,6);
means = mean(ResultsMatrixavgS);
stddevs = std(ResultsMatrixavgS);
% PCC, KS, AUC, PG, H, BS
meansavgS(1) = means(1); meansavgS(2) = means(5); meansavgS(3) = means(2); meansavgS(4) = means(3); meansavgS(5) = means(6); meansavgS(6) = means(4);  
stddevsavgS(1) = stddevs(1); stddevsavgS(2) = stddevs(5); stddevsavgS(3) = stddevs(2); stddevsavgS(4) = stddevs(3); stddevsavgS(5) = stddevs(6); stddevsavgS(6) = stddevs(4);                

AverageResultsMatrix(8,:) = meansavgS;
AverageResultsMatrix(9,:) = stddevsavgS;

%%%%%%%%%%%%%%%%%%%%% avgW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansavgW = zeros(1,6);
stddevsavgW = zeros(1,6);
means = mean(ResultsMatrixavgW);
stddevs = std(ResultsMatrixavgW);
% PCC, KS, AUC, PG, H, BS
meansavgW(1) = means(1); meansavgW(2) = means(5); meansavgW(3) = means(2); meansavgW(4) = means(3); meansavgW(5) = means(6); meansavgW(6) = means(4);  
stddevsavgW(1) = stddevs(1); stddevsavgW(2) = stddevs(5); stddevsavgW(3) = stddevs(2); stddevsavgW(4) = stddevs(3); stddevsavgW(5) = stddevs(6); stddevsavgW(6) = stddevs(4);                
 
AverageResultsMatrix(10,:) = meansavgW;
AverageResultsMatrix(11,:) = stddevsavgW;

%%%%%%%%%%%%%%%%%%%% Bagging DT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansBagDT = zeros(1,6);
stddevsBagDT = zeros(1,6);
means = mean(ResultsMatrixBaggingDT);
stddevs = std(ResultsMatrixBaggingDT);
% PCC, KS, AUC, PG, H, BS
meansBagDT(1) = means(1); meansBagDT(2) = means(5); meansBagDT(3) = means(2); meansBagDT(4) = means(3); meansBagDT(5) = means(6); meansBagDT(6) = means(4);  
stddevsBagDT(1) = stddevs(1); stddevsBagDT(2) = stddevs(5); stddevsBagDT(3) = stddevs(2); stddevsBagDT(4) = stddevs(3); stddevsBagDT(5) = stddevs(6); stddevsBagDT(6) = stddevs(4);                
 
AverageResultsMatrix(4,:) = meansBagDT;
AverageResultsMatrix(5,:) = stddevsBagDT;
	
%%%%%%%%%%%%%%%%%%%% Bagging FNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansBagFNN = zeros(1,6);
stddevsBagFNN = zeros(1,6);
means = mean(ResultsMatrixBaggingFNNnew);
stddevs = std(ResultsMatrixBaggingFNNnew);
% PCC, KS, AUC, PG, H, BS
meansBagFNN(1) = means(1); meansBagFNN(2) = means(5); meansBagFNN(3) = means(2); meansBagFNN(4) = means(3); meansBagFNN(5) = means(6); meansBagFNN(6) = means(4);  
stddevsBagFNN(1) = stddevs(1); stddevsBagFNN(2) = stddevs(5); stddevsBagFNN(3) = stddevs(2); stddevsBagFNN(4) = stddevs(3); stddevsBagFNN(5) = stddevs(6); stddevsBagFNN(6) = stddevs(4);                
 
AverageResultsMatrix(2,:) = meansBagFNN;
AverageResultsMatrix(3,:) = stddevsBagFNN;

%%%%%%%%%%%%%%%%%%%%%% AdaBoostDT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansAdaBoostDT = zeros(1,6);
stddevsAdaBoostDT = zeros(1,6);
means = mean(ResultsMatrixAdaBoostDT);
stddevs = std(ResultsMatrixAdaBoostDT);
% PCC, KS, AUC, PG, H, BS
meansAdaBoostDT(1) = means(1); meansAdaBoostDT(2) = means(5); meansAdaBoostDT(3) = means(2); meansAdaBoostDT(4) = means(3); meansAdaBoostDT(5) = means(6); meansAdaBoostDT(6) = means(4);  
stddevsAdaBoostDT(1) = stddevs(1); stddevsAdaBoostDT(2) = stddevs(5); stddevsAdaBoostDT(3) = stddevs(2); stddevsAdaBoostDT(4) = stddevs(3); stddevsAdaBoostDT(5) = stddevs(6); stddevsAdaBoostDT(6) = stddevs(4);                
 
AverageResultsMatrix(6,:) = meansAdaBoostDT;
AverageResultsMatrix(7,:) = stddevsAdaBoostDT;

%%%%%%%%%%%%%%%%%%%%%% Stacking %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
meansStacking = zeros(1,6);
stddevsStacking = zeros(1,6);
means = mean(ResultsMatrixStacking);
stddevs = std(ResultsMatrixStacking);
% PCC, KS, AUC, PG, H, BS
meansStacking(1) = means(1); meansStacking(2) = means(5); meansStacking(3) = means(2); meansStacking(4) = means(3); meansStacking(5) = means(6); meansStacking(6) = means(4);  
stddevsStacking(1) = stddevs(1); stddevsStacking(2) = stddevs(5); stddevsStacking(3) = stddevs(2); stddevsStacking(4) = stddevs(3); stddevsStacking(5) = stddevs(6); stddevsStacking(6) = stddevs(4);                
 
AverageResultsMatrix(12,:) = meansStacking;
AverageResultsMatrix(13,:) = stddevsStacking;


% PCC, AUC, PG, BS, KS, H
% Create plots

figure;
%PCC
subplot(3,2,1);
Stacking = plot(TimeVector,ResultsMatrixStacking(:,1),'-','LineWidth',1.65);
hold on 
avgS = plot(TimeVector,ResultsMatrixavgS(:,1), '-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixavgW(:,1), '-k','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixBaggingFNNnew(:,1), '-m','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixBaggingDT(:,1), '-g','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixAdaBoostDT(:,1), '-y','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixLR(:,1),'-','LineWidth',1.8);

xlabel('Year');
ylabel('PCC');
grid on;
legend([Stacking avgS],'Stacking','AvgS');

%AUC
subplot(3,2,2);
plot(TimeVector,ResultsMatrixStacking(:,2),'-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixavgS(:,2), '-','LineWidth',1.65);
hold on 
avgW = plot(TimeVector,ResultsMatrixavgW(:,2), '-k','LineWidth',1.65);
hold on
BagFNN = plot(TimeVector,ResultsMatrixBaggingFNNnew(:,2), '-m','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixBaggingDT(:,2), '-g','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixAdaBoostDT(:,2), '-y','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixLR(:,2),'-','LineWidth',1.8);

xlabel('Year');
ylabel('AUC');
grid on;
legend([avgW BagFNN],'AvgW','BagFNN');

%PG
subplot(3,2,3);
plot(TimeVector,ResultsMatrixStacking(:,3),'-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixavgS(:,3), '-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixavgW(:,3), '-k','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixBaggingFNNnew(:,3), '-m','LineWidth',1.65);
hold on
BagDT = plot(TimeVector,ResultsMatrixBaggingDT(:,3), '-g','LineWidth',1.65);
hold on
AdaBoostDT = plot(TimeVector,ResultsMatrixAdaBoostDT(:,3), '-y','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixLR(:,3),'-','LineWidth',1.8);

xlabel('Year');
ylabel('PG');
grid on;
legend([BagDT AdaBoostDT ],'BagDT', 'AdaBoostDT');

%BS
subplot(3,2,4);
plot(TimeVector,ResultsMatrixStacking(:,4),'-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixavgS(:,4), '-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixavgW(:,4), '-k','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixBaggingFNNnew(:,4), '-m','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixBaggingDT(:,4), '-g','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixAdaBoostDT(:,4), '-y','LineWidth',1.65);
hold on
LRBenchmark = plot(TimeVector,ResultsMatrixLR(:,4),'-','LineWidth',1.8);

xlabel('Year');
ylabel('BS');
grid on;
legend([LRBenchmark],'LR benchmark');

%KS
subplot(3,2,5);
plot(TimeVector,ResultsMatrixStacking(:,5),'-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixavgS(:,5), '-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixavgW(:,5), '-k','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixBaggingFNNnew(:,5), '-m','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixBaggingDT(:,5), '-g','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixAdaBoostDT(:,5), '-y','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixLR(:,5),'-','LineWidth',1.8);


xlabel('Year');
ylabel('KS');
grid on;

%H-measure
subplot(3,2,6);
plot(TimeVector,ResultsMatrixStacking(:,6),'-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixavgS(:,6), '-','LineWidth',1.65);
hold on 
plot(TimeVector,ResultsMatrixavgW(:,6), '-k','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixBaggingFNNnew(:,6), '-m','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixBaggingDT(:,6), '-g','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixAdaBoostDT(:,6), '-y','LineWidth',1.65);
hold on
plot(TimeVector,ResultsMatrixLR(:,6),'-','LineWidth',1.8);

xlabel('Year');
ylabel('H-measure');
grid on;




end