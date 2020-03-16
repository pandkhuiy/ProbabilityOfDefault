%%
load('Data2018.mat'); %
load('Data2017.mat'); 

% We do not makes use of IV values here for forward selection.
% InfoValue2017Q1(end-1) = [];
% InfoValue2017Q2(end-1) = [];
% InfoValue2017Q3(end-1) = [];
% InfoValue2017Q4(end-1) = [];

X_WoE2017Q1(:,[16,19:20]) = [];
X_WoE2017Q2(:,[16,19:20]) = [];
X_WoE2017Q3(:,[16,19:20]) = [];
X_WoE2017Q4(:,[16,19:20]) = [];

sprintf('Start Q1 feature selection')
temp2Q1 = ForwardSelection(X_WoE2017Q1,y2017Q1);
sprintf('Q1 feature selection done')

sprintf('Start Q2 feature selection')
temp2Q2 = ForwardSelection(X_WoE2017Q2,y2017Q2);
sprintf('Q2 feature selection done')

sprintf('Start Q3 feature selection')
temp2Q3 = ForwardSelection(X_WoE2017Q3,y2017Q3);
sprintf('Q3 feature selection done')

sprintf('Start Q4 feature selection')
temp2Q4 = ForwardSelection(X_WoE2017Q4,y2017Q4);
sprintf('Q4 feature selection done')

X1= normalize(X_WoE2017Q1(:,temp2Q1 )); %
X12 = normalize(X_WoE2017Q2(:,temp2Q1)); %
 
X2=normalize(X_WoE2017Q2(:,temp2Q2)); %
X23 = normalize(X_WoE2017Q3(:,temp2Q2)); %
 
X3=normalize(X_WoE2017Q3(:,temp2Q3)); %
X34 = normalize(X_WoE2017Q4(:,temp2Q3));%
 
X4=normalize(X_WoE2017Q4(:,temp2Q4));%
X4new = normalize(X_WoE2018Q1(:,temp2Q4)); %

 
y1=y2017Q1; %
 
y2=y2017Q2; %
 
y3=y2017Q3; %
 
y4=y2017Q4; %

ynew=y2018Q1; %
 


% LR
results_LR2017_NoAdaSyn_MW = zeros(4,6);

[PCC, AUC, PG,BS, KS, H] = LRMW(X1,X12,y1,y2);

results_LR2017_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];

[PCC, AUC, PG,BS, KS, H] = LRMW(X2,X23,y2,y3);

results_LR2017_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];

[PCC, AUC, PG,BS, KS, H] = LRMW(X3,X34,y3,y4);

results_LR2017_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];

[PCC, AUC, PG,BS, KS, H] = LRMW(X4,X4new,y4,ynew);

results_LR2017_NoAdaSyn_MW(4,:) = [PCC, AUC, PG,BS, KS, H];

results_LR2017_NoAdaSyn_MW = mean(results_LR2017_NoAdaSyn_MW);


% DT
results_DT2017_NoAdaSyn_MW = zeros(4,6);
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_DTQ1] = DTMW2(X1,X12,y1,y2);
 
results_DT2017_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_DTQ2] = DTMW2(X2,X23,y2,y3);
 
results_DT2017_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_DTQ3] = DTMW2(X3,X34,y3,y4);
 
results_DT2017_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_DTQ4] = DTMW2(X4,X4new,y4,ynew);
 
results_DT2017_NoAdaSyn_MW(4,:) = [PCC, AUC, PG,BS, KS, H];

results_DT2017_NoAdaSyn_MW = mean(results_DT2017_NoAdaSyn_MW);


% FNNnew

results_FNN2017_NoAdaSyn_MW = zeros(4,6);

[PCC, AUC, PG,BS, KS, H, Optimal_HP_FNNQ1] = FNNnewMW2(X1,X12,y1,y2);

results_FNN2017_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];

[PCC, AUC, PG,BS, KS, H, Optimal_HP_FNNQ2] = FNNnewMW2(X2,X23,y2,y3);

results_FNN2017_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];

[PCC, AUC, PG,BS, KS, H, Optimal_HP_FNNQ3] = FNNnewMW2(X3,X34,y3,y4);

results_FNN2017_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];

[PCC, AUC, PG,BS, KS, H, Optimal_HP_FNNQ4] = FNNnewMW2(X4,X4new,y4,ynew);

results_FNN2017_NoAdaSyn_MW(4,:) = [PCC, AUC, PG,BS, KS, H];

results_FNN2017_NoAdaSyn_MW = mean(results_FNN2017_NoAdaSyn_MW);



% LinearSVM

results_LinearSVM2017_NoAdaSyn_MW = zeros(4,6);
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_LinSVMQ1] = LinearSVMMW2(X1,X12,y1,y2);
 
results_LinearSVM2017_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_LinSVMQ2] = LinearSVMMW2(X2,X23,y2,y3);
 
results_LinearSVM2017_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_LinSVMQ3] = LinearSVMMW2(X3,X34,y3,y4);
 
results_LinearSVM2017_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_LinSVMQ4] = LinearSVMMW2(X4,X4new,y4,ynew);
 
results_LinearSVM2017_NoAdaSyn_MW(4,:) = [PCC, AUC, PG,BS, KS, H];

results_LinearSVM2017_NoAdaSyn_MW = mean(results_LinearSVM2017_NoAdaSyn_MW);


% rbfSVMMW

results_rbfSVM2017_NoAdaSyn_MW = zeros(4,6);
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_RbfSVMQ1] = rbfSVMMW2(X1,X12,y1,y2);
 
results_rbfSVM2017_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_RbfSVMQ2] = rbfSVMMW2(X2,X23,y2,y3);
 
results_rbfSVM2017_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_RbfSVMQ3] = rbfSVMMW2(X3,X34,y3,y4);
 
results_rbfSVM2017_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];
 
 [PCC, AUC, PG,BS, KS, H, Optimal_HP_RbfSVMQ4] = rbfSVMMW2(X4,X4new,y4,ynew);
 
results_rbfSVM2017_NoAdaSyn_MW(4,:) = [PCC, AUC, PG,BS, KS, H];

results_rbfSVM2017_NoAdaSyn_MW = mean(results_rbfSVM2017_NoAdaSyn_MW);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% ensembles: BaggingDT   BaggingFNN, AdaBoostDT, and HeterogeneousEnsembles: avgS, avgW, Stacking %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Bagging DT
results_BaggingDT2017_NoAdaSyn_MW = zeros(4,6);
 
[PCC, AUC, PG,BS, KS, H] = BaggingDTMW(X1,X12,y1,y2, Optimal_HP_DTQ1);
 
results_BaggingDT2017_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = BaggingDTMW(X2,X23,y2,y3, Optimal_HP_DTQ2);
 
results_BaggingDT2017_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = BaggingDTMW(X3,X34,y3,y4, Optimal_HP_DTQ3);
 
results_BaggingDT2017_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];

[PCC, AUC, PG,BS, KS, H] = BaggingDTMW(X4,X4new,y4,ynew, Optimal_HP_DTQ4);
 
results_BaggingDT2017_NoAdaSyn_MW(4,:) = [PCC, AUC, PG,BS, KS, H];

results_BaggingDT2017_NoAdaSyn_MW = mean(results_BaggingDT2017_NoAdaSyn_MW);



% Bagging FNN
results_BaggingFNNnew2017_NoAdaSyn_MW = zeros(4,6);
 
[PCC, AUC, PG,BS, KS, H] = BaggingFNNnewMW(X1,X12,y1,y2, Optimal_HP_FNNQ1);
 
results_BaggingFNNnew2017_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = BaggingFNNnewMW(X2,X23,y2,y3, Optimal_HP_FNNQ2);
 
results_BaggingFNNnew2017_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = BaggingFNNnewMW(X3,X34,y3,y4, Optimal_HP_FNNQ3);
 
results_BaggingFNNnew2017_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];

[PCC, AUC, PG,BS, KS, H] = BaggingFNNnewMW(X4,X4new,y4,ynew, Optimal_HP_FNNQ4);
 
results_BaggingFNNnew2017_NoAdaSyn_MW(4,:) = [PCC, AUC, PG,BS, KS, H];

results_BaggingFNNnew2017_NoAdaSyn_MW = mean(results_BaggingFNNnew2017_NoAdaSyn_MW);


% AdaBoostDT
results_AdaBoostDT2017_NoAdaSyn_MW = zeros(4,6);
 
[PCC, AUC, PG,BS, KS, H] = AdaBoostDTMW(X1,X12,y1,y2, Optimal_HP_DTQ1);
 
results_AdaBoostDT2017_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = AdaBoostDTMW(X2,X23,y2,y3, Optimal_HP_DTQ2);
 
results_AdaBoostDT2017_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = AdaBoostDTMW(X3,X34,y3,y4, Optimal_HP_DTQ3);
 
results_AdaBoostDT2017_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];

[PCC, AUC, PG,BS, KS, H] = AdaBoostDTMW(X4,X4new,y4,ynew, Optimal_HP_DTQ4);
 
results_AdaBoostDT2017_NoAdaSyn_MW(4,:) = [PCC, AUC, PG,BS, KS, H];

results_AdaBoostDT2017_NoAdaSyn_MW = mean(results_AdaBoostDT2017_NoAdaSyn_MW);


% avgS, avgW and Stacking in one function
results_avgS2017_NoAdaSyn_MW = zeros(4,6);
results_avgW2017_NoAdaSyn_MW = zeros(4,6);
results_Stacking2017_NoAdaSyn_MW = zeros(4,6);

[PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS,...
 PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW,...
 PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking] = avgS_avgW_Stacking(X1,X12,y1,y2,...
                                                                                                    Optimal_HP_FNNQ1, Optimal_HP_DTQ1, Optimal_HP_RbfSVMQ1, Optimal_HP_LinSVMQ1 );

results_avgS2017_NoAdaSyn_MW(1,:) = [PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS];
results_avgW2017_NoAdaSyn_MW(1,:) = [PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW];
results_Stacking2017_NoAdaSyn_MW(1,:) = [PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking];

[PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS,...
 PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW,...
 PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking] = avgS_avgW_Stacking(X2,X23,y2,y3,...
                                                                                                    Optimal_HP_FNNQ2, Optimal_HP_DTQ2, Optimal_HP_RbfSVMQ2, Optimal_HP_LinSVMQ2 );

results_avgS2017_NoAdaSyn_MW(2,:) = [PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS];
results_avgW2017_NoAdaSyn_MW(2,:) = [PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW];
results_Stacking2017_NoAdaSyn_MW(2,:) = [PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking];

[PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS,...
 PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW,...
 PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking] = avgS_avgW_Stacking(X3,X34,y3,y4,...
                                                                                                    Optimal_HP_FNNQ3, Optimal_HP_DTQ3, Optimal_HP_RbfSVMQ3, Optimal_HP_LinSVMQ3 );

results_avgS2017_NoAdaSyn_MW(3,:) = [PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS];
results_avgW2017_NoAdaSyn_MW(3,:) = [PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW];
results_Stacking2017_NoAdaSyn_MW(3,:) = [PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking];

[PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS,...
 PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW,...
 PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking] = avgS_avgW_Stacking(X4,X4new,y4,ynew,....
                                                                                                    Optimal_HP_FNNQ4, Optimal_HP_DTQ4, Optimal_HP_RbfSVMQ4, Optimal_HP_LinSVMQ4 );

results_avgS2017_NoAdaSyn_MW(4,:) = [PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS];
results_avgW2017_NoAdaSyn_MW(4,:) = [PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW];
results_Stacking2017_NoAdaSyn_MW(4,:) = [PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking];

results_avgS2017_NoAdaSyn_MW     = mean( results_avgS2017_NoAdaSyn_MW );
results_avgW2017_NoAdaSyn_MW     = mean( results_avgW2017_NoAdaSyn_MW );
results_Stacking2017_NoAdaSyn_MW = mean( results_Stacking2017_NoAdaSyn_MW );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                                                                                        
                                                                                        