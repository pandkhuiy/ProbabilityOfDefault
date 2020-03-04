%%
load('Year2018.mat'); %

InfoValue2018Q1(end-1) = [];
InfoValue2018Q2(end-1) = [];
InfoValue2018Q3(end-1) = [];
InfoValue2018Q4(end-1) = [];

X_WoE2018Q1(:,14) = [];
X_WoE2018Q2(:,14) = [];
X_WoE2018Q3(:,14) = [];
X_WoE2018Q4(:,14) = [];

 
tempQ1=(InfoValue2018Q1>0.08).*(InfoValue2018Q1<inf);

temp2Q1=(find(tempQ1==1)+8)';

tempQ2=(InfoValue2018Q2>0.08).*(InfoValue2018Q2<inf);

temp2Q2=(find(tempQ2==1)+8)';

tempQ3=(InfoValue2018Q3>0.08).*(InfoValue2018Q3<inf);

temp2Q3=(find(tempQ3==1)+8)';


X1=normalize(X_WoE2018Q1(:,[1,2,3,4,5,6,7,8,temp2Q1])); %
X12 = normalize(X_WoE2018Q2(:,[1,2,3,4,5,6,7,8,temp2Q1])); %
 
X2=normalize(X_WoE2018Q2(:,[1,2,3,4,5,6,7,8,temp2Q2])); %
X23 = normalize(X_WoE2018Q3(:,[1,2,3,4,5,6,7,8,temp2Q2])); % 

X3=normalize(X_WoE2018Q3(:,[1,2,3,4,5,6,7,8,temp2Q3])); %
X34 = normalize(X_WoE2018Q4(:,[1,2,3,4,5,6,7,8,temp2Q3])); % 

 
y1=y2018Q1; %
 
y2=y2018Q2; %
 
y3=y2018Q3; %
 
y4=y2018Q4; %



% LR
results_LR2018_NoAdaSyn_MW = zeros(3,6);
 
[PCC, AUC, PG,BS, KS, H] = LRMW(X1,X12,y1,y2);
 
results_LR2018_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = LRMW(X2,X23,y2,y3);
 
results_LR2018_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = LRMW(X3,X34,y3,y4);
 
results_LR2018_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];
  
results_LR2018_NoAdaSyn_MW = mean(results_LR2018_NoAdaSyn_MW);
 
 
% DT
results_DT2018_NoAdaSyn_MW = zeros(3,6);
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_DTQ1] = DTMW2(X1,X12,y1,y2);
 
results_DT2018_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_DTQ2] = DTMW2(X2,X23,y2,y3);
 
results_DT2018_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_DTQ3] = DTMW2(X3,X34,y3,y4);
 
results_DT2018_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];
  
results_DT2018_NoAdaSyn_MW = mean(results_DT2018_NoAdaSyn_MW);
 
 
% FNNnew
 
results_FNN2018_NoAdaSyn_MW = zeros(3,6);
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_FNNQ1] = FNNnewMW2(X1,X12,y1,y2);
 
results_FNN2018_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_FNNQ2] = FNNnewMW2(X2,X23,y2,y3);
 
results_FNN2018_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_FNNQ3] = FNNnewMW2(X3,X34,y3,y4);
 
results_FNN2018_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];
  
results_FNN2018_NoAdaSyn_MW = mean(results_FNN2018_NoAdaSyn_MW);
 
 
 
% LinearSVM
 
results_LinearSVM2018_NoAdaSyn_MW = zeros(3,6);
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_LinSVMQ1] = LinearSVMMW2(X1,X12,y1,y2);
 
results_LinearSVM2018_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_LinSVMQ2] = LinearSVMMW2(X2,X23,y2,y3);
 
results_LinearSVM2018_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_LinSVMQ3] = LinearSVMMW2(X3,X34,y3,y4);
 
results_LinearSVM2018_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];
  
results_LinearSVM2018_NoAdaSyn_MW = mean(results_LinearSVM2018_NoAdaSyn_MW);
 
 
% rbfSVMMW
 
results_rbfSVM2018_NoAdaSyn_MW = zeros(3,6);
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_RbfSVMQ1] = rbfSVMMW2(X1,X12,y1,y2);
 
results_rbfSVM2018_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_RbfSVMQ2] = rbfSVMMW2(X2,X23,y2,y3);
 
results_rbfSVM2018_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H, Optimal_HP_RbfSVMQ3] = rbfSVMMW2(X3,X34,y3,y4);
 
results_rbfSVM2018_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];
  
results_rbfSVM2018_NoAdaSyn_MW = mean(results_rbfSVM2018_NoAdaSyn_MW);
 
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% ensembles: BaggingDT   BaggingFNN, AdaBoostDT, and HeterogeneousEnsembles: avgS, avgW, Stacking %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
 
% Bagging DT
results_BaggingDT2018_NoAdaSyn_MW = zeros(3,6);
 
[PCC, AUC, PG,BS, KS, H] = BaggingDTMW(X1,X12,y1,y2, Optimal_HP_DTQ1);
 
results_BaggingDT2018_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = BaggingDTMW(X2,X23,y2,y3, Optimal_HP_DTQ2);
 
results_BaggingDT2018_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = BaggingDTMW(X3,X34,y3,y4, Optimal_HP_DTQ3);
 
results_BaggingDT2018_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];
  
results_BaggingDT2018_NoAdaSyn_MW = mean(results_BaggingDT2018_NoAdaSyn_MW);
 
 
 
% Bagging FNN
results_BaggingFNNnew2018_NoAdaSyn_MW = zeros(3,6);
 
[PCC, AUC, PG,BS, KS, H] = BaggingFNNnewMW(X1,X12,y1,y2, Optimal_HP_FNNQ1);
 
results_BaggingFNNnew2018_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = BaggingFNNnewMW(X2,X23,y2,y3, Optimal_HP_FNNQ2);
 
results_BaggingFNNnew2018_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = BaggingFNNnewMW(X3,X34,y3,y4, Optimal_HP_FNNQ3);
 
results_BaggingFNNnew2018_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];

results_BaggingFNNnew2018_NoAdaSyn_MW = mean(results_BaggingFNNnew2018_NoAdaSyn_MW);
 
 
% AdaBoostDT
results_AdaBoostDT2018_NoAdaSyn_MW = zeros(3,6);
 
[PCC, AUC, PG,BS, KS, H] = AdaBoostDTMW(X1,X12,y1,y2, Optimal_HP_DTQ1);
 
results_AdaBoostDT2018_NoAdaSyn_MW(1,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = AdaBoostDTMW(X2,X23,y2,y3, Optimal_HP_DTQ2);
 
results_AdaBoostDT2018_NoAdaSyn_MW(2,:) = [PCC, AUC, PG,BS, KS, H];
 
[PCC, AUC, PG,BS, KS, H] = AdaBoostDTMW(X3,X34,y3,y4, Optimal_HP_DTQ3);
 
results_AdaBoostDT2018_NoAdaSyn_MW(3,:) = [PCC, AUC, PG,BS, KS, H];
 
results_AdaBoostDT2018_NoAdaSyn_MW = mean(results_AdaBoostDT2018_NoAdaSyn_MW);
 
 
% avgS, avgW and Stacking in one function
results_avgS2018_NoAdaSyn_MW = zeros(3,6);
results_avgW2018_NoAdaSyn_MW = zeros(3,6);
results_Stacking2018_NoAdaSyn_MW = zeros(3,6);
 
[PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS,...
 PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW,...
 PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking] = avgS_avgW_Stacking(X1,X12,y1,y2,...
                                                                                                    Optimal_HP_FNNQ1, Optimal_HP_DTQ1, Optimal_HP_RbfSVMQ1, Optimal_HP_LinSVMQ1 );
 
results_avgS2018_NoAdaSyn_MW(1,:) = [PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS];
results_avgW2018_NoAdaSyn_MW(1,:) = [PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW];
results_Stacking2018_NoAdaSyn_MW(1,:) = [PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking];
 
[PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS,...
 PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW,...
 PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking] = avgS_avgW_Stacking(X2,X23,y2,y3,...
                                                                                                    Optimal_HP_FNNQ2, Optimal_HP_DTQ2, Optimal_HP_RbfSVMQ2, Optimal_HP_LinSVMQ2 );
 
results_avgS2018_NoAdaSyn_MW(2,:) = [PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS];
results_avgW2018_NoAdaSyn_MW(2,:) = [PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW];
results_Stacking2018_NoAdaSyn_MW(2,:) = [PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking];
 
[PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS,...
 PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW,...
 PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking] = avgS_avgW_Stacking(X3,X34,y3,y4,...
                                                                                                    Optimal_HP_FNNQ3, Optimal_HP_DTQ3, Optimal_HP_RbfSVMQ3, Optimal_HP_LinSVMQ3 );
 
results_avgS2018_NoAdaSyn_MW(3,:) = [PCC_avgS, AUC_avgS, PG_avgS,BS_avgS, KS_avgS, H_avgS];
results_avgW2018_NoAdaSyn_MW(3,:) = [PCC_avgW, AUC_avgW, PG_avgW,BS_avgW, KS_avgW, H_avgW];
results_Stacking2018_NoAdaSyn_MW(3,:) = [PCC_Stacking, AUC_Stacking, PG_Stacking,BS_Stacking, KS_Stacking, H_Stacking];
 

results_avgS2018_NoAdaSyn_MW     = mean( results_avgS2018_NoAdaSyn_MW );
results_avgW2018_NoAdaSyn_MW     = mean( results_avgW2018_NoAdaSyn_MW );
results_Stacking2018_NoAdaSyn_MW = mean( results_Stacking2018_NoAdaSyn_MW );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

