%%	
 load('Year2018.mat'); %
%  
temp=(avgIV>0.08).*(avgIV<inf);
temp2=(find(temp==1)+8)';
%  
X1=normalize(X_WoE2018Q1(:,[1,2,3,4,5,6,7,8,[temp2]])); %
 
X2=normalize(X_WoE2018Q2(:,[1,2,3,4,5,6,7,8,[temp2]])); %
 
X3=normalize(X_WoE2018Q3(:,[1,2,3,4,5,6,7,8,[temp2]])); %
 
X4=normalize(X_WoE2018Q4(:,[1,2,3,4,5,6,7,8,[temp2]])); %
 
y1=y2018Q1; %
 
y2=y2018Q2; %
 
y3=y2018Q3; %
 
y4=y2018Q4; %
 
[PCC, AUC, PG,BS, KS, H] = LR(X1,X2,X3,X4,y1,y2,y3,y4);
 
results_LR2018_AdaSyn = [PCC, AUC, PG,BS, KS, H]
 
[PCC, AUC, PG,BS, KS, H] = FNNnew(X1,X2,X3,X4,y1,y2,y3,y4);
 
results_FNN2018_AdaSyn = [PCC, AUC, PG,BS, KS, H]
 
[PCC, AUC, PG,BS, KS, H] = LinearSVM(X1,X2,X3,X4,y1,y2,y3,y4);
 
results_linearSVM2018_AdaSyn = [PCC, AUC, PG,BS, KS, H]
 
[PCC, AUC, PG,BS, KS, H] = rbfSVM(X1,X2,X3,X4,y1,y2,y3,y4);
 
results_RbfSVM2018_AdaSyn = [PCC, AUC, PG,BS, KS, H]
 
 
 


