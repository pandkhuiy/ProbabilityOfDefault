%%
load('Year2010.mat'); %

temp=(avgIV>0.08).*(avgIV<inf);

temp2=(find(temp==1)+8)';

X1=normalize(X_WoE2010Q1(:,[1,2,3,4,5,6,7,8,[temp2]])); %

X2=normalize(X_WoE2010Q2(:,[1,2,3,4,5,6,7,8,[temp2]])); %

X3=normalize(X_WoE2010Q3(:,[1,2,3,4,5,6,7,8,[temp2]])); %

X4=normalize(X_WoE2010Q4(:,[1,2,3,4,5,6,7,8,[temp2]])); %

y1=y2010Q1; %

y2=y2010Q2; %

y3=y2010Q3; %

y4=y2010Q4; %

[PCC, AUC, PG,BS, KS, H] = LinearSVM(X1,X2,X3,X4,y1,y2,y3,y4);

2010
[PCC, AUC, PG,BS, KS, H]