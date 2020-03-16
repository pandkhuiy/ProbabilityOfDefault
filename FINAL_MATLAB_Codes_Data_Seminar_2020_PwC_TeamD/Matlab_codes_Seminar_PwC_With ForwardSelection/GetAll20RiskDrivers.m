function [X_WoE, InfoValue, y] = GetAll20RiskDrivers(acq_file,y)
%% This function extracts 20 features from the acquisition file and encodes them with Weights-of-Evidence.
% Input: 
% acq_filq:  the txt file containing features obtained from the fannie mae site of a given year and quarter.
% y:   the default vector within 12 months of a given year and quarter
% This function also takes into account the fact that file 2018Q1 has one more loan in 
% only works for all quarters except 2018Q1.

% This index is found using the Getdata function (for computing the default
% flag vector) FOR the data set of 2018Q1. This is because this is the only
% data set that has one more loan redundant in the acquisition file compared to the
% performance. If we not treat this, the observations of the
% acquisition data set and the performance dataset are misalligned.
FinalIndex = 295304; 

% Remove the reduantant loan obervation in the acq. file
if strcmp(acq_file, 'Acquisition_2018Q1.txt') 
   T = readtable(acq_file,'ReadVariableNames',false,'Delimiter','|');
   Xt = table2cell(T); 
   Xt(FinalIndex,:) = [];
else
   Xtable = readtable(acq_file,'ReadVariableNames',false,'Delimiter','|');
   %Xtable =  Xtable(:,[4 5 6 9 11 12 13 17 2 14 15 16 18 24 25]);
   Xt = table2cell(Xtable);
end

% Process the numerical and categorical variables (imputing/ make less or
% more levels for the categorical variables. Note that the variables are cell arrays, for example X19 is a
% cell array.

% Transform the state variable X19 in four categories: West, MidWest, South and NorthEast.
X19 = ClusterStates(Xt(:,19));

% We deal with blank values by making the numeric variable X21 a categoric
% variable with 6 levels.
X21 = PMIP(Xt(:,21));

% We deal with blank values by making the numeric variables X13 and X23 a categoric
% variable with 6 levels.
X13 = CSAO(Xt(:,13));
X23 = CSAO(Xt(:,23));

% We deal with blank values by making the numeric variables X12 a categoric
% variable with 7 levels.
X12 = DTIR(Xt(:,12));

% We deal with blank values by imputing blank observation with 97% for
% variable X9 and 200% for LTV and CLTV variables.
X9  = LTV( Xt(:,9) );
X10 = CLTV( Xt(:,10) );

% Impute the missing values of the OIR variable X4 with the mean value:
X4 = ImputeOIR(Xt(:,4));

% the 20 variables we use prior to forward selection. Some variables might
% be correlated. Therefore, we do not use them in the forward selection method with logistic regression.
% Note that is not mentioned that we do not use zip code variable 
% due to the large different levels this categorical
% variable has.
% -----  Numeric variables: X4, X5, X6, X9, X10, X11, X17  ( X12, X13, X21 and X23 become categorical because we dont want to mitigate the blank values in these variables )
% -----  categorical variables: X2 (R/C/B), X14 (Y/N/U), X15 (P/C/R/U), X16 (SF/CO/CP/MH/PU) , X18 (P/S/I/U), X24 (1/2/3/NaN) , X25 (Y/N) 
%                               X12 (1/2/3/4/5/6/Blank), X13 (Very Poor/Fair/Good/Very Good/Exceptional/Blank), X19 (WEST/MIDWEST/SOUTH/NORTHEAST),
%                               X21 (1/2/3/4/5/Blank) , X22(FRM), X23 (Very Poor/Fair/Good/Very Good/Exceptional/Blank). 

% Number of features: 7 numeric + 13 categorical = 20
Numerical = 7;
Categorical = 13;

k =  Numerical + Categorical;
X_WoE = zeros(size(Xt,1),k);

InfoValue = zeros(Categorical,1);

IV2  = zeros(3,1);  IV12 = zeros(7,1);
IV14 = zeros(2,1);  IV13 = zeros(6,1);
IV15 = zeros(4,1);  IV19 = zeros(4,1);
IV16 = zeros(5,1);  IV21 = zeros(6,1);
IV18 = zeros(4,1);  IV22 = zeros(1,1);
IV24 = zeros(4,1);  IV23 = zeros(6,1);
IV25 = zeros(2,1);  
 

sumGood = sum(y); sumBad = numel(y)- sumGood;

% Variable 2.
r = strcmp( Xt(:,2) , 'R' );
c = strcmp( Xt(:,2) , 'C' );
b = strcmp( Xt(:,2) , 'B' );

ysubr = y(r); ysubc = y(c); ysubb = y(b);

if sum(r) == 0 || sum(ysubr) == 0
    WoE_2r = 0;

    IV2(1) = 0;
else
    Distr_bad = sum(ysubr)/sumGood; 
    Distr_good = ( numel(ysubr)- sum(ysubr) )/sumBad;
    WoE_2r = ( log(Distr_good/Distr_bad) )*100;

    IV2(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(c) == 0 || sum(ysubc) == 0
    WoE_2c = 0;

    IV2(2) = 0;
else
    Distr_bad = sum(ysubc)/sumGood; 
    Distr_good = ( numel(ysubc)- sum(ysubc) )/sumBad;
    WoE_2c = ( log(Distr_good/Distr_bad) )*100;

    IV2(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(b) == 0 || sum(ysubb) == 0
    WoE_2b = 0;

    IV2(3) = 0;
else
    Distr_bad = sum(ysubb)/sumGood; 
    Distr_good = ( numel(ysubb)- sum(ysubb) )/sumBad;
    WoE_2b = ( log(Distr_good/Distr_bad) )*100;

    IV2(3) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end
InfoValue(1) = sum(IV2);

% Variable 14.
Y = strcmp( Xt(:,14) , 'Y' );
N = strcmp( Xt(:,14) , 'N' );
U = strcmp( Xt(:,14) , 'U' );

ysubr = y(Y); ysubc = y(N); ysubu = y(U);

if sum(Y) == 0 || sum(ysubr) == 0
    WoE_14Y = 0;

    IV14(1) = 0;
else
    Distr_bad = sum(ysubr)/sumGood; 
    Distr_good = ( numel(ysubr)- sum(ysubr) )/sumBad;
    WoE_14Y = ( log(Distr_good/Distr_bad) )*100;

    IV14(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(N) == 0 || sum(ysubc) == 0
    WoE_14N = 0;

    IV14(2) = 0;
else
    Distr_bad = sum(ysubc)/sumGood; 
    Distr_good = ( numel(ysubc)- sum(ysubc) )/sumBad;
    WoE_14N = ( log(Distr_good/Distr_bad) )*100;

    IV14(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(U) == 0 || sum(ysubu) == 0
    WoE_14U = 0;

    IV14(3) = 0;
else
    Distr_badu = sum(ysubu)/sumGood; 
    Distr_goodu = ( numel(ysubu)- sum(ysubu) )/sumBad;
    WoE_14U = ( log(Distr_goodu/Distr_badu) )*100;

    IV14(3) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(2) = sum(IV14);


% Variable 15.
var15P = strcmp( Xt(:,15) , 'P' );
var15C = strcmp( Xt(:,15) , 'C' );
var15R = strcmp( Xt(:,15) , 'R' );
var15U = strcmp( Xt(:,15) , 'U' );

ysub1 = y(var15P); ysub2 = y(var15C); ysub3 = y(var15R); ysub4 = y(var15U);

if sum(var15P) == 0 || sum(ysub1) == 0 
    WoE_15P = 0;

    IV15(1) = 0;
else
    Distr_bad = sum(ysub1)/sumGood; 
    Distr_good = ( numel(ysub1)- sum(ysub1) )/sumBad;
    WoE_15P = ( log(Distr_good/Distr_bad) )*100;

    IV15(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var15C) == 0 || sum(ysub2) == 0 
    WoE_15C = 0;

    IV15(2) = 0;
else
    Distr_bad = sum(ysub2)/sumGood; 
    Distr_good = ( numel(ysub2)- sum(ysub2) )/sumBad;
    WoE_15C = ( log(Distr_good/Distr_bad) )*100;

    IV15(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var15C) == 0 || sum(ysub3) == 0 
    WoE_15R = 0;

    IV15(3) = 0;
else
    Distr_bad = sum(ysub3)/sumGood; 
    Distr_good = ( numel(ysub3)- sum(ysub3) )/sumBad;
    WoE_15R = ( log(Distr_good/Distr_bad) )*100;

    IV15(3) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var15U) == 0 || sum(ysub4) == 0 || ( numel(ysub4)- sum(ysub4) == 0 )
    WoE_15U = 0;

    IV15(4) = 0;
else
    Distr_bad = sum(ysub4)/sumGood; 
    Distr_good = ( numel(ysub4)- sum(ysub4) )/sumBad;
    WoE_15U = ( log(Distr_good/Distr_bad) )*100;

    IV15(4) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(3) = sum(IV15);

% Variable 16.
var16SF = strcmp( Xt(:,16) , 'SF' );
var16CO = strcmp( Xt(:,16) , 'CO' );
var16CP = strcmp( Xt(:,16) , 'CP' );
var16MH = strcmp( Xt(:,16) , 'MH' );
var16PU = strcmp( Xt(:,16) , 'PU' );

ysub1 = y(var16SF); ysub2 = y(var16CO); ysub3 = y(var16CP); ysub4 = y(var16MH);ysub5 = y(var16PU);

if sum(var16SF) == 0 || sum(ysub1) == 0 
    WoE_16SF = 0;

    IV16(1) = 0;
else
    Distr_bad = sum(ysub1)/sumGood; 
    Distr_good = ( numel(ysub1)- sum(ysub1) )/sumBad;
    WoE_16SF = ( log(Distr_good/Distr_bad) )*100;

    IV16(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var16CO) == 0 || sum(ysub2) == 0 
    WoE_16CO = 0;

    IV16(2) = 0;
else
    Distr_bad = sum(ysub2)/sumGood; 
    Distr_good = ( numel(ysub2)- sum(ysub2) )/sumBad;
    WoE_16CO = ( log(Distr_good/Distr_bad) )*100;

    IV16(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var16CP) == 0 || sum(ysub3) == 0 
    WoE_16CP = 0;

    IV16(3) = 0;
else
    Distr_bad = sum(ysub3)/sumGood; 
    Distr_good = ( numel(ysub3)- sum(ysub3) )/sumBad;
    WoE_16CP = ( log(Distr_good/Distr_bad) )*100;

    IV16(3) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var16MH) == 0 || sum(ysub4) == 0 
    WoE_16MH = 0;

    IV16(4) = 0;
else
    Distr_bad = sum(ysub4)/sumGood; 
    Distr_good = ( numel(ysub4)- sum(ysub4) )/sumBad;
    WoE_16MH = ( log(Distr_good/Distr_bad) )*100;

    IV16(4) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var16PU) == 0 || sum(ysub5) == 0 
    WoE_16PU = 0;

    IV16(5) = 0;
else
    Distr_bad = sum(ysub5)/sumGood; 
    Distr_good = ( numel(ysub5)- sum(ysub5) )/sumBad;
    WoE_16PU = ( log(Distr_good/Distr_bad) )*100;

    IV16(5) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(4) = sum(IV16);



% Variable 18.
var18P = strcmp( Xt(:,18) , 'P' );
var18S = strcmp( Xt(:,18) , 'S' );
var18I = strcmp( Xt(:,18) , 'I' );
var18U = strcmp( Xt(:,18) , 'U' );

ysub1 = y(var18P); ysub2 = y(var18S); ysub3 = y(var18I); ysub4 = y(var18U);

if sum(var18P) == 0 || sum(ysub1) == 0 
    WoE_18P = 0;

    IV18(1) = 0;
else
    Distr_bad = sum(ysub1)/sumGood; 
    Distr_good = ( numel(ysub1)- sum(ysub1) )/sumBad;
    WoE_18P = ( log(Distr_good/Distr_bad) )*100;

    IV18(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var18S) == 0 || sum(ysub2) == 0 
    WoE_18S = 0;

    IV18(2) = 0;
else
    Distr_bad = sum(ysub2)/sumGood; 
    Distr_good = ( numel(ysub2)- sum(ysub2) )/sumBad;
    WoE_18S = ( log(Distr_good/Distr_bad) )*100;

    IV18(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var18I) == 0 || sum(ysub3) == 0 
    WoE_18I = 0;

    IV18(3) = 0;
else
    Distr_bad = sum(ysub3)/sumGood; 
    Distr_good = ( numel(ysub3)- sum(ysub3) )/sumBad;
    WoE_18I = ( log(Distr_good/Distr_bad) )*100;

    IV18(3) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var18U) == 0 || sum(ysub4) == 0 
    WoE_18U = 0;

    IV18(4) = 0;
else
    Distr_bad = sum(ysub4)/sumGood; 
    Distr_good = ( numel(ysub4)- sum(ysub4) )/sumBad;
    WoE_18U = ( log(Distr_good/Distr_bad) )*100;

    IV18(4) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(5) = sum(IV18);


% Variable 24.
var241    =   strcmp( Xt(:,24) , '1' );
var242    =   strcmp( Xt(:,24) , '2' );
var243    =   strcmp( Xt(:,24) , '3' );
var24nan  =   cellfun(@isnan,Xt(:,24));

ysub1 = y(var241); ysub2 = y(var242); ysub3 = y(var243); ysub4 = y(var24nan);

if sum(var241) == 0 || sum(ysub1) == 0 
    WoE_241 = 0;

    IV24(1) = 0;
else
    Distr_bad = sum(ysub1)/sumGood; 
    Distr_good = ( numel(ysub1)- sum(ysub1) )/sumBad;
    WoE_241 = ( log(Distr_good/Distr_bad) )*100;

    IV24(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var242) == 0 || sum(ysub2) == 0 
    WoE_242 = 0;

    IV24(2) = 0;
else
    Distr_bad = sum(ysub2)/sumGood; 
    Distr_good = ( numel(ysub2)- sum(ysub2) )/sumBad;
    WoE_242 = ( log(Distr_good/Distr_bad) )*100;

    IV24(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var243) == 0 || sum(ysub3) == 0 
    WoE_243 = 0;

    IV24(3) = 0;
else
    Distr_bad = sum(ysub3)/sumGood; 
    Distr_good = ( numel(ysub3)- sum(ysub3) )/sumBad;
    WoE_243 = ( log(Distr_good/Distr_bad) )*100;

    IV24(3) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var24nan) == 0 || sum(ysub4) == 0 
    WoE_24nan = 0;

    IV24(4) = 0;
else
    Distr_bad = sum(ysub4)/sumGood; 
    Distr_good = ( numel(ysub4)- sum(ysub4) )/sumBad;
    WoE_24nan = ( log(Distr_good/Distr_bad) )*100;

    IV24(4) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(6) = sum(IV24);

% Variable 25.
Y = strcmp( Xt(:,25) , 'Y' );
N = strcmp( Xt(:,25) , 'N' );

ysubr = y(Y); ysubc = y(N);

if sum(Y) == 0 || sum(ysubr) == 0 
    WoE_25Y = 0;

    IV25(1) = 0;
else
    Distr_bad = sum(ysubr)/sumGood; 
    Distr_good = ( numel(ysubr)- sum(ysubr) )/sumBad;
    WoE_25Y = ( log(Distr_good/Distr_bad) )*100;

    IV25(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(N) == 0 || sum(ysubc) == 0 
    WoE_25N = 0;

    IV25(2) = 0;
else
    Distr_bad = sum(ysubc)/sumGood; 
    Distr_good = ( numel(ysubc)- sum(ysubc) )/sumBad;
    WoE_25N = ( log(Distr_good/Distr_bad) )*100;

    IV25(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(7) = sum(IV25);


% Variable 12.
var12_1 = strcmp( X12, '1' );
var12_2 = strcmp( X12, '2' );
var12_3 = strcmp( X12, '3' );
var12_4 = strcmp( X12, '4' );
var12_5 = strcmp( X12, '5' );
var12_6 = strcmp( X12, '6' );
var12_blank = strcmp( X12, 'Blank' );

ysub1 = y(var12_1); ysub2 = y(var12_2); ysub3 = y(var12_3); ysub4 = y(var12_4); ysub5 = y(var12_5); ysub6 = y(var12_6); ysub7 = y(var12_blank);

if sum(var12_1) == 0 || sum(ysub1) == 0 
    WoE_12_1 = 0;

    IV12(1) = 0;
else
    Distr_bad = sum(ysub1)/sumGood; 
    Distr_good = ( numel(ysub1)- sum(ysub1) )/sumBad;
    WoE_12_1 = ( log(Distr_good/Distr_bad) )*100;

    IV12(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var12_2) == 0 || sum(ysub2) == 0 
    WoE_12_2 = 0;

    IV12(2) = 0;
else
    Distr_bad = sum(ysub2)/sumGood; 
    Distr_good = ( numel(ysub2)- sum(ysub2) )/sumBad;
    WoE_12_2 = ( log(Distr_good/Distr_bad) )*100;

    IV12(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var12_3) == 0 || sum(ysub3) == 0 
    WoE_12_3 = 0;

    IV12(3) = 0;
else
    Distr_bad = sum(ysub3)/sumGood; 
    Distr_good = ( numel(ysub3)- sum(ysub3) )/sumBad;
    WoE_12_3 = ( log(Distr_good/Distr_bad) )*100;

    IV12(3) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var12_4) == 0 || sum(ysub4) == 0 
    WoE_12_4 = 0;

    IV12(4) = 0;
else
    Distr_bad = sum(ysub4)/sumGood; 
    Distr_good = ( numel(ysub4)- sum(ysub4) )/sumBad;
    WoE_12_4 = ( log(Distr_good/Distr_bad) )*100;

    IV12(4) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var12_5) == 0 || sum(ysub5) == 0 
    WoE_12_5 = 0;

    IV12(5) = 0;
else
    Distr_bad = sum(ysub5)/sumGood; 
    Distr_good = ( numel(ysub5)- sum(ysub5) )/sumBad;
    WoE_12_5 = ( log(Distr_good/Distr_bad) )*100;

    IV12(5) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var12_6) == 0 || sum(ysub6) == 0 
    WoE_12_6 = 0;

    IV12(6) = 0;
else
    Distr_bad = sum(ysub6)/sumGood; 
    Distr_good = ( numel(ysub6)- sum(ysub6) )/sumBad;
    WoE_12_6 = ( log(Distr_good/Distr_bad) )*100;

    IV12(6) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var12_blank) == 0 || sum(ysub7) == 0 
    WoE_12_blank = 0;

    IV12(7) = 0;
else
    Distr_bad = sum(ysub7)/sumGood; 
    Distr_good = ( numel(ysub7)- sum(ysub7) )/sumBad;
    WoE_12_blank = ( log(Distr_good/Distr_bad) )*100;

    IV12(7) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(8) = sum(IV12);


% Variable 13
var13_1 = strcmp( X13, 'Very Poor' );
var13_2 = strcmp( X13, 'Fair' );
var13_3 = strcmp( X13, 'Good' );
var13_4 = strcmp( X13, 'Very Good' );
var13_5 = strcmp( X13, 'Exceptional' );
var13_6 = strcmp( X13, 'Blank' );

ysub1 = y(var13_1); ysub2 = y(var13_2); ysub3 = y(var13_3); ysub4 = y(var13_4); ysub5 = y(var13_5); ysub6 = y(var13_6); 

if sum(var13_1) == 0 ||  sum(ysub1) == 0 
    WoE_13_1 = 0;

    IV13(1) = 0;
else
    Distr_bad = sum(ysub1)/sumGood; 
    Distr_good = ( numel(ysub1)- sum(ysub1) )/sumBad;
    WoE_13_1 = ( log(Distr_good/Distr_bad) )*100;

    IV13(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var13_2) == 0 || sum(ysub2) == 0 
    WoE_13_2 = 0;

    IV13(2) = 0;
else
    Distr_bad = sum(ysub2)/sumGood; 
    Distr_good = ( numel(ysub2)- sum(ysub2) )/sumBad;
    WoE_13_2 = ( log(Distr_good/Distr_bad) )*100;

    IV13(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var13_3) == 0 || sum(ysub3) == 0 
    WoE_13_3 = 0;

    IV13(3) = 0;
else
    Distr_bad = sum(ysub3)/sumGood; 
    Distr_good = ( numel(ysub3)- sum(ysub3) )/sumBad;
    WoE_13_3 = ( log(Distr_good/Distr_bad) )*100;

    IV13(3) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var13_4) == 0 || sum(ysub4) == 0 
    WoE_13_4 = 0;

    IV13(4) = 0;
else
    Distr_bad = sum(ysub4)/sumGood; 
    Distr_good = ( numel(ysub4)- sum(ysub4) )/sumBad;
    WoE_13_4 = ( log(Distr_good/Distr_bad) )*100;

    IV13(4) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var13_5) == 0 || sum(ysub5) == 0 
    WoE_13_5 = 0;

    IV13(5) = 0;
else
    Distr_bad = sum(ysub5)/sumGood; 
    Distr_good = ( numel(ysub5)- sum(ysub5) )/sumBad;
    WoE_13_5 = ( log(Distr_good/Distr_bad) )*100;

    IV13(5) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var13_6) == 0  || sum(ysub6) == 0 
    WoE_13_blank = 0;

    IV13(6) = 0;
else
    Distr_bad = sum(ysub6)/sumGood; 
    Distr_good = ( numel(ysub6)- sum(ysub6) )/sumBad;
    WoE_13_blank = ( log(Distr_good/Distr_bad) )*100;

    IV13(6) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(9) = sum(IV13);



% Variable 19.
var19West = strcmp( X19 , 'WEST' );
var19MidWest = strcmp( X19 , 'MIDWEST' );
var19South = strcmp( X19 , 'SOUTH' );
var19NorthEast = strcmp( X19 , 'NORTHEAST' );

ysub1 = y(var19West); ysub2 = y(var19MidWest); ysub3 = y(var19South); ysub4 = y(var19NorthEast);
 
if sum(var19West) == 0 || sum(ysub1) == 0 
    WoE_19W = 0;

    IV19(1) = 0;
else
    Distr_bad = sum(ysub1)/sumGood; 
    Distr_good = ( numel(ysub1)- sum(ysub1) )/sumBad;
    WoE_19W = ( log(Distr_good/Distr_bad) )*100;

    IV19(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var19MidWest) == 0 || sum(ysub2) == 0 
    WoE_19MW = 0;

    IV19(2) = 0;
else
    Distr_bad = sum(ysub2)/sumGood; 
    Distr_good = ( numel(ysub2)- sum(ysub2) )/sumBad;
    WoE_19MW = ( log(Distr_good/Distr_bad) )*100;

    IV19(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var19South) == 0  || sum(ysub3) == 0 
    WoE_19S = 0;

    IV19(3) = 0;
else
    Distr_bad = sum(ysub3)/sumGood; 
    Distr_good = ( numel(ysub3)- sum(ysub3) )/sumBad;
    WoE_19S = ( log(Distr_good/Distr_bad) )*100;

    IV19(3) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var19NorthEast) == 0  || sum(ysub4) == 0 
    WoE_19NE = 0;

    IV19(4) = 0;
else
    Distr_bad = sum(ysub4)/sumGood; 
    Distr_good = ( numel(ysub4)- sum(ysub4) )/sumBad;
    WoE_19NE = ( log(Distr_good/Distr_bad) )*100;

    IV19(4) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(10) = sum(IV19);

    
% Variable 21.
var21_1 = strcmp( X21, '1' );
var21_2 = strcmp( X21, '2' );
var21_3 = strcmp( X21, '3' );
var21_4 = strcmp( X21, '4' );
var21_5 = strcmp( X21, '5' );
var21_blank = strcmp( X21, 'Blank' );

ysub1 = y(var21_1); ysub2 = y(var21_2); ysub3 = y(var21_3); ysub4 = y(var21_4); ysub5 = y(var21_5); ysub6 = y(var21_blank);

if sum(var21_1) == 0  || sum(ysub1) == 0 
    WoE_21_1 = 0;

    IV21(1) = 0;
else
    Distr_bad = sum(ysub1)/sumGood; 
    Distr_good = ( numel(ysub1)- sum(ysub1) )/sumBad;
    WoE_21_1 = ( log(Distr_good/Distr_bad) )*100;

    IV21(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var21_2) == 0  || sum(ysub2) == 0 
    WoE_21_2 = 0;

    IV21(2) = 0;
else
    Distr_bad = sum(ysub2)/sumGood; 
    Distr_good = ( numel(ysub2)- sum(ysub2) )/sumBad;
    WoE_21_2 = ( log(Distr_good/Distr_bad) )*100;

    IV21(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var21_3) == 0 || sum(ysub3) == 0 
    WoE_21_3 = 0;

    IV21(3) = 0;
else
    Distr_bad = sum(ysub3)/sumGood; 
    Distr_good = ( numel(ysub3)- sum(ysub3) )/sumBad;
    WoE_21_3 = ( log(Distr_good/Distr_bad) )*100;

    IV21(3) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var21_4) == 0  || sum(ysub4) == 0 
    WoE_21_4 = 0;

    IV21(4) = 0;
else
    Distr_bad = sum(ysub4)/sumGood; 
    Distr_good = ( numel(ysub4)- sum(ysub4) )/sumBad;
    WoE_21_4 = ( log(Distr_good/Distr_bad) )*100;

    IV21(4) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var21_5) == 0  || sum(ysub5) == 0 
    WoE_21_5 = 0;

    IV21(5) = 0;
else
    Distr_bad = sum(ysub5)/sumGood; 
    Distr_good = ( numel(ysub5)- sum(ysub5) )/sumBad;
    WoE_21_5 = ( log(Distr_good/Distr_bad) )*100;

    IV21(5) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var21_blank) == 0 ||  sum(ysub6) == 0 
    WoE_21_blank = 0;

    IV21(6) = 0;
else
    Distr_bad = sum(ysub6)/sumGood; 
    Distr_good = ( numel(ysub6)- sum(ysub6) )/sumBad;
    WoE_21_blank = ( log(Distr_good/Distr_bad) )*100;

    IV21(6) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(11) = sum(IV21);


% Variable 22.
var22_1 = strcmp( Xt(:,22), 'FRM' );

ysub1 = y(var22_1);

if sum(var22_1) == 0 || sum(ysub1) == 0 
    WoE_22_1 = 0;

    IV22(1) = 0;
else
    Distr_bad = sum(ysub1)/sumGood; 
    Distr_good = ( numel(ysub1)- sum(ysub1) )/sumBad;
    WoE_22_1 = ( log(Distr_good/Distr_bad) )*100;

    IV22(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(12) = sum(IV22);


% Variable 13
var23_1 = strcmp( X23, 'Very Poor' );
var23_2 = strcmp( X23, 'Fair' );
var23_3 = strcmp( X23, 'Good' );
var23_4 = strcmp( X23, 'Very Good' );
var23_5 = strcmp( X23, 'Exceptional' );
var23_6 = strcmp( X23, 'Blank' );

ysub1 = y(var23_1); ysub2 = y(var23_2); ysub3 = y(var23_3); ysub4 = y(var23_4); ysub5 = y(var23_5); ysub6 = y(var23_6); 

if sum(var23_1) == 0  || sum(ysub1) == 0 
    WoE_23_1 = 0;

    IV23(1) = 0;
else
    Distr_bad = sum(ysub1)/sumGood; 
    Distr_good = ( numel(ysub1)- sum(ysub1) )/sumBad;
    WoE_23_1 = ( log(Distr_good/Distr_bad) )*100;

    IV23(1) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var23_2) == 0 || sum(ysub2) == 0 
    WoE_23_2 = 0;

    IV23(2) = 0;
else
    Distr_bad = sum(ysub2)/sumGood; 
    Distr_good = ( numel(ysub2)- sum(ysub2) )/sumBad;
    WoE_23_2 = ( log(Distr_good/Distr_bad) )*100;

    IV23(2) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var23_3) == 0  || sum(ysub3) == 0 
    WoE_23_3 = 0;

    IV23(3) = 0;
else
    Distr_bad = sum(ysub3)/sumGood; 
    Distr_good = ( numel(ysub3)- sum(ysub3) )/sumBad;
    WoE_23_3 = ( log(Distr_good/Distr_bad) )*100;

    IV23(3) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var23_4) == 0  ||  sum(ysub4) == 0 
    WoE_23_4 = 0;

    IV23(4) = 0;
else
    Distr_bad = sum(ysub4)/sumGood; 
    Distr_good = ( numel(ysub4)- sum(ysub4) )/sumBad;
    WoE_23_4 = ( log(Distr_good/Distr_bad) )*100;

    IV23(4) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var23_5) == 0 || sum(ysub5) == 0 
    WoE_23_5 = 0;

    IV23(5) = 0;
else
    Distr_bad = sum(ysub5)/sumGood; 
    Distr_good = ( numel(ysub5)- sum(ysub5) )/sumBad;
    WoE_23_5 = ( log(Distr_good/Distr_bad) )*100;

    IV23(5) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

if sum(var23_6) == 0  || sum(ysub6) == 0 
    WoE_23_blank = 0;

    IV23(6) = 0;
else
    Distr_bad = sum(ysub6)/sumGood; 
    Distr_good = ( numel(ysub6)- sum(ysub6) )/sumBad;
    WoE_23_blank = ( log(Distr_good/Distr_bad) )*100;

    IV23(6) = (Distr_good - Distr_bad)*log(Distr_good/Distr_bad);
end

InfoValue(13) = sum(IV23);


for i = 1:size(Xt,1)
     % Now, make the X matrix consisting of numeric variables and WoE
     % encoded nominal variables with corresponding InfoValue for each
     % nominal variable indicating the strength of the relat.ship  between
     % default variable y.
   
     % numeric variables    
     X_WoE(i,1) = X4{i};
     X_WoE(i,2) = Xt{i,5};
     X_WoE(i,3) = Xt{i,6};
     X_WoE(i,4) = X9{i};
     X_WoE(i,5) = X10{i};
     X_WoE(i,6) = Xt{i,11};
     X_WoE(i,7) = Xt{i,17};
   
     % variable 2   
     if strcmp( Xt{i,2} , 'R' )
       X_WoE(i,8) = WoE_2r;
     elseif strcmp( Xt{i,2} , 'C' )
       X_WoE(i,8) = WoE_2c; 
     elseif strcmp( Xt{i,2} , 'B' )
       X_WoE(i,8) = WoE_2b;
     end
     
     % variable 14  
     if strcmp( Xt{i,14} , 'Y' )
       X_WoE(i,9) = WoE_14Y;
     elseif strcmp( Xt{i,14} , 'N' )
       X_WoE(i,9) = WoE_14N;
     elseif strcmp( Xt{i,14} , 'U' )
       X_WoE(i,9) =  WoE_14U;
     end

     % variable 15  
     if strcmp( Xt{i,15} , 'P' )
       X_WoE(i,10) = WoE_15P;
     elseif strcmp( Xt{i,15} , 'C' )
       X_WoE(i,10) = WoE_15C; 
     elseif strcmp( Xt{i,15} , 'R' )
       X_WoE(i,10) = WoE_15R;
     elseif strcmp( Xt{i,15} , 'U' )
       X_WoE(i,10) = WoE_15U;
     end
     
     % variable 16  
     if strcmp( Xt{i,16} , 'SF' )
       X_WoE(i,11) = WoE_16SF;
     elseif strcmp( Xt{i,16} , 'CO' )
       X_WoE(i,11) = WoE_16CO; 
     elseif strcmp( Xt{i,16} , 'CP' )
       X_WoE(i,11) = WoE_16CP;
     elseif strcmp( Xt{i,16} , 'MH' )
       X_WoE(i,11) = WoE_16MH;
     elseif strcmp( Xt{i,16} , 'PU' )
       X_WoE(i,11) = WoE_16PU;
     end
     
     % variable 18 
     if strcmp( Xt{i,18} , 'P' )
       X_WoE(i,12) = WoE_18P;
     elseif strcmp( Xt{i,18} , 'S' )
       X_WoE(i,12) = WoE_18S; 
     elseif strcmp( Xt{i,18} , 'I' )
       X_WoE(i,12) = WoE_18I;
     elseif strcmp( Xt{i,18} , 'U' )
       X_WoE(i,12) = WoE_18U;
     end
    
        
     % variable 24
     if strcmp( Xt{i,24} , '1' )
       X_WoE(i,13) = WoE_241;
     elseif strcmp( Xt{i,24} , '2' )
       X_WoE(i,13) = WoE_242; 
     elseif strcmp( Xt{i,24} , '3' )
       X_WoE(i,13) = WoE_243;
     elseif cellfun(@isnan,Xt(i,24))
       X_WoE(i,13) = WoE_24nan;
     end
     
     % variable 25
     if strcmp( Xt{i,25} , 'Y' )
       X_WoE(i,14) = WoE_25Y;
     elseif strcmp( Xt{i,25} , 'N' )
       X_WoE(i,14) = WoE_25N;
     end
     
        
     % variable 12
     if strcmp( X12{i} , '1' )
       X_WoE(i,15) = WoE_12_1;
     elseif strcmp( X12{i} , '2' )
       X_WoE(i,15) = WoE_12_2; 
     elseif strcmp( X12{i} , '3' )
       X_WoE(i,15) = WoE_12_3;
     elseif strcmp( X12{i} , '4' )
       X_WoE(i,15) = WoE_12_4;
     elseif strcmp( X12{i} , '5' )
       X_WoE(i,15) = WoE_12_5;
     elseif strcmp( X12{i} , '6' )
       X_WoE(i,15) = WoE_12_6;
     elseif strcmp( X12{i} , 'Blank' )
       X_WoE(i,15) = WoE_12_blank;
     end
     
     % variable 13
     if strcmp( X13{i} , 'Very Poor' )
       X_WoE(i,16) = WoE_13_1;
     elseif strcmp( X13{i} , 'Fair' )
       X_WoE(i,16) = WoE_13_2; 
     elseif strcmp( X13{i} , 'Good' )
       X_WoE(i,16) = WoE_13_3;
     elseif strcmp( X13{i} , 'Very Good' )
       X_WoE(i,16) = WoE_13_4;
     elseif strcmp( X13{i} , 'Exceptional' )
       X_WoE(i,16) = WoE_13_5;
     elseif strcmp( X13{i} , 'Blank' )
       X_WoE(i,16) = WoE_13_blank;
     end
     
     % variable 19 
     if strcmp( X19{i} , 'WEST' )
       X_WoE(i,17) = WoE_19W;
     elseif strcmp( X19{i} , 'MIDWEST' )
       X_WoE(i,17) = WoE_19MW; 
     elseif strcmp( X19{i} , 'SOUTH' )
       X_WoE(i,17) = WoE_19S;
     elseif strcmp( X19{i} , 'NORTHEAST' )
       X_WoE(i,17) = WoE_19NE;
     end
     
      % variable 21
     if strcmp( X21{i} , '1' )
       X_WoE(i,18) = WoE_21_1;
     elseif strcmp( X21{i} , '2' )
       X_WoE(i,18) = WoE_21_2; 
     elseif strcmp( X21{i} , '3' )
       X_WoE(i,18) = WoE_21_3;
     elseif strcmp( X21{i} , '4' )
       X_WoE(i,18) = WoE_21_4;
     elseif strcmp( X21{i} , '5' )
       X_WoE(i,18) = WoE_21_5;
     elseif strcmp( X21{i} , 'Blank' )
       X_WoE(i,18) = WoE_21_blank;
     end
     
     % variable 22
     if strcmp( Xt{i,22} , 'FRM' )
       X_WoE(i,19) = WoE_22_1;
     end
     
     % variable 23
     if strcmp( X23{i} , 'Very Poor' )
       X_WoE(i,20) = WoE_23_1;
     elseif strcmp( X23{i} , 'Fair' )
       X_WoE(i,20) = WoE_23_2; 
     elseif strcmp( X23{i} , 'Good' )
       X_WoE(i,20) = WoE_23_3;
     elseif strcmp( X23{i} , 'Very Good' )
       X_WoE(i,20) = WoE_23_4;
     elseif strcmp( X23{i} , 'Exceptional' )
       X_WoE(i,20) = WoE_23_5;
     elseif strcmp( X23{i} , 'Blank' )
       X_WoE(i,20) = WoE_23_blank;
     end
end

% Get rid of missing values (Nan's) (not much to lose though).
indices = any(isnan(X_WoE), 2);
X_WoE  = X_WoE(logical(1-indices), :);
y  = y(logical(1-indices), :);

end