function [X_WoE,InfoValue,y] = Get15RiskDrivers(acq_file,y)
%% This function extracts 15 features from the acquisition file and encodes them with Weights-of-Evidence.
% Input: 
% acq_filq:  the txt file containing features obtained from the fannie mae
%            site of a given year and quarter.(This can be downloaded from 
%            the website of Fannie Mae.)
% y:   the default vector within 12 months of a given year and quarter
%      obtained with the GetDefaultVector.m function.
% This function also takes into account the fact that file 2018Q1 has misaligned 
% observations compared with the observations in performance file of 2018Q1.


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

% the 15 variables we use prior to selection with Information Value are
% given below. Also we note that variable 24 will not used in the
% Information Value selection as all the observations of that variables are the same.
% -----  Numeric variables: 4, 5, 6, 9 or 10, 11 , 12, 13 and 17.
% -----  categorical/nominal variables: 2 (R/C/B), 14 (Y/N), 15 (P/C/R/U), 16 (SF/CO/CP/MH/PU) , 18 (P/S/I/U)  , 24 (1/2/3/NaN) , 25 (Y/N) 

Numerical = 8;
Categorical = 7;

k =  Numerical + Categorical;

X_WoE = zeros(size(Xt,1),k );

InfoValue = zeros(7,1);

IV2 = zeros(3,1);
IV14 = zeros(2,1);
IV15 = zeros(4,1);
IV16 = zeros(5,1);
IV18 = zeros(4,1);
IV24 = zeros(4,1);
IV25 = zeros(2,1);

sumGood = sum(y); sumBad = numel(y)- sumGood;

% Variable 2.
     r = strcmp( Xt(:,2) , 'R' );
     c = strcmp( Xt(:,2) , 'C' );
     b = strcmp( Xt(:,2) , 'B' );
     
     ysubr = y(r); ysubc = y(c); ysubb = y(b);
     
     if sum(r) == 0
      WoE_2r = 0;
      
      IV2(1) = 0;
     else
      Distr_goodr = sum(ysubr)/sumGood;
      Distr_badr = ( numel(ysubr)- sum(ysubr) )/sumBad;
      WoE_2r = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV2(1) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(c) == 0
      WoE_2c = 0;
      
      IV2(2) = 0;
     else
      Distr_goodc = sum(ysubc)/sumGood;
      Distr_badc = ( numel(ysubc)- sum(ysubc) )/sumBad;
      WoE_2c = ( log(Distr_goodc/Distr_badc) )*100;
      
      IV2(2) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(b) == 0
      WoE_2b = 0;
      
      IV2(3) = 0;
     else
      Distr_goodb = sum(ysubb)/sumGood;
      Distr_badb = ( numel(ysubb)- sum(ysubb) )/sumBad;
      WoE_2b = ( log(Distr_goodb/Distr_badb) )*100;
      
      IV2(3) = (Distr_goodb - Distr_badb)*log(Distr_goodb/Distr_badb);
     end
     InfoValue(1) = sum(IV2);
     
     % Variable 14.
     Y = strcmp( Xt(:,14) , 'Y' );
     N = strcmp( Xt(:,14) , 'N' );
     
     ysubr = y(Y); ysubc = y(N);
     
     if sum(Y) == 0
      WoE_14Y = 0;
      
      IV14(1) = 0;
     else
      Distr_goodr = sum(ysubr)/sumGood;
      Distr_badr = ( numel(ysubr)- sum(ysubr) )/sumBad;
      WoE_14Y = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV14(1) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(N) == 0
      WoE_14N = 0;
      
      IV14(2) = 0;
     else
      Distr_goodc = sum(ysubc)/sumGood;
      Distr_badc = ( numel(ysubc)- sum(ysubc) )/sumBad;
      WoE_14N = ( log(Distr_goodc/Distr_badc) )*100;
      
      IV14(2) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
    
     InfoValue(2) = sum(IV14);
     
      % Variable 15.
     var15P = strcmp( Xt(:,15) , 'P' );
     var15C = strcmp( Xt(:,15) , 'C' );
     var15R = strcmp( Xt(:,15) , 'R' );
     var15U = strcmp( Xt(:,15) , 'U' );
     
     ysub1 = y(var15P); ysub2 = y(var15C); ysub3 = y(var15R); ysub4 = y(var15U);
     
     if sum(var15P) == 0
      WoE_15P = 0;
      
      IV15(1) = 0;
     else
      Distr_goodr = sum(ysub1)/sumGood;
      Distr_badr = ( numel(ysub1)- sum(ysub1) )/sumBad;
      WoE_15P = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV15(1) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(var15C) == 0
      WoE_15C = 0;
      
      IV15(2) = 0;
     else
      Distr_goodr = sum(ysub2)/sumGood;
      Distr_badr = ( numel(ysub2)- sum(ysub2) )/sumBad;
      WoE_15C = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV15(2) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end

     if sum(var15C) == 0
      WoE_15R = 0;
      
      IV15(3) = 0;
     else
      Distr_goodr = sum(ysub3)/sumGood;
      Distr_badr = ( numel(ysub3)- sum(ysub3) )/sumBad;
      WoE_15R = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV15(3) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(var15U) == 0
      WoE_15U = 0;
      
      IV15(4) = 0;
     else
      Distr_goodc = sum(ysub4)/sumGood;
      Distr_badc = ( numel(ysub4)- sum(ysub4) )/sumBad;
      WoE_15U = ( log(Distr_goodc/Distr_badc) )*100;
      
      IV15(4) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
    
     InfoValue(3) = sum(IV15);
     
          % Variable 16.
     var16SF = strcmp( Xt(:,16) , 'SF' );
     var16CO = strcmp( Xt(:,16) , 'CO' );
     var16CP = strcmp( Xt(:,16) , 'CP' );
     var16MH = strcmp( Xt(:,16) , 'MH' );
     var16PU = strcmp( Xt(:,16) , 'PU' );
     
     ysub1 = y(var16SF); ysub2 = y(var16CO); ysub3 = y(var16CP); ysub4 = y(var16MH);ysub5 = y(var16PU);
     
     if sum(var16SF) == 0
      WoE_16SF = 0;
      
      IV16(1) = 0;
     else
      Distr_goodr = sum(ysub1)/sumGood;
      Distr_badr = ( numel(ysub1)- sum(ysub1) )/sumBad;
      WoE_16SF = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV16(1) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(var16CO) == 0
      WoE_16CO = 0;
      
      IV16(2) = 0;
     else
      Distr_goodr = sum(ysub2)/sumGood;
      Distr_badr = ( numel(ysub2)- sum(ysub2) )/sumBad;
      WoE_16CO = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV16(2) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end

     if sum(var16CP) == 0
      WoE_16CP = 0;
      
      IV16(3) = 0;
     else
      Distr_goodr = sum(ysub3)/sumGood;
      Distr_badr = ( numel(ysub3)- sum(ysub3) )/sumBad;
      WoE_16CP = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV16(3) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(var16MH) == 0
      WoE_16MH = 0;
      
      IV16(4) = 0;
     else
      Distr_goodc = sum(ysub4)/sumGood;
      Distr_badc = ( numel(ysub4)- sum(ysub4) )/sumBad;
      WoE_16MH = ( log(Distr_goodc/Distr_badc) )*100;
      
      IV16(4) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(var16PU) == 0
      WoE_16PU = 0;
      
      IV16(5) = 0;
     else
      Distr_goodc = sum(ysub5)/sumGood;
      Distr_badc = ( numel(ysub5)- sum(ysub5) )/sumBad;
      WoE_16PU = ( log(Distr_goodc/Distr_badc) )*100;
      
      IV16(5) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
    
     InfoValue(4) = sum(IV16);
     
     
     
     % Variable 18.
     var18P = strcmp( Xt(:,18) , 'P' );
     var18S = strcmp( Xt(:,18) , 'S' );
     var18I = strcmp( Xt(:,18) , 'I' );
     var18U = strcmp( Xt(:,18) , 'U' );
     
     ysub1 = y(var18P); ysub2 = y(var18S); ysub3 = y(var18I); ysub4 = y(var18U);
     
     if sum(var18P) == 0
      WoE_18P = 0;
      
      IV18(1) = 0;
     else
      Distr_goodr = sum(ysub1)/sumGood;
      Distr_badr = ( numel(ysub1)- sum(ysub1) )/sumBad;
      WoE_18P = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV18(1) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(var18S) == 0
      WoE_18S = 0;
      
      IV18(2) = 0;
     else
      Distr_goodr = sum(ysub2)/sumGood;
      Distr_badr = ( numel(ysub2)- sum(ysub2) )/sumBad;
      WoE_18S = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV18(2) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end

     if sum(var18I) == 0
      WoE_18I = 0;
      
      IV18(3) = 0;
     else
      Distr_goodr = sum(ysub3)/sumGood;
      Distr_badr = ( numel(ysub3)- sum(ysub3) )/sumBad;
      WoE_18I = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV18(3) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(var18U) == 0
      WoE_18U = 0;
      
      IV18(4) = 0;
     else
      Distr_goodc = sum(ysub4)/sumGood;
      Distr_badc = ( numel(ysub4)- sum(ysub4) )/sumBad;
      WoE_18U = ( log(Distr_goodc/Distr_badc) )*100;
      
      IV18(4) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
    
     InfoValue(5) = sum(IV18);
     
     % Variable 24.
     var241 = strcmp( Xt(:,24) , '1' );
     var242 = strcmp( Xt(:,24) , '2' );
     var243 = strcmp( Xt(:,24) , '3' );
     var24nan = strcmp( Xt(:,24) , '' );
     
     ysub1 = y(var241); ysub2 = y(var242); ysub3 = y(var243); ysub4 = y(var24nan);
     
     if sum(var241) == 0
      WoE_241 = 0;
      
      IV24(1) = 0;
     else
      Distr_goodr = sum(ysub1)/sumGood;
      Distr_badr = ( numel(ysub1)- sum(ysub1) )/sumBad;
      WoE_241 = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV24(1) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(var242) == 0
      WoE_242 = 0;
      
      IV24(2) = 0;
     else
      Distr_goodr = sum(ysub2)/sumGood;
      Distr_badr = ( numel(ysub2)- sum(ysub2) )/sumBad;
      WoE_242 = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV24(2) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end

     if sum(var243) == 0
      WoE_243 = 0;
      
      IV24(3) = 0;
     else
      Distr_goodr = sum(ysub3)/sumGood;
      Distr_badr = ( numel(ysub3)- sum(ysub3) )/sumBad;
      WoE_243 = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV24(3) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(var24nan) == 0
      WoE_24nan = 0;
      
      IV24(4) = 0;
     else
      Distr_goodc = sum(ysub4)/sumGood;
      Distr_badc = ( numel(ysub4)- sum(ysub4) )/sumBad;
      WoE_24nan = ( log(Distr_goodc/Distr_badc) )*100;
      
      IV24(4) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
    
     InfoValue(6) = sum(IV24);
     
     % Variable 25.
     Y = strcmp( Xt(:,25) , 'Y' );
     N = strcmp( Xt(:,25) , 'N' );
     
     ysubr = y(Y); ysubc = y(N);
     
     if sum(Y) == 0
      WoE_25Y = 0;
      
      IV25(1) = 0;
     else
      Distr_goodr = sum(ysubr)/sumGood;
      Distr_badr = ( numel(ysubr)- sum(ysubr) )/sumBad;
      WoE_25Y = ( log(Distr_goodr/Distr_badr) )*100;
      
      IV25(1) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
     
     if sum(N) == 0
      WoE_25N = 0;
      
      IV25(2) = 0;
     else
      Distr_goodc = sum(ysubc)/sumGood;
      Distr_badc = ( numel(ysubc)- sum(ysubc) )/sumBad;
      WoE_25N = ( log(Distr_goodc/Distr_badc) )*100;
      
      IV25(2) = (Distr_goodr - Distr_badr)*log(Distr_goodr/Distr_badr);
     end
    
     InfoValue(7) = sum(IV25);
    

for i = 1:size(Xt,1)
     % Now, make the X matrix consisting of numeric variables and WoE
     % encoded nominal variables with corresponding InfoValue for each
     % nominal variable indicating the strength of the relat.ship  between
     % default variable y.
   
     % numeric variables    
     X_WoE(i,1) = Xt{i,4};
     X_WoE(i,2) = Xt{i,5};
     X_WoE(i,3) = Xt{i,6};
     X_WoE(i,4) = Xt{i,9};
     X_WoE(i,5) = Xt{i,11};
     X_WoE(i,6) = Xt{i,12};
     X_WoE(i,7) = Xt{i,13};
     X_WoE(i,8) = Xt{i,17};
   
     % variable 2   
     if strcmp( Xt{i,2} , 'R' )
       X_WoE(i,9) = WoE_2r;
     elseif strcmp( Xt{i,2} , 'C' )
       X_WoE(i,9) = WoE_2c; 
     elseif strcmp( Xt{i,2} , 'B' )
       X_WoE(i,9) = WoE_2b;
     end
     
     % variable 14  
     if strcmp( Xt{i,14} , 'Y' )
       X_WoE(i,10) = WoE_14Y;
     elseif strcmp( Xt{i,14} , 'N' )
       X_WoE(i,10) = WoE_14N; 
     end

     % variable 15  
     if strcmp( Xt{i,15} , 'P' )
       X_WoE(i,11) = WoE_15P;
     elseif strcmp( Xt{i,15} , 'C' )
       X_WoE(i,11) = WoE_15C; 
     elseif strcmp( Xt{i,15} , 'R' )
       X_WoE(i,11) = WoE_15R;
     elseif strcmp( Xt{i,15} , 'U' )
       X_WoE(i,11) = WoE_15U;
     end
     
     % variable 16  
     if strcmp( Xt{i,16} , 'SF' )
       X_WoE(i,12) = WoE_16SF;
     elseif strcmp( Xt{i,16} , 'CO' )
       X_WoE(i,12) = WoE_16CO; 
     elseif strcmp( Xt{i,16} , 'CP' )
       X_WoE(i,12) = WoE_16CP;
     elseif strcmp( Xt{i,16} , 'MH' )
       X_WoE(i,12) = WoE_16MH;
     elseif strcmp( Xt{i,16} , 'PU' )
       X_WoE(i,12) = WoE_16PU;
     end
     
     % variable 18 
     if strcmp( Xt{i,18} , 'P' )
       X_WoE(i,13) = WoE_18P;
     elseif strcmp( Xt{i,18} , 'S' )
       X_WoE(i,13) = WoE_18S; 
     elseif strcmp( Xt{i,18} , 'I' )
       X_WoE(i,13) = WoE_18I;
     elseif strcmp( Xt{i,18} , 'U' )
       X_WoE(i,13) = WoE_18U;
     end
     
     % variable 24
     if strcmp( Xt{i,24} , '1' )
       X_WoE(i,14) = WoE_241;
     elseif strcmp( Xt{i,24} , '2' )
       X_WoE(i,14) = WoE_242; 
     elseif strcmp( Xt{i,24} , '3' )
       X_WoE(i,14) = WoE_243;
     elseif strcmp( Xt{i,24} , '' )
       X_WoE(i,14) = WoE_24nan;
     end
     
     % variable 25
     if strcmp( Xt{i,25} , 'Y' )
       X_WoE(i,15) = WoE_25Y;
     elseif strcmp( Xt{i,25} , 'N' )
       X_WoE(i,15) = WoE_25N;
     end
     
end

% Get rid of Nan's and +/- inf values in the X_WoE matrix (has 15 variables
% as columns). The number of observations (rows) will be reduced not much
% (not much will be lost).
[X_WoE,y] = cleanseDataMatrix(X_WoE,y);


end