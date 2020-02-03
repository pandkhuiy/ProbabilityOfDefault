function [X_WoE,InfoValue,Xtable,y] = GetData(perfFile,h)
%%
%GetData('Performance_2017Q2.txt','Acquisition_2017Q2.txt'); will give the
%acquisition data with default vector, the amount of month until default
%and the defaulted within a year
% X_WoE contains 8 numeric variables (and 7 nominal variables that are
% encoded with WoE values).InfoValue is the vector of IV values of each
% nominal variables indicating how strong the nominal variable is with
% regards to the default variable y. Xtable is a table containg numerical
% variables and categorical variables (No WoE encoding used as it will be 
% used for decision trees function)
% y is the binary variables which is 1 if loan defaulted within 12 months
% and 0 if not.
tic
fid = fopen(perfFile);
n = textscan(fid,'%f %*s %*s %*s %*s %*s %*s %*s %*s %*s %s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s','delimiter','|');
a_perf=n{1,1};
e=n{1,2};

g=unique(a_perf);
numberofloans=length(g);
defaulted=zeros(numberofloans,4);
defaulted(:,1)=g;

id=0;

for i=1:size(a_perf,1)
    if id~=a_perf(i,1)
        id=a_perf(i,1);
        teller=0;
        check = 0;
    end
    if e{i}=='0'|e{i}=='1'|e{i}=='2'|e{i}=='X'
        teller=teller+1;
    elseif check == 0
        temp=find(defaulted(:,1)==a_perf(i,1));
        defaulted(temp,2)=1;
        defaulted(temp,3)=teller;
        if teller<12
            defaulted(temp,4)=1;
        end
        check = 1;
    end
end

y = defaulted(:,4);

% read the acquisition file h.
fid = fopen(h);
k = textscan(fid,'%f %s %s %f %f %f %s %s %f %f %f %f %f %s %s %s %f %s %s %s %s %s %s %s %s','delimiter','|');

acq_loanID = k{1,1};


% the 15 variables we are interested in are: 
% -----  Numeric variables: 4, 5, 6, 9 or 10, 11 , 12, 13 and 17.
% -----  categorical/nominal variables: 2 (R/C/B), 14 (Y/N), 15 (P/C/R/U), 16 (SF/CO/CP/MH/PU) , 18 (P/S/I/U)  , 24 (1/2/3/NaN) , 25 (Y/N) 

X_WoE = zeros(numel(g),8+7);
Xt = cell(numel(g),8+7); 

InfoValue = zeros(7,1);

IV2 = zeros(3,1);
IV14 = zeros(2,1);
IV15 = zeros(4,1);
IV16 = zeros(5,1);
IV18 = zeros(4,1);
IV24 = zeros(4,1);
IV25 = zeros(2,1);

for i = 1: numel(g)
     temp = find(acq_loanID == g(i));
     % numeric variables
     Xt{i,1} = k{1,4}(temp);
     Xt{i,2} = k{1,5}(temp);
     Xt{i,3} = k{1,6}(temp);
     Xt{i,4} = k{1,9}(temp);
     Xt{i,5} = k{1,11}(temp);
     Xt{i,6} = k{1,12}(temp);
     Xt{i,7} = k{1,13}(temp);
     Xt{i,8} = k{1,17}(temp);
     % nominal variables.
     Xt{i,9} = k{1,2}{temp,1};
     Xt{i,10} = k{1,14}{temp,1};
     Xt{i,11} = k{1,15}{temp,1};
     Xt{i,12} = k{1,16}{temp,1};
     Xt{i,13} = k{1,18}{temp,1};
     Xt{i,14} = k{1,24}{temp,1};
     Xt{i,15} = k{1,25}{temp,1};
end

sumGood = sum(y); sumBad = numel(y)- sumGood;

% Variable 2.
     r = strcmp( Xt(:,9) , 'R' );
     c = strcmp( Xt(:,9) , 'C' );
     b = strcmp( Xt(:,9) , 'B' );
     
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
     Y = strcmp( Xt(:,10) , 'Y' );
     N = strcmp( Xt(:,10) , 'N' );
     
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
     var15P = strcmp( Xt(:,11) , 'P' );
     var15C = strcmp( Xt(:,11) , 'C' );
     var15R = strcmp( Xt(:,11) , 'R' );
     var15U = strcmp( Xt(:,11) , 'U' );
     
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
     var16SF = strcmp( Xt(:,12) , 'SF' );
     var16CO = strcmp( Xt(:,12) , 'CO' );
     var16CP = strcmp( Xt(:,12) , 'CP' );
     var16MH = strcmp( Xt(:,12) , 'MH' );
     var16PU = strcmp( Xt(:,12) , 'PU' );
     
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
     var18P = strcmp( Xt(:,13) , 'P' );
     var18S = strcmp( Xt(:,13) , 'S' );
     var18I = strcmp( Xt(:,13) , 'I' );
     var18U = strcmp( Xt(:,13) , 'U' );
     
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
     var241 = strcmp( Xt(:,14) , '1' );
     var242 = strcmp( Xt(:,14) , '2' );
     var243 = strcmp( Xt(:,14) , '3' );
     var24nan = strcmp( Xt(:,14) , '' );
     
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
     Y = strcmp( Xt(:,15) , 'Y' );
     N = strcmp( Xt(:,15) , 'N' );
     
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
    

for i = 1:numel(g)
     % Now, make the X matrix consisting of numeric variables and WoE
     % encoded nominal variables with corresponding InfoValue for each
     % nominal variable indicating the strength of the relat.ship  between
     % default variable y.
    
     temp = find(acq_loanID == g(i));
     % numeric variables
     X_WoE(i,1) = k{1,4}(temp);
     X_WoE(i,2) = k{1,5}(temp);
     X_WoE(i,3) = k{1,6}(temp);
     X_WoE(i,4) = k{1,9}(temp);
     X_WoE(i,5) = k{1,11}(temp);
     X_WoE(i,6) = k{1,12}(temp);
     X_WoE(i,7) = k{1,13}(temp);
     X_WoE(i,8) = k{1,17}(temp);
     
     % variable 2   
     if strcmp( k{1,2}{temp,1} , 'R' )
       X_WoE(i,9) = WoE_2r;
     elseif strcmp( k{1,2}{temp,1} , 'C' )
       X_WoE(i,9) = WoE_2c; 
     elseif strcmp( k{1,2}{temp,1} , 'B' )
       X_WoE(i,9) = WoE_2b;
     end
     
     % variable 14  
     if strcmp( k{1,14}{temp,1} , 'Y' )
       X_WoE(i,10) = WoE_14Y;
     elseif strcmp( k{1,14}{temp,1} , 'N' )
       X_WoE(i,10) = WoE_14N; 
     end

     % variable 15  
     if strcmp( k{1,15}{temp,1} , 'P' )
       X_WoE(i,11) = WoE_15P;
     elseif strcmp( k{1,15}{temp,1} , 'C' )
       X_WoE(i,11) = WoE_15C; 
     elseif strcmp( k{1,15}{temp,1} , 'R' )
       X_WoE(i,11) = WoE_15R;
     elseif strcmp( k{1,15}{temp,1} , 'U' )
       X_WoE(i,11) = WoE_15U;
     end
     
     % variable 16  
     if strcmp( k{1,16}{temp,1} , 'SF' )
       X_WoE(i,12) = WoE_16SF;
     elseif strcmp( k{1,16}{temp,1} , 'CO' )
       X_WoE(i,12) = WoE_16CO; 
     elseif strcmp( k{1,16}{temp,1} , 'CP' )
       X_WoE(i,12) = WoE_16CP;
     elseif strcmp( k{1,16}{temp,1} , 'MH' )
       X_WoE(i,12) = WoE_16MH;
     elseif strcmp( k{1,16}{temp,1} , 'PU' )
       X_WoE(i,12) = WoE_16PU;
     end
     
     % variable 18 
     if strcmp( k{1,18}{temp,1} , 'P' )
       X_WoE(i,13) = WoE_18P;
     elseif strcmp( k{1,18}{temp,1} , 'S' )
       X_WoE(i,13) = WoE_18S; 
     elseif strcmp( k{1,18}{temp,1} , 'I' )
       X_WoE(i,13) = WoE_18I;
     elseif strcmp( k{1,18}{temp,1} , 'U' )
       X_WoE(i,13) = WoE_18U;
     end
     
     % variable 24
     if strcmp( k{1,24}{temp,1} , '1' )
       X_WoE(i,14) = WoE_241;
     elseif strcmp( k{1,24}{temp,1} , '2' )
       X_WoE(i,14) = WoE_242; 
     elseif strcmp( k{1,24}{temp,1} , '3' )
       X_WoE(i,14) = WoE_243;
     elseif strcmp( k{1,24}{temp,1} , '' )
       X_WoE(i,14) = WoE_24nan;
     end
     
     % variable 25
     if strcmp( k{1,25}{temp,1} , 'Y' )
       X_WoE(i,15) = WoE_25Y;
     elseif strcmp( k{1,25}{temp,1} , 'N' )
       X_WoE(i,15) = WoE_25N;
     end
     
end

Xtable = cell2table(Xt);
Xtable.Properties.VariableNames={'ORIGINAL_INTEREST_RATE','ORIGINAL_UPB','ORIGINAL_LOAN_TERM','ORIGINAL_LOAN_TO_VALUE',...
     'NUMBER_OF_BORROWERS','ORIGINAL_DEBT_TO_INCOME_RATIO ','BORROWER_CREDIT_SCORE_AT_ORIGINATION','NUMBER_OF_UNITS','ORIGINATION_CHANNEL','FIRST_TIME_HOME_BUYER_INDICATOR',...
     'LOAN_PURPOSE','PROPERTY_TYPE','OCCUPANCY_TYPE','MORTGAGE_INSURANCE_TYPE','RELOCATION_MORTGAGE_INDICATOR'};
toc
    


end