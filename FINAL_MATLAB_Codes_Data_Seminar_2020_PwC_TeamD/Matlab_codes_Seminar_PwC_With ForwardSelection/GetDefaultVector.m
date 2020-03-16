function [y, FinalIndex] = GetDefaultVector(perfFile)
%%
% [y, FinalIndex] = GetDefaultVector('Performance_2017Q2.txt'); will give
% the default vector y within 12 months of a given quarter of a year.
% y is the binary variables which is 1 if loan defaulted within 12 months
% and 0 if not. 
% FinalIndex is the index for the files of 2018Q1 to remove the misaligned
% observations in the Get15RiskDrivers.m function and
% GetAll20RiskDrivers.m function.

fid = fopen(perfFile);
n = textscan(fid,'%f %*s %*s %*s %*s %*s %*s %*s %*s %*s %s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s','delimiter','|');
a_perf=n{1,1};
e=n{1,2};

g=unique(a_perf);
numberofloans=length(g);
defaulted=zeros(numberofloans,4);
defaulted(:,1)=g;

id=0;

% For the data set of 2018Q1 one more loan is in the performance fils
% compared to acquisition file. Therefore, we remove this redundant loan
% from the performance file in the Get15RiskDrivers.m function and
% GetAll20RiskDrivers.m function by using the FinalIndex.
% Extract the index for which the loan IDs are not the same to get rid of
% the aforementioned redundant loan observation.
if strcmp( perfFile, 'Performance_2018Q1.txt')
   T = readtable('Acquisition_2018Q1.txt','ReadVariableNames',false,'Delimiter','|');
   T = table2cell(T);
   LoanID2018Q1 =  T(:,1);
   
   FinalIndex =[];
   check = 1;
   indexLoan = 1;
   while check == 1
       if g(indexLoan) ~= LoanID2018Q1{indexLoan}
         FinalIndex = indexLoan;
         check = 0;
       else
         indexLoan = indexLoan + 1;    
       end
   end
end



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

sprintf('The final index to used for removing the redundant loan in the acquisition file of 2018Q1 is %s.', FinalIndex)

end