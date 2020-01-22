function [data] = GetData(a,h)
%GetData('Performance_2017Q2.txt','Acquisition_2017Q2.txt'); will give the
%acquisition data with default vector, the amount of month until default
%and the defaulted within a year


fid = fopen(a);
n = textscan(fid,'%f %*s %*s %*s %*s %*s %*s %*s %*s %*s %s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s','delimiter','|');
a=cell2mat(n(1,1));
b=cell2table(n{1,2}(:));
c=table2array(b);
d=char(c);
e=d(:,1);

f=[a,e];

g=unique(a);
numberofloans=length(g);
defaulted=zeros(numberofloans,3,4);
defaulted(:,1)=g;

id=0;

for i=1:size(f,1)
    if id~=a(i,1)
        id=a(i,1);
        teller=0;
    end
    if f(i,2)=='0'||'1'||'2'||'X'
        teller=teller+1;
    end
    if f(i,2)=='3'
        temp=find(defaulted(:,1)==a(i,1));
        defaulted(temp,2)=1;
        defaulted(temp,3)=teller;
        if teller<12
            defaulted(temp,4)=1;
        end
    end
end

temp=array2table(defaulted(:,2:4));
temp.Properties.VariableNames={'default','monthsToDefault','defaultedWithinYear'};
data = [readtable(h,'ReadVariableNames',false,'Delimiter','|'),temp];
end