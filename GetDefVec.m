function [defaulted] = GetDefVec(filename)


fid = fopen(filename);
n = textscan(fid,'%f %*s %*s %*s %*s %*s %*s %*s %*s %*s %s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s','delimiter','|');
a=cell2mat(n(1,1));
b=cell2table(n{1,2}(:));
c=table2array(b);
d=char(c);
e=d(:,1);

f=[a,e];

g=unique(a);
numberofloans=length(g);
defaulted=zeros(numberofloans,2);
defaulted(:,1)=g;


for i=1:size(f,1)
    default=0;
    if f(i,2)=='3'
        temp=find(defaulted(:,1)==a(i,1));
        defaulted(temp,2)=1;
    end
end
end