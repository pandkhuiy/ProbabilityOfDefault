function [X,y1,y2]= GetXY(data,columns)

X=[];

for i=1:25
    if any(columns(:)==i)
        temp=table2array(data(:,i));
        temp2=class(temp);
        if strcmp(temp2,'cell')
            temp3=GetDummy(data(:,i));
            X=[X,temp3(:,1:end-1)];
        else
            X=[X,temp];
        end
    end
    y1=table2array(data(:,26));
    y2=table2array(data(:,28));
end