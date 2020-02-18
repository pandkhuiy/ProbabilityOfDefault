function[newX1,newy1]=BalancedData(X1,y1)
%%
temp1 = [X1,y1];
temp3 = sortrows(temp1,size(temp1,2));

numberones1=sum(temp3(:,end));

if numberones1<=5000
    onesmatrix=temp3(end-numberones1+1:end,:);
    zerorows=randperm(size(temp1,1)-numberones1,10000-5000);
    zeromatrix=temp3(zerorows,:);
    tx=[onesmatrix(:,1:end-1);zeromatrix(:,1:end-1)];
    ty=[onesmatrix(:,end);zeromatrix(:,end)];
    [X1ada,y1ada]= ADASYN(tx,ty,1,[],[],false);
    rows=randperm(size(X1ada,1),10000-numberones1-5000);
    newX1=[tx;X1ada(rows,:)];
    newy1=[ty;y1ada(rows,:)];
else
    onesrows=randperm(numberones1,5000)';
    onesrows=onesrows+ones(5000,1)*(size(temp1,1)-numberones1);
    onesmatrix=temp3(onesrows,:);
    
    zerorows=randperm(size(temp1,1)-numberones1,10000-5000);
    zeromatrix=temp3(zerorows,:);
    
    newX1=[onesmatrix(:,1:end-1);zeromatrix(:,1:end-1)];
    newy1=[onesmatrix(:,end);zeromatrix(:,end)];
end

end