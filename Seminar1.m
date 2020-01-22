% read data_: 2009 Q1

%file= fopen('Performance_2009Q1.txt');
file= fopen('test_file_pef_2009');
data = textscan(file, '%f %*s %*s %*s %*s %*s %*s %*s %*s %*s %s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s', 'Delimiter', '|');  
data_size = size(data{1,2},1);
data_new_col_1 = unique(data{1,1});
data_new_col_2 = zeros(size(data_new_col_1,1),1);
data_new_col_3 = zeros(size(data_new_col_1,1),1);
A = [data_new_col_1,data_new_col_2,data_new_col_3];

indic = str2double(data{1,2});
for t=1:data_size
    
    if indic(t)==3
    Id_number = data{1,1}(t);
    numberinA=find(A(:,1)==Id_number);
    A(numberinA,2) = 1;
    indexes =find(data{1,1}==data{1,1}(t))
    index = indexes(1)
    if t - index <12
        A(numberinA,3)=1;
    end
        
    end
end
acceptance_rate = sum(A(:,2))/size(A,1)
% read aquisiation data






% merge data