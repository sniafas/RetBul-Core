close all
clear all
load houses.dat
j=1

for i = {'03','13','15','22','39','60'}
    
    path1 = sprintf('/home/steve/Documents/master/Thesis/clean/source/surf_results/map/surf_MAP_%s',i{1})
    path2 = sprintf('/home/steve/Documents/master/Thesis/clean/source/sift_results/map/sift_MAP_%s',i{1})
    
    delimiter = '';
    formatSpec = '%f%[^\n\r]';
    fileID = fopen(path1,'r');
    dataArray1 = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
    fclose(fileID);

    fileID = fopen(path2,'r');
    dataArray2 = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
    fclose(fileID);

    surf(:,j) = dataArray1{:,1};
    sift(:,j) = dataArray2{:,1}; 
            
    j = j + 1    
end 

figure(1)
plot(houses,surf,'bo','LineWidth',2, 'MarkerSize',10)
hold on
plot(houses,sift,'rx','LineWidth',2, 'MarkerSize',10)
grid on
legend('SURF - 10 Inliers','SIFT - 8 Inliers')
set(gca, 'YLim',[0 1],'XLim',[0 35], 'XTickLabel' , [ 0 3 13 15 22 39 60]);
xlabel('Building Id');
ylabel('Percent');
title('mAP / Building')