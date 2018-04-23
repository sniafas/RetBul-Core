close all
clear all
load recall.dat
  
path1 = sprintf('/home/steve/Documents/master/Thesis/clean/source/surf_results/elp/surf_pr11_a')
path2 = sprintf('/home/steve/Documents/master/Thesis/clean/source/sift_results/elp/sift_pr11_a')

delimiter = '';
formatSpec = '%f%[^\n\r]';
fileID = fopen(path1,'r');
dataArray1 = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
fclose(fileID);

fileID = fopen(path2,'r');
dataArray2 = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
fclose(fileID);

surf(:,1) = dataArray1{:,1};
sift(:,1) = dataArray2{:,1};

plot(recall,surf,'bo-','LineWidth',2, 'MarkerSize',10)
hold on
plot(recall,sift,'rx-','LineWidth',2, 'MarkerSize',10)
hold on

grid on

set(gca, 'YLim',[0 1],'XLim',[0 1]);
% set(gca, 'YLim',[0 1]);
xlabel('Recall');
ylabel('Precision');
legend('SURF','SIFT')
title('Precision vs Recall')

  