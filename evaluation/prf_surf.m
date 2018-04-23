close all
clear all
load thresshold.dat

exp = 1;
type = 0.75
class = 'a';

for i = 1:11
    path = sprintf('/home/steve/Documents/master/Thesis/clean/source/surf_results/prf/surf_results_%d',i+4)
    delimiter = ',';
    formatSpec = '%f%f%f%f%[^\n\r]';
    fileID = fopen(path,'r');
    dataArray = textscan(fileID, formatSpec,'Delimiter', delimiter,'ReturnOnError', false)
    fclose(fileID);
    P(:,i) = dataArray{:,1};
    R(:,i) = dataArray{:,2};
    F(:,i) = dataArray{:,3};
    v(:,i) = dataArray{:,4};
    
end


figure(1)

plot(thresshold,P,'ro-','LineWidth',2, 'MarkerSize',10)
hold on
plot(thresshold,R,'kx-','LineWidth',2, 'MarkerSize',10)
hold on
plot(thresshold,F,'b*-','LineWidth',2, 'MarkerSize',10)
grid on
t = sprintf('SURF');
title(t);
xlabel('Inlier Thresshold');
ylabel('Percent');
lgn = legend('Precision','Recall','F Measure')