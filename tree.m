clc;
clear;
%Fetch file
filename = "spambase.csv";
%Extract table
dataTable = readtable(filename);
TableMatrix = table2array(dataTable);
[numRow,numCol] = size(TableMatrix);
ind = randperm(numRow);
TableMatrix = TableMatrix(ind,:);
%split train and test
partition = cvpartition(size(TableMatrix,1),'HoldOut',0.2);
idx = partition.test;
dataForTest = TableMatrix(idx,:);
dataForTrain = TableMatrix(~idx,:);
xTrain = dataForTrain(:,1:57); % Xcoor for training
yTrain = dataForTrain(:,58);% Ycoor for training
xTest = dataForTest(:,1:57); % Xcoor for testing
yTest = dataForTest(:,58);%Ycoor for testing

treeModel = fitctree(xTrain,yTrain);% the decision tree
yPred = predict(treeModel,xTest);
accuracy = sum(yPred == yTest)/numel(yTest);
treeChart = confusionchart(yTest,yPred);