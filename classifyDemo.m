clear;clc;

% SVM
trainData = [0 1; -1 0; 2 2; 3 3; -2 -1;-4.5 -4; 2 -1; -1 -3];
group = [1 1 -1 -1 1 1 -1 -1]';
testData = [5 2;3 1;-4 -3];
svm_struct = svmtrain(trainData,group);    
Group = svmclassify(svm_struct,testData);

%Logistic
trainData = [0 1; -1 0; 2 2; 3 3; -2 -1;-4.5 -4; 2 -1; -1 -3];
group = [1 1 0 0 1 1 0 0]';
testData = [5 2;3 1;-4 -3];
[testNum, attrNum] = size(testData);
testData2 = [ones(testNum,1), testData];
B = glmfit(trainData, [group ones(size(group))],'binomial', 'link', 'logit')
p = 1.0 ./ (1 + exp(- testData2 * B))