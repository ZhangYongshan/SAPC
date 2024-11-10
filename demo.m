clear; close all; clc;

addpath(genpath('funs'));

% dataset path
addpath('D:\datasets');

%%
dataType = 'Salinas'; dk = 20; projDim = 60;  % 0.7881, 0.7637, 0.8464, 0.8056, 0.6929, 0.6972, 0.6767, 0.7373
% dataType = 'PaviaU'; dk = 12; projDim = 20;   % 0.6394, 0.5294, 0.6189, 0.7132, 0.4491, 0.5602, 0.5621, 0.5920
% dataType = 'PaviaC'; dk = 12; projDim = 40;   % 0.9334, 0.9055, 0.9097, 0.9351, 0.9743, 0.7081, 0.7970, 0.7139
% dataType = 'XuZhou'; dk = 8; projDim = 140;   % 0.8517, 0.8135, 0.8251, 0.8806, 0.7884, 0.7421, 0.7658, 0.7582
% dataType = 'LongKou'; dk = 8; projDim = 100;  % 0.8288, 0.7713, 0.7999, 0.8920, 0.7547, 0.5301, 0.5789, 0.5044

%%
switch dataType
    case 'Salinas'
        load('Salinas_corrected.mat');
        load('Salinas_gt.mat');
        data3D = salinas_corrected;
        gt2D = salinas_gt;
        clear salinas_corrected salinas_gt;
    case 'PaviaU'
        load('PaviaU.mat');
        load('PaviaU_gt.mat');
        data3D = paviaU;
        gt2D = double(paviaU_gt);
        clear paviaU paviaU_gt;
    case 'PaviaC'
        load('Pavia.mat');
        load('Pavia_gt.mat');
        data3D = pavia;
        gt2D = double(pavia_gt);
        clear pavia pavia_gt
    case 'XuZhou'
        load('XuZhou.mat');
        load('XuZhou_gt.mat');
        data3D = double(xuzhou);
        gt2D = double(xuzhou_gt);
        clear xuzhou xuzhou_gt
    case 'LongKou'
        load('WHU_Hi_LongKou.mat');
        load('WHU_Hi_LongKou_gt.mat');
        data3D = double(WHU_Hi_LongKou);
        gt2D = double(WHU_Hi_LongKou_gt);
        clear WHU_Hi_LongKou WHU_Hi_LongKou_gt
end

gt = double(gt2D(:));
ind = find(gt);
c = length(unique(gt(ind)));

tic;
% HSI data preprocessing
[X,spLabel,num_Pixel] = preData(data3D,dk);

alpha1 = num_Pixel/size(X,2);

[y_pred, Z, S, W, clusternum] = SAPC(X, spLabel(:), num_Pixel, c, projDim, alpha1);
time = toc;
%%
results = evaluate_results_clustering(gt(ind),y_pred(ind));

result4 = round(results',4);


