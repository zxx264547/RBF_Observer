close all
clear
clc
%% 定义改进的状态矩阵A
% A = [-0.1, -0.1, -0.1, 0, 0;
% 2*pi, 0, 0, 0, 0;
% 0, 0, -3.3, 3.3, 0;
% -1, 0, 0, -2.7, 0;
% 2, 0, 0, 0, 1];
% C = [2, 0, 0, 0, 1];
% B = [0; 0; 0; -2.7; 0];
% 定义状态矩阵A（稳定）
A = [-1 0.5 0.2 0.1 0.05;
     0.3 -1 0.4 0.2 0.1;
     0.2 0.1 -0.8 0.3 0.2;
     0.1 0.2 0.4 -0.9 0.3;
     0.05 0.1 0.2 0.3 -0.7];

% 定义输入矩阵B
B = [1; 
     0; 
     0; 
     0; 
     0];

% 定义输出矩阵C
C = [1 0 0 0 0];
% 状态维度
state_dim = size(A,1);
% 输入维度
input_dim = size(C,1);
save('system.mat', 'A', 'B', 'C', 'state_dim', 'input_dim');

%% 收集数据
data_generation;

%% 训练模型
load('rbf_training_data.mat');
min_centers = 100;
max_centers = 300;
[fault_centers, fault_sigma, fault_weights_optimal] = rbf_train(x_all, y_model, min_centers, max_centers);
[model_centers, model_sigma, model_weights_optimal] = rbf_train(x_all, y_fault, min_centers, max_centers);
save('rbf_model_parameters.mat', 'fault_centers', 'fault_sigma', "fault_weights_optimal", ...
    'model_centers', 'model_sigma', 'model_weights_optimal');

%% 计算KP
resolve_KP;

%% 观测器仿真
% T1;
T2;