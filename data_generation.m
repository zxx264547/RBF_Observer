% 初始化
clear; clc;

% 仿真时间设置
t_start = 0; % 开始时间
t_end = 100; % 结束时间
dt = 0.1; % 时间步长
t = t_start:dt:t_end; % 时间向量

% 初始化状态和控制输入
x = zeros(length(t), 2); % 状态向量 [x1, x2]
u = 0.5 * sin(2 * pi * t)' + 0.5 * cos(0.008 * pi * t)'; % 控制输入，随时间变化
x(1, :) = [0, 0]; % 初始状态
A = [0, 1; -2, -3];
B = [0; 1];
C = [1, 0];

% 初始化故障信号
theta = zeros(length(t), 1);

% 初始化记录线性部分和不确定项
f_lin = zeros(length(t), 2); % [f1, f2]
eta = zeros(length(t), 2); % [eta1, eta2]

% 进行仿真
for k = 1:length(t)-1
    % 当前状态
    x1 = x(k, 1);
    x2 = x(k, 2);
    u_k = u(k);
    theta_k = 2 + x2 / (x1 +1) * sin(pi * 0.1 * k);
    
    theta(k) = theta_k;
    
    f = A * [x1; x2] + B * u_k;
    f1 = f(1);
    f2 = f(2);
    
    % 计算不确定项 eta1 和 eta2
    eta1 = -sin(x2) - 0.1 * sin(x1) * cos(x2);
    eta2 = 0.1 * sin(x1) * cos(x2);
    
    % 记录非线性部分和不确定项
    f_lin(k, :) = [f1, f2];
    eta(k, :) = [eta1, eta2];
    
    % 计算状态更新 (使用欧拉法)
    dx1 = f1 + eta1 + theta_k;
    dx2 = f2 + eta2 + theta_k;
    
    % 更新状态
    x(k+1, 1) = x1 + dx1 * dt;
    x(k+1, 2) = x2 + dx2 * dt;

end
plot(theta);
plot(eta);

% 组合数据
y_model = eta;
y_fault = theta;

% 设置训练集和测试集的比例
train_ratio = 0.8; % 80%用于训练，20%用于测试
num_samples = size(x, 1);
num_train = round(train_ratio * num_samples);


indices = 1:num_samples;

% 分割数据为训练集和测试集
x_train = x(indices(1:num_train), :);
y_train_model = y_model(indices(1:num_train), :);
y_train_fault = y_fault(indices(1:num_train), :);

x_test = x(indices(num_train+1:end), :);
y_test_model = y_model(indices(num_train+1:end), :);
y_test_fault = y_fault(indices(num_train+1:end), :);


% 保存数据到文件
save('rbf_training_data.mat', 'x_train', 'y_train_model', 'y_train_fault', 'x_test', 'y_test_model', 'y_test_fault', 't');

% 显示生成的数据
disp('仿真数据已生成并保存至 rbf_training_data.mat 文件中。');
