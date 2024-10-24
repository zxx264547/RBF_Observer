% 初始化
clear;
load system.mat
% 仿真时间设置
t_start = 0; % 开始时间
t_end = 100; % 结束时间
dt = 0.1; % 时间步长
time = t_start:dt:t_end; % 时间向量
max_iterations = length(time); % 总迭代次数

% 初始化状态和控制输入

% 注意每一行是一个样本，每一列是一个变量
x_all = zeros(length(time), state_dim);
u_all = sys_input(time); % 控制输入，随时间变化
% 初始化故障信号
theta_all = zeros(length(time), 1);
% 初始化记录线性部分和不确定项
f_all = zeros(length(time), state_dim); 
eta_all = zeros(length(time), state_dim);

% 进行仿真
for iteration = 1:max_iterations
    % 当前时刻
    t = time(iteration); 
    % 当前状态(注意当前状态是列的形式)
    for i = 1:state_dim
        x(i,1) = x_all(iteration,i);
    end
    for i = 1:input_dim
        u(i,1) = u_all(iteration,i);
    end
    theta = fault(x);
    theta_all(iteration) = theta;    
    f = A * x + B * u;
    eta = uncertain(x);
    
    % 记录非线性部分和不确定项
    f_all(iteration,:) = f';
    eta_all(iteration,:) = eta';
    
    % 计算状态更新 (使用欧拉法)
    x_dot = f + eta + B * theta;
    
    % 更新状态
    if(iteration < max_iterations)
        x_all(iteration+1, :) = (x + x_dot * dt)';
    end

end
figure('Name','故障信号')
plot(time,theta_all, 'DisplayName','故障');
legend show;
xlabel('时间');
ylabel('故障幅值');
title('故障变化曲线');

figure('Name','未知项')
hold on; 
% 循环遍历每个变量（每一列）
for i = 1:size(eta_all,2)
    plot(time, eta_all(:,i), 'DisplayName', ['eta ', num2str(i)]);
end
hold off;
% 添加图例和标签
legend show; % 显示图例
xlabel('时间');
ylabel('');
title('未知量变化曲线');

figure('Name','状态')
hold on; 
% 循环遍历每个变量（每一列）
for i = 1:size(x_all,2)
    plot(time, x_all(:,i), 'DisplayName', ['x ', num2str(i)]);
end
hold off;
% 添加图例和标签
legend show; % 显示图例
xlabel('时间');
ylabel('');
title('x变化曲线');

% 组合数据
y_model = eta_all;
y_fault = theta_all;

% % 设置训练集和测试集的比例
% train_ratio = 0.8; % 80%用于训练，20%用于测试
% num_samples = size(x_all, 1);
% num_train = round(train_ratio * num_samples);
% 
% 
% indices = 1:num_samples;
% 
% % 分割数据为训练集和测试集
% x_train = x_all(indices(:, 1:num_train),:);
% y_train_model = y_model(indices(1:num_train),:);
% y_train_fault = y_fault(indices(1:num_train),:);
% 
% x_test = x_all(indices(num_train+1:end),:);
% y_test_model = y_model(indices(num_train+1:end),:);
% y_test_fault = y_fault(indices(num_train+1:end),:);


% 保存数据到文件
save('rbf_training_data.mat', 'x_all', 'y_model', 'y_fault');

% 显示生成的数据
disp('仿真数据已生成并保存至 rbf_training_data.mat 文件中。');
