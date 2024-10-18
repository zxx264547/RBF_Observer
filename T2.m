% MATLAB仿真代码：带不确定项和故障信号的系统故障检测

% 初始化参数
clear; clc; close all;
load optimal_fault_rbf_parameters.mat;
load optimal_model_rbf_parameters.mat

%% 系统参数初始化
A = [0, 1; -2, -3];
B = [0; 1];
C = [1, 0];
K = [1; 1];
P = [2.2761,0.2845;0.2845,0.5690];
Psi = [1; 1]; % 故障影响矩阵,维度为输入*故障信号维数

% 初始化系统状态和观测器
x = [0; 1]; % 初始系统状态
x_hat = [0; 1]; % 初始观测器状态
y = C * x; % 系统输出
y_hat = C * x_hat; % 观测器输出

%% 神经网络参数初始化
% 训练后的RBF的中心位置
model_num_neurons = size(model_centers,1); % RBF神经量元数
fault_num_neurons = size(fault_centers,1);

% 训练后的RBF宽度
model_widths = model_sigma;
fault_widths = fault_sigma;

%% W更新自适应律初始化
% 权值矩阵初始化
W1 = zeros(2, model_num_neurons); 
W2 = zeros(1, fault_num_neurons); 

% 自适应率的参数设置
L1 = eye(2); 
L2 = eye(1); 
lambda1 = 0.03; % 学习率参数
lambda2 = 0.03; % 学习率参数

%% 仿真参数初始化
% 设置仿真参数
t_start = 0;
t_end = 20; % 仿真时间,单位（s）
fault_time = 5; % 故障开始的时间为5s
dt = 0.01; % 仿真时间步长为0.01s
time = t_start:dt:t_end; % 时间向量
max_iterations = length(time); % 总迭代次数

% 数据记录
true_fault = zeros(max_iterations, 1);
observed_fault = zeros(max_iterations, 1);
state_observation_error_x1 = zeros(max_iterations, 1);
state_observation_error_x2 = zeros(max_iterations, 1);
true_x1 = zeros(max_iterations, 1);
true_x2 = zeros(max_iterations, 1);
observation_x1 = zeros(max_iterations, 1);
observation_x2 = zeros(max_iterations, 1);
true_y = zeros(max_iterations, 1);
observation_y = zeros(max_iterations, 1);

%% 开始迭代
for iteration = 1:max_iterations
    % 当前时刻
    t = time(iteration); 
    % 输入信号
    u = 0.5 * sin(2 * pi * t) + 0.5 * cos(0.008 * pi * t);
    % 不确定项
    eta1 = -sin(x(2)) - 0.1 * sin(x(1)) * cos(x(2));
    eta2 = 0.1 * sin(x(1)) * cos(x(2));
    % 故障信号
    if t >= fault_time
        theta = 2 + x(2) / (x(1) +1) * sin(pi * t); % 故障信号
    else
        theta = 0; % 无故障
    end
    
   
    %% 真实系统
    % 系统线性部分
    f = A * x + B * u;
    f1 = f(1);
    f2 = f(2);
    
    % 系统状态方程
    x_dot = [f1 + eta1 + Psi(1) * theta; 
             f2 + eta2 + Psi(2) * theta];
    x = x + x_dot * dt; % 更新状态
    y = C * x; % 更新系统输出

    %% W更新律
    % 扩展 rbf_input，使其与 centers 的维度匹配
    rbf_input_hat_fault = repmat(x_hat', size(fault_centers, 1), 1);
    rbf_input_hat_model = repmat(x_hat', size(model_centers, 1), 1);
    % 计算RBF激活函数输出
    phi1 = exp(-sum((rbf_input_hat_model - model_centers).^2, 2) ./ (2 * model_widths.^2));
    phi2 = exp(-sum((rbf_input_hat_fault - fault_centers).^2, 2) ./ (2 * fault_widths.^2));
    
    % 计算观测误差
    y_epsilon = y - y_hat;
        
    % 权值更新公式中的各项计算
    norm_C = norm(C, 'fro');
    norm_y_epsilon_CP = norm((y_epsilon' * C) * P, 'fro');
    norm_y_epsilon_CP_Psi = norm((y_epsilon' * C) * P * Psi, 'fro');
    
    % 计算 W1 和 W2 的更新
    W1_dot = (L1 * ((P * C') * y_epsilon) * phi1' - lambda1 * norm_y_epsilon_CP * L1 * W1) / (norm_C^2);
    if t < fault_time
        W2_dot = 0;
    else
        W2_dot = (L2 * Psi' * ((P * C') * y_epsilon) * phi2' - lambda2 * norm_y_epsilon_CP_Psi * L2 * W2) / (norm_C^2);
    end    
    W1 = W1 + W1_dot * dt;
    W2 = W2 + W2_dot * dt;

    %% 计算神经网络逼近的信号

    % 系统的非线性部分
    f_nonlin_hat = W1 * phi1;

    % 故障信号
    theta_hat = W2 * phi2;

    %% 观测器
    % dx_hat = A_KC * x_hat + W1 * phi1 + Psi * (W2 * phi2) + K * C * x;
    x_hat_dot = A * x_hat + f_nonlin_hat + Psi * theta_hat + K * (y_epsilon);
    x_hat = x_hat + x_hat_dot * dt; % 更新观测器状态
    y_hat = C * x_hat; % 更新观测器输出
    
    %% 记录数据
    observed_fault(iteration, 1) = theta_hat; % 记录观测到的故障信号
    state_observation_error_x1(iteration) = abs(x(1) - x_hat(1)); % 记录x1的观测误差
    state_observation_error_x2(iteration) = abs(x(2) - x_hat(2)); % 记录x2的观测误差
    true_x1(iteration) = x(1);
    true_x2(iteration) = x(2);
    observation_x1(iteration) = x_hat(1);
    observation_x2(iteration) = x_hat(2);
    true_y(iteration) = y;
    observation_y(iteration) = y_hat;
    % 记录真实的故障输入
    true_fault(iteration) = theta;
end

%% 绘制定理1故障观测结果和状态观测误差曲线
figure('Name',"故障观测结果和状态观测误差曲线");
subplot(3, 1, 1);
plot(time, true_fault(1:iteration), 'g');
xlabel('t/s');
ylabel('幅度');
title('真实故障信号');

subplot(3, 1, 2);
plot(time, observed_fault(1:iteration), 'b');
xlabel('t/s');
ylabel('幅度');
title('观测故障信号');

subplot(3, 1, 3);
plot(time, state_observation_error_x1(1:iteration), 'r', 'DisplayName', 'x1误差');
hold on;
plot(time, state_observation_error_x2(1:iteration), 'b', 'DisplayName', 'x2误差');
hold off;
xlabel('t/s');
ylabel('幅度');
title('状态的观测误差曲线');
legend('x1误差', 'x2误差');

figure('Name',"真实值与观测值对比");
subplot(3, 1, 1);
plot(time, true_x1(1:iteration), 'g');
hold on;
plot(time, observation_x1(1:iteration), 'r');
xlabel('t/s');
ylabel('幅度');
title('x1对比');
legend('真实x1', '观测x1');

subplot(3, 1, 2);
plot(time, true_x2(1:iteration), 'g');
hold on;
plot(time, observation_x2(1:iteration), 'r');
xlabel('t/s');
ylabel('幅度');
title('x2对比');
legend('真实x2', '观测x2');

subplot(3, 1, 3);
plot(time, true_y(1:iteration), 'g');
hold on;
plot(time, observation_y(1:iteration), 'r');
xlabel('t/s');
ylabel('幅度');
title('y对比');
legend('真实y', '观测y');

disp('仿真完成');
disp(['总共迭代次数: ', num2str(iteration)]);