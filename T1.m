% MATLAB仿真代码：带不确定项和故障信号的系统故障检测

% 初始化参数
clear; clc; close all;
load optimal_fault_rbf_parameters.mat;
load optimal_model_rbf_parameters.mat
% 设置仿真参数
max_iterations = 1000; % 最大迭代次数
T = 1 / (0.01 * max_iterations);
learning_rate = 0.01; % 权值更新的学习率
iteration = 0; % 当前迭代次数

% 初始化系统参数
A = [0, 1; -2, -3];
B = [0; 1];
C = [1, 0];
K = [1; 1];
P = [-0.3794,-0.0474;-0.0474,-0.2371];
% 初始化系统状态和观测器
x = [0; 1]; % 初始系统状态
x_hat = [0; 1]; % 初始观测器状态
y = C * x; % 系统输出
y_hat = C * x_hat; % 观测器输出



% 假设已经训练好的RBF神经网络参数

% 训练后的RBF的中心位置（二维情况）

model_num_neurons = size(model_centers,1); % RBF神经量元数
fault_num_neurons = size(fault_centers,1);
% 训练后的RBF宽度
model_widths = model_sigma;
fault_widths = fault_sigma;
% 权值矩阵初始化
W1 = randn(2, model_num_neurons) * 0.1; % 初始化
W2 = randn(1, fault_num_neurons) * 0.1; 

% 论文中的参数设置
L1 = eye(2); % L1 矩阵，设置为2x2单位矩阵
L2 = eye(1); % L2 矩阵，设置为2x2单位矩阵

Psi = [1; 1]; % 故障影响矩阵,维度为输入*故障信号维数
lambda1 = 0.03; % 学习率参数
lambda2 = 0.03; % 学习率参数

% 故障输入设置（从500次迭代开始，故障信号震荡）
fault_time = 5; % 故障开始的迭代次数

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

%% 仿真循环
while iteration < max_iterations
    % 输入信号
    t = 0.01 * iteration;
    u = 0.5 * sin(2 * pi * t) + 0.5 * cos(0.008 * pi * t); % 输入
    % 计算不确定项
    eta1 = -sin(x(2)) - 0.1 * sin(x(1)) * cos(x(2));
    eta2 = 0.1 * sin(x(1)) * cos(x(2));
    % 故障信号：在迭代次数超过500时变为2
    if t >= fault_time
        theta = 2 + x(2) / (x(1) +1) * sin(pi * t); % 故障信号
    else
        theta = 0; % 无故障
    end
    
    % 记录真实的故障输入
    true_fault(iteration + 1) = theta;
    %% 真实的系统

    % 计算系统的非线性部分
    f = A * x + B * u;
    f1 = f(1);
    f2 = f(2);
    
    % 更新系统状态方程，包含不确定项和故障信号
    dx = [f1 + eta1 + Psi(1) * theta; f2 + eta2 + Psi(2) * theta];
    x = x + dx * T; % 更新状态
    y = C * x; % 更新系统输出

    %% 观测器

    % 将系统状态和输入结合起来作为RBF网络的输入
    % rbf_input = [x; u]; % 将x和u作为输入
    rbf_input_hat = x_hat; % 观测器状态和输入的结合
    
    % 扩展 rbf_input，使其与 centers 的维度匹配
    % rbf_input_expanded = repmat(rbf_input', size(model_centers, 1), 1); % 扩展为 num_neurons x 3
    rbf_input_hat_fault_expanded = repmat(rbf_input_hat', size(fault_centers, 1), 1);
    rbf_input_hat_model_expanded = repmat(rbf_input_hat', size(model_centers, 1), 1);
    % 计算RBF激活函数输出
    phi1 = exp(-sum((rbf_input_hat_model_expanded - model_centers).^2, 2) ./ (2 * model_widths.^2));
    phi2 = exp(-sum((rbf_input_hat_fault_expanded - fault_centers).^2, 2) ./ (2 * fault_widths.^2));
    
    
    % 计算观测误差
    y_epsilon = y - y_hat;
    
    
    % 权值更新公式中的各项计算
    norm_C = norm(C, 'fro');
    norm_y_epsilon_CP = norm((y_epsilon' * C) * P, 'fro');
    norm_y_epsilon_CP_Psi = norm((y_epsilon' * C) * P * Psi, 'fro');
    
    % 计算 W1 和 W2 的更新
    W1_dot = (L1 * ((P * C') * y_epsilon) * phi1' - lambda1 * norm_y_epsilon_CP * L1 * W1) / (norm_C^2);
    W2_dot = (L2 * Psi' * ((P * C') * y_epsilon) * phi2' - lambda2 * norm_y_epsilon_CP_Psi * L2 * W2) / (norm_C^2);


    
    %% 计算矩阵A
    % syms a b
    % 
    % % 定义符号函数 f，使用常量值代替符号常量
    % f = [
    %     -a^2 - a*b - sin(b) - 0.1*sin(a)*cos(b) + 3*t + 0.25*u + theta;
    %     -b^2 + 4*a*b + 0.1*sin(a)*cos(b) + 2*u + theta
    % ];
    % 
    % % 定义变量向量
    % vars = [a; b];
    % 
    % % 解方程
    % solutions = vpasolve(f, vars);
    % 
    % % 将解数值化
    % sol_a = double(solutions.a);
    % sol_b = double(solutions.b);
    % 
    % % 计算雅可比矩阵并代入求解结果
    % A = jacobian(f, vars);
    % A_num = double(subs(A, [a, b], [sol_a, sol_b]));
    
   
    %%
    W2 = fault_weights_optimal';
    W1 = model_weights_optimal';
    
    % dx_hat = A_KC * x_hat + W1 * phi1 + Psi * (W2 * phi2) + K * C * x;
    dx_hat = A * x_hat + W1 * phi1 + Psi * W2 * phi2 + K*(y_epsilon);
    x_hat = x_hat + dx_hat * T; % 更新观测器状态
    y_hat = C * x_hat; % 更新观测器输出
    
    % 记录故障观测和状态观测误差
    
    
    observed_fault(iteration + 1, 1) = W2 * phi2; % 记录观测到的故障信号
    state_observation_error_x1(iteration + 1) = abs(x(1) - x_hat(1)); % 记录x1的观测误差
    state_observation_error_x2(iteration + 1) = abs(x(2) - x_hat(2)); % 记录x2的观测误差
    true_x1(iteration + 1) = x(1);
    true_x2(iteration + 1) = x(2);
    observation_x1(iteration + 1) = x_hat(1);
    observation_x2(iteration + 1) = x_hat(2);
    true_y(iteration + 1) = y;
    observation_y(iteration + 1) = y_hat;

    
    % 迭代次数增加
    iteration = iteration + 1;
    disp(iteration);
end

%% 绘制定理1故障观测结果和状态观测误差曲线
figure;
time = (1:iteration) / 100;
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

figure;
subplot(3, 1, 1);
plot(time, true_x1(1:iteration), 'g');
hold on;
plot(time, observation_x1(1:iteration), 'r');
xlabel('t/s');
ylabel('幅度');
title('x1对比');

subplot(3, 1, 2);
plot(time, true_x2(1:iteration), 'g');
hold on;
plot(time, observation_x2(1:iteration), 'r');
xlabel('t/s');
ylabel('幅度');
title('x2对比');

subplot(3, 1, 3);
plot(time, true_y(1:iteration), 'g');
hold on;
plot(time, observation_y(1:iteration), 'r');
xlabel('t/s');
ylabel('幅度');
title('y对比');

disp('仿真完成');
disp(['总共迭代次数: ', num2str(iteration)]);