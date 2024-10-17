% MATLAB仿真代码：去掉阈值的故障检测实现

% 初始化参数
clear; clc;close all

% 设置仿真参数
max_iterations = 1000; % 最大迭代次数
T = 1 / (0.1 * max_iterations);
learning_rate = 0.01; % 权值更新的学习率
iteration = 0; % 当前迭代次数

% 初始化系统状态和观测器
x = [0; 1]; % 初始系统状态
x_hat = [0; 1]; % 初始观测器状态
y = x(1) + x(2); % 系统输出
y_hat = x_hat(1) + x_hat(2); % 观测器输出

% 假设已经训练好的RBF神经网络参数
num_neurons = 10; % RBF神经元数量

% 训练后的RBF的中心位置（二维情况）
centers = [1,0;
          1,0;
          1,0;
          1,0;
          1,0;
          1,0;
          1,0;
          1,0;
          1,0;
          1,0;];

% 训练后的RBF宽度
widths = 0.05 * ones(1, num_neurons);

% 权值矩阵初始化
W1 = randn(2, num_neurons) * 0.1; % 初始化为2x10的随机值
W2 = randn(2, num_neurons) * 0.1; % 同样初始化为2x10的随机值

% 论文中的参数设置
L1 = eye(2); % L1 矩阵，设置为2x2单位矩阵
L2 = eye(2); % L2 矩阵，设置为2x2单位矩阵
P = [0.9509,-0.1161;-0.1161,0.2253]; % P 矩阵
C = [1, 1]; % 输出矩阵
Psi = [1; 1]; % 故障影响矩阵
lambda1 = 0.03; % 学习率参数
lambda2 = 0.03; % 学习率参数

% 阈值计算
epsilon = 1e-6; % 允许的误差界
W1_bar = 0.5; % 神经网络权值矩阵的范数上界
phi_bar = 1; % 激活函数的上界

% 故障输入设置（震荡信号从500次迭代开始，中心为2，幅值为0.5）
fault_time = 500; % 故障开始的迭代次数
fault_frequency = 0.05; % 故障信号的频率
fault_amplitude = 1; % 故障信号的幅度
fault_center = 2; % 故障信号的震荡中心

% 数据记录
true_fault = zeros(max_iterations, 1);
observed_fault = zeros(max_iterations, 1);
state_observation_error_x1 = zeros(max_iterations, 1);
state_observation_error_x2 = zeros(max_iterations, 1);

%% 定理1仿真循环
while iteration < max_iterations
    % 计算系统的状态变化（简单非线性系统的模拟）
    u = 0.5 * sin(2 * pi * 0.1 * iteration) + 0.5 * cos(0.08 * pi * iteration); % 输入
    eta1=-sin(x(2))-0.1*sin(x(1)*cos(x(2)))+3*iteration;
    eta2=0.1*sin(x(1))*cos(x(2));
    % 设置故障信号：在迭代次数超过500时开始震荡
    if iteration >= fault_time
        theta = fault_center + fault_amplitude * sin(2 * pi * fault_frequency * (iteration - fault_time));
    else
        theta = 0; % 故障未发生时
    end
    
    
    % 记录真实的故障输入
    true_fault(iteration + 1) = theta;
    
    % 系统状态方程包含故障输入 
    dx = [-x(1)^2 - x(1) * x(2); -x(2)^2 + 4 * x(1) * x(2)] + [eta1; eta2] + [0.25; 2] * u + [1; 1] * theta;
    x = x + dx * T; % 更新状态
    y = x(1) + x(2); % 输出更新
    
    % 计算RBF激活函数输出
    phi1 = exp(-sum((x - centers').^2, 1) ./ (2 * widths.^2))';
    phi2 = exp(-sum((x_hat - centers').^2, 1) ./ (2 * widths.^2))';
    
    % 计算观测误差
    y_epsilon = y - y_hat;
    
    % 权值更新公式中的各项计算
    norm_C = norm(C, 'fro');
    norm_y_epsilon = norm(y_epsilon, 'fro');

    norm_y_epsilon_CP = norm((y_epsilon' * C) * P, 'fro');
    norm_y_epsilon_CP_Psi = norm((y_epsilon' * C) * P * Psi, 'fro');
    
    % 计算 W1 和 W2 的更新
    W1_dot = (L1 * ((P * C') * y_epsilon) * phi1' - lambda1 * norm_y_epsilon_CP * L1 * W1) / (norm_C^2);
    W2_dot = (L2 * ((P * C') * y_epsilon) * phi2' - lambda2 * norm_y_epsilon_CP_Psi * L2 * W2) / (norm_C^2);
    
    if norm_y_epsilon > 1
        norm_y_epsilon = 0;
    end
    if norm_y_epsilon <= 1
        norm_y_epsilon = 0;
        W1_dot = (L1 * ((P * C') * y_epsilon) * phi1' - lambda1 * norm_y_epsilon_CP * L1 * W1) / (norm_C^2);
    end
    % 更新权值矩阵
    W1 = W1 + learning_rate * W1_dot;
    W2 = W2 + learning_rate * W2_dot;
    
    % 更新观测器的状态
    dx_hat = [-x_hat(1)^2 - x_hat(1) * x_hat(2); -x_hat(2)^2 + 4 * x_hat(1) * x_hat(2)] +[eta1; eta2]+ [0.25; 2] * u;
    x_hat = x_hat + dx_hat * T; % 观测器状态更新
    y_hat = x_hat(1) + x_hat(2); % 更新观测器输出
    
    % 记录故障观测和状态观测误差
    fn = W2 * phi2;
    observed_fault(iteration + 1,1) = fn(1,:); % 记录观测到的故障信号
    state_observation_error_x1(iteration + 1) = abs(x(1) - x_hat(1)); % 记录x1的观测误差
    state_observation_error_x2(iteration + 1) = abs(x(2) - x_hat(2)); % 记录x2的观测误差
    
    % 迭代次数增加
    iteration = iteration + 1;
end

 % %更新权值矩阵
 %     if norm(y_epsilon)>= dead_zone_threshold
 %         W1_theorem2=W1_theorem2+gamma_theorem2*W1_theorem_dot;
 %          W2_theorem2=W2_theorem2+gamma_theorem2*W2_theorem_dot;
 %     end


 
%% 绘制定理1故障观测结果和状态观测误差曲线
figure;
time = [1:iteration]/100;
subplot(3,1,1);
plot(time, true_fault(1:iteration), 'g');
xlabel('t/s');
ylabel('幅度');
title('真实故障信号');

subplot(3,1,2);
plot(time, observed_fault(1:iteration), 'b');
xlabel('t/s');
ylabel('幅度');
title('观测故障信号');

subplot(3,1,3);
plot(time, state_observation_error_x1(1:iteration), 'r', 'DisplayName', 'x1误差');
hold on;
plot(time, state_observation_error_x2(1:iteration), 'b', 'DisplayName', 'x2误差');
hold off;
xlabel('t/s');
ylabel('幅度');
title('状态的观测误差曲线');
legend('x1误差', 'x2误差');

disp('仿真完成');
disp(['总共迭代次数: ', num2str(iteration)]);
