% MATLAB仿真代码：带不确定项和故障信号的系统故障检测

% 初始化参数
% clear;
load rbf_model_parameters.mat
load KP_values.mat;
load system.mat;
% load KP_values.mat;
%% 系统参数初始化
Psi = B; % 故障影响矩阵,维度为输入*故障信号维数

% 初始化系统状态和观测器
x = zeros(state_dim, 1); % 初始系统状态
x_hat = zeros(state_dim, 1); % 初始观测器状态
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
W1 = zeros(state_dim, model_num_neurons); 
W2 = zeros(input_dim, fault_num_neurons); 

% 自适应率的参数设置
L1 = eye(state_dim); 
L2 = eye(input_dim); 
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
state_observation_error_x = zeros(state_dim, max_iterations);
true_x = zeros(state_dim, max_iterations);
observation_x = zeros(state_dim, max_iterations);
true_y = zeros(max_iterations, 1);
observation_y = zeros(max_iterations, 1);

%% 开始迭代
for iteration = 1:max_iterations
    % 当前时刻
    t = time(iteration); 
    % 输入信号
    u = sys_input(t);
    % 不确定项
    eta = uncertain(x);
    % 故障信号
    if t >= fault_time
        theta = fault(x);
    else
        theta = 0; % 无故障
    end
    
   
    %% 真实系统
    % 系统线性部分
    f = A * x + B * u;
    
    
    % 系统状态方程
    x_dot = f + eta + Psi * theta;
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
    true_fault(iteration) = theta;
    observed_fault(iteration, 1) = theta_hat; % 记录观测到的故障信号
    state_observation_error_x(:,iteration) = abs(x - x_hat); 
    true_x(:,iteration) = x;
    observation_x(:,iteration) = x_hat;
    true_y(iteration) = y;
    observation_y(iteration) = y_hat;
end

%% 绘制定理1故障观测结果和状态观测误差曲线
figure('Name',"对比图");
subplot(2, 2, 1);
plot(time, true_fault(1:iteration),'DisplayName','真实故障');
hold on;
plot(time, observed_fault(1:iteration),'DisplayName','观测故障');
xlabel('t/s');
ylabel('幅度');
title('故障信号对比');
legend show;

subplot(2, 2, 2);
plot(time, true_fault(1:iteration)-observed_fault(1:iteration));
xlabel('t/s');
ylabel('幅度');
title('故障误差');

subplot(2, 2, 3);
plot(time, true_y(1:iteration),'DisplayName','真实输出');
hold on;
plot(time, observation_y(1:iteration),'DisplayName','观测输出');
xlabel('t/s');
ylabel('幅度');
title('输出对比');
legend show;

subplot(2, 2, 4);
plot(time, true_y(1:iteration)-observation_y(1:iteration));
xlabel('t/s');
ylabel('幅度');
title('输出误差');


figure('Name',"状态真实值与观测值对比");
for i = 1:state_dim
    subplot(state_dim, 1, i);
    plot(time, true_x(i, 1:iteration), 'g');
    hold on;
    plot(time, observation_x(i, 1:iteration), 'r');
    xlabel('t/s');
    ylabel('幅度');
    title(sprintf('x%d对比', i));
    legend(sprintf('真实x%d',i),sprintf('观测x%d',i));
end

figure('Name',"状态真实值与观测值误差");
for i = 1:state_dim
    subplot(state_dim, 1, i);
    plot(time, state_observation_error_x(i, 1:iteration));
    xlabel('t/s');
    ylabel('幅度');
    title(sprintf('x%d误差', i));
end

disp('仿真完成');
disp(['总共迭代次数: ', num2str(iteration)]);
     
