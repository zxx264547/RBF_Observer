% 定义已知矩阵 A 和 C
load system.mat

% 定义矩阵 Q（应为正定矩阵）和参数 alpha
Q = eye(size(A)); % 使用单位矩阵作为正定矩阵
alpha = 0.1; % 可以根据具体情况选择

% 使用可观测性判断来计算矩阵 K
if rank(obsv(A, C)) == size(A, 1)
    % 系统是可观测的，选择合适的增益矩阵 K
    % 使用 place 函数来选择合适的极点位置，使得 (A - KC) 稳定
    % 定义极点范围
    min_pole = -2.25; % 极点的最小值
    max_pole = -2; % 极点的最大值
    
    % 随机生成极点，确保在 [min_pole, max_pole] 范围内的小数
    desired_poles = min_pole + (max_pole - min_pole) * rand(1, state_dim);

    K = place(A', C', desired_poles)'; 
else
    error('系统不可观测，无法求解矩阵 K');
end

% 定义矩阵 A_KC
A_KC = A - K * C;

% 求解 Lyapunov 方程来得到矩阵 P
P = lyap((A_KC)', Q + 2 * alpha * eye(size(A)));

% 验证矩阵 P 是否为正定
if all(eig(P) > 0)
    disp('矩阵 P 是正定的');
else
    error('矩阵 P 不是正定的，请调整极点位置或矩阵 Q');
end

% 输出结果
disp('矩阵 K 为：');
disp(K);
disp('矩阵 P 为：');
disp(P);

% 将 K 和 P 保存到 mat 文件中
save('KP_values.mat', 'K', 'P');
