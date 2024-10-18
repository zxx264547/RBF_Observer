% 设定参数 alpha 和正定矩阵 Q
alpha = 0.5;
Q = eye(2);  % 单位矩阵，确保其为正定

% 选择期望的观测器特征值（必须具有负的实部，以确保稳定性）
% 例如，我们希望误差动态矩阵的特征值为 -2 和 -3
desired_poles = [-2, -3];

% 计算观测器增益矩阵 K
% 注意：由于 MATLAB 的 'place' 函数用于状态反馈，我们需要对 A 和 C 进行转置
K = place(A', C', desired_poles)';

% 计算误差动态矩阵 A_e
A_e = A - K * C;

% 计算修正后的矩阵 S，用于李雅普诺夫方程
S = A_e + alpha * eye(size(A));

% 解决李雅普诺夫方程 S' * P + P * S = -Q，求解对称正定矩阵 P
P = lyap(S', Q);

% 检查 P 是否为正定矩阵（可选）
eig_P = eig(P);
if all(eig_P > 0)
    disp('P 是正定矩阵。');
else
    disp('P 不是正定矩阵，请检查计算。');
end

% 显示计算结果
disp('观测器增益矩阵 K：');
disp(K);

disp('李雅普诺夫矩阵 P：');
disp(P);

% 将 K 和 P 保存到 mat 文件中
save('system.mat', 'A', 'B', 'C', 'K', 'P');
