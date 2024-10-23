% 状态空间系统的参数
A = [0.5, 0.1, 0, 0, 0;
     0,   0.6, 0.2, 0, 0;
     0,   0,   0.7, 0.3, 0;
     0,   0,   0,   0.8, 0.4;
     0,   0,   0,   0,   0.9];

B = [0; 0; 0; 0; 1];  % 假设输入矩阵 B 为单一输入
C = [1, 1, 1, 1, 1];  % 输出矩阵 C (1x5)

% 检查系统的可控性
ControllabilityMatrix = ctrb(A, B);
if rank(ControllabilityMatrix) < size(A, 1)
    disp('系统不可控');
else
    disp('系统可控');
end

% 检查系统的可观测性
ObservabilityMatrix = obsv(A, C);
if rank(ObservabilityMatrix) < size(A, 1)
    disp('系统不可观测');
else
    disp('系统可观测');
end

% 修改权重矩阵 Q，可以根据需要调整
Q = diag([1, 1, 1, 1, 1]);  % 给予第一个状态更高的权重
a = 0.01;  % 进一步减小 a 的值

% 初始化 LMI 系统
setlmis([]);

% 定义 P 和 K
[P, nP] = lmivar(1, [size(A, 1) 1]);  % P 为对称矩阵 (5x5)
[K, nK] = lmivar(2, [size(A, 1) 1]);  % K 为 5x1 矩阵 (增益矩阵)

% 定义 LMI (Acl' * P + P * Acl + Q <= 0)，其中 Acl = A - K * C
% Acl' * P + P * Acl
lmiterm([1 1 1 P], A', 1, 's');        % A' * P + P * A
lmiterm([1 1 1 K], -1, C, 's');        % -K * C 项，加入到 Acl

% 去掉或进一步减弱 2 * a * P 项
lmiterm([1 1 1 P], a, 1);              % 减小 a 来减弱这一项

% 加入 Q 矩阵
lmiterm([1 1 1 0], Q);                 % 加入新的 Q

% 约束 P 为正定矩阵
lmiterm([2 1 1 P], 1, 1);              % P > 0 的约束

% 增加对增益 K 的范数限制 (Schur 补形式)
% t = lmivar(1, [1 1]);                  % 定义标量 t
% lmiterm([3 1 1 t], 1, 1);              % 对应 t 项
% lmiterm([3 1 2 K], 1, 1);              % 对应 K^T
% lmiterm([3 2 2 0], 1);                 % 单位矩阵 I

% 获取 LMI 系统并求解
lmis = getlmis;
[cost, xfeas] = feasp(lmis);

% 提取解
P_sol = dec2mat(lmis, xfeas, P);
K_sol = dec2mat(lmis, xfeas, K);

% 输出结果
disp('P 的解 = '), disp(P_sol);
disp('K 的解 = '), disp(K_sol);

% 检查 P 是否正定（所有特征值是否为正）
eig_P = eig(P_sol);
disp('P 的特征值 = '), disp(eig_P);

if all(eig_P > 0)
    disp('P 是正定矩阵');
else
    disp('P 不是正定矩阵');
end