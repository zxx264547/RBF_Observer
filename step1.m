%% 初始化 
clc
clear
%% 系统设计
A = [0, 1; -2, -3];
B = [0; 1];
C = [1, 0];

Q = [1 0; 0 1];
alpha = 1;

%% LMI设计
P = sdpvar(2, 2, 'symmetric');

K_values = -10:1:10;  % K的两个元素的范围
found_solution = false;  % 标志位，判断是否找到解

for k1 = K_values
    for k2 = K_values
        K = [k1; k2];  % 调整K矩阵
        A_KC = A - K * C;
        a = P * A_KC + A_KC' * P + 2 * alpha * P + Q;

        %% 求解LMI
        J = (a <= 0);
        ops = sdpsettings('solver', 'mosek');
        sol = optimize(J, [], ops);  % 保持求解方法不变

        if sol.problem == 0
            P_value = value(P);
            % 检查P矩阵是否正定
            eigenvalues = eig(P_value);
            if all(eigenvalues > 0)
                disp(['找到正定的P矩阵，K值为: [', num2str(k1), ', ', num2str(k2), ']']);
                P = P_value;
                found_solution = true;
                break;
            end
        end
    end
    if found_solution
        break;
    end
end

if found_solution
    disp('Congratuations! Solutions are found!');
    save('K_P_results.mat', 'K', 'P');
else
    disp('未找到合适的K和P矩阵。');
end
