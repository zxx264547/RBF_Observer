%% ��ʼ�� 
clc
clear
%% ϵͳ���
A = [0, 1; -2, -3];
B = [0; 1];
C = [1, 0];

Q = [1 0; 0 1];
alpha = 1;

%% LMI���
P = sdpvar(2, 2, 'symmetric');

K_values = -10:1:10;  % K������Ԫ�صķ�Χ
found_solution = false;  % ��־λ���ж��Ƿ��ҵ���

for k1 = K_values
    for k2 = K_values
        K = [k1; k2];  % ����K����
        A_KC = A - K * C;
        a = P * A_KC + A_KC' * P + 2 * alpha * P + Q;

        %% ���LMI
        J = (a <= 0);
        ops = sdpsettings('solver', 'mosek');
        sol = optimize(J, [], ops);  % ������ⷽ������

        if sol.problem == 0
            P_value = value(P);
            % ���P�����Ƿ�����
            eigenvalues = eig(P_value);
            if all(eigenvalues > 0)
                disp(['�ҵ�������P����KֵΪ: [', num2str(k1), ', ', num2str(k2), ']']);
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
    disp('δ�ҵ����ʵ�K��P����');
end
