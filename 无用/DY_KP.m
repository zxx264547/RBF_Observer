load system.mat
Q = zeros(size(A,1));
alpha = 0.03;
%% LMI设计
P=sdpvar(size(A, 1),size(A, 1),'symmetric');
P_=sdpvar(size(A, 1),1); %%P_=P*K
a=A'*P-C'*P_'+P*A-P_*C+2*alpha*P+Q;
%%a=P*A_KC+A_KC'*P+2*alpha*P+Q;
%% 求解LMI
J=(a<=0)+(P>=0);
ops=sdpsettings('solver','mosek');
sol=optimize(J,[],ops);
check(J)
if sol.problem == 0
    disp('Congratuations! Solutions are found!');
    P=value(P);
    P_=value(P_);
    E=eig(P);
end
K = inv(P) * P_;
% 显示计算结果
disp('观测器增益矩阵 K：');
disp(K);
disp('李雅普诺夫矩阵 P：');
disp(P);
% 将 K 和 P 保存到 mat 文件中
save('KP_values.mat', 'K', 'P');