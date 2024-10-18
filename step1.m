%% 初始化 
clc
clear
%% 系统设计
A = [0, 1; -2, -3];
B = [0; 1];
C = [1, 0];
K = [0.2361; 0.2361];
A_KC = A - K * C;
Q = [1 0;0 1];
alpha = 1;
%% LMI设计
P=sdpvar(2,2,'symmetric');
a=P*A_KC+A_KC'*P+2*alpha*P+Q;
%% 求解LMI
J=(a<=0);
ops=sdpsettings('solver','mosek');
sol=optimize(J,[],ops);
check(J)
if sol.problem == 0
disp('Congratuations! Solutions are found!');
P=value(P);
end
eig(P)
%% 看看结果
% h=0.01;T=100;
% n=h:h:T;
% e(1:2,1:T/h)=0;
% e(:,1)=[0.01;0.01];
% for i=1:length(n)-1
% e(:,i+1)=(h*(D-L*B*E)+eye(size((D-L*B*E))))*e(:,i);
% end
% plot(n,e(1,:));hold on
% plot(n,e(2,:));hold on