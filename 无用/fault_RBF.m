% 清空工作空间，确保之前的变量不会影响本次运行
clear;

% 1. 加载训练数据
load('rbf_training_data.mat'); % 假定此文件中包含变量 x_train 和 y_train_fault

% 2. 设置交叉验证的相关参数
max_centers = 50; % 最大节点数量，即径向基函数的中心数量
min_centers = 5;
num_folds = 5; % k折交叉验证的折数
num_iterations = 100; % 每次训练的迭代次数
learning_rate = 0.01; % 用于梯度下降的学习率

% 3. 分层采样，选择候选中心点
num_candidates = 50; % 候选中心点的数量
candidate_centers = datasample(x_train, num_candidates); % 从训练数据中随机采样候选中心点

% 4. 分割数据用于交叉验证
indices = crossvalind('Kfold', size(x_train, 1), num_folds); % 生成交叉验证的索引

% 5. 开始交叉验证，遍历不同的节点数量
validation_errors = zeros(max_centers, 1); % 保存验证误差
loss_history_all = zeros(max_centers, num_iterations); % 保存损失历史

for num_centers = min_centers:max_centers
    fold_errors = zeros(num_folds, 1); % 保存每一折的误差
    loss_history = zeros(num_iterations, 1); % 记录损失随迭代变化

    for fold = 1:num_folds
        % 划分训练集和验证集
        test_idx = (indices == fold); % 当前fold的验证集索引
        train_idx = ~test_idx; % 其余为训练集索引
        
        % 提取当前fold的训练数据和验证数据
        x_train_fold = x_train(train_idx, :);
        y_train_fault_fold = y_train_fault(train_idx, :);
        x_val_fold = x_train(test_idx, :);
        y_val_fold = y_train_fault(test_idx, :);
        
        % 正交最小二乘法选择中心点，前 num_centers 个候选中心作为当前模型的中心点
        fault_centers = candidate_centers(1:num_centers, :);
        fault_sigma = ones(num_centers, 1) * mean(pdist(fault_centers)); % 初始宽度参数[num_centers × 1]

        % 循环迭代训练模型
        G_train = zeros(size(x_train_fold, 1), num_centers); % 保存径向基函数输出(样本数*中心点数）
        for iter = 1:num_iterations
            for i = 1:num_centers
                % 计算每个样本点到各中心的欧氏距离，并计算径向基函数的输出
                G_train(:, i) = exp(-sum((x_train_fold - fault_centers(i, :)).^2, 2) / (2 * fault_sigma(i)^2));
            end
            
            % 训练输出权重
            weights = pinv(G_train) * y_train_fault_fold; % 使用伪逆求解权重
            
            % 验证集预测
            G_val = zeros(length(x_val_fold), num_centers);
            for i = 1:num_centers
                G_val(:, i) = exp(-sum((x_val_fold - fault_centers(i, :)).^2, 2) / (2 * fault_sigma(i)^2));
            end
            y_val_pred =  G_val * weights; % 预测验证集输出
            
            % 计算损失函数（均方误差）
            loss = mean((y_train_fault_fold - G_train * weights).^2, 'all');
            loss_history(iter) = loss; % 记录当前迭代的损失
            
            % 更新径向基函数宽度参数，使用梯度下降
            for i = 1:num_centers
                % 计算径向基函数对 sigma 的导数
                dG_dsigma = G_train(:, i) .* (sum((x_train_fold - fault_centers(i, :)).^2, 2) / (fault_sigma(i)^3));
                gradient = sum((G_train * weights - y_train_fault_fold) .* (weights(i) * dG_dsigma), 'all') / length(x_train_fold);
                fault_sigma(i) = fault_sigma(i) - learning_rate * gradient; % 更新 sigma
            end
        end
        
        % 计算验证误差
        fold_errors(fold) = mean((y_val_fold - y_val_pred).^2, 'all');
    end
    
    % 平均验证误差
    validation_errors(num_centers) = mean(fold_errors);
    loss_history_all(num_centers, :) = loss_history; % 保存每个节点数量的损失历史
end

% 6. 找到验证误差最小的最优节点数量
[~, optimal_num_centers] = min(validation_errors);
fprintf('最优节点数量为：%d\n', optimal_num_centers);

% 7. 绘制每个节点数量的损失随迭代次数的变化（已注释）

% 8. 使用最优节点数量重新训练RBF网络
fault_centers = candidate_centers(1:optimal_num_centers, :);
fault_sigma = ones(optimal_num_centers, 1) * mean(pdist(fault_centers));

% 计算最优节点数量下的径向基函数输出
G_train_optimal = zeros(size(x_train, 1), optimal_num_centers);
for i = 1:optimal_num_centers
    G_train_optimal(:, i) = exp(-sum((x_train - fault_centers(i, :)).^2, 2) / (2 * fault_sigma(i)^2));
end
fault_weights_optimal = pinv(G_train_optimal) * y_train_fault; % 最优权重

% 预测最优节点数量下的结果
x_test = x_train; % 使用相同的输入数据进行预测
G_test_optimal = zeros(length(x_test), optimal_num_centers);
for i = 1:optimal_num_centers
    G_test_optimal(:, i) = exp(-sum((x_test - fault_centers(i, :)).^2, 2) / (2 * fault_sigma(i)^2));
end
y_test_fault_pred_optimal = G_test_optimal * fault_weights_optimal;

% 绘制模型预测输出与实际输出的对比
figure;
plot(1:length(y_train_fault), y_train_fault, 'b-', 'LineWidth', 1.5, 'DisplayName', '实际输出');
hold on;
plot(1:length(y_test_fault_pred_optimal), y_test_fault_pred_optimal, 'r--', 'LineWidth', 1.5, 'DisplayName', '模型预测输出');
xlabel('样本编号');
ylabel('输出值');
title('模型输出与实际输出的对比');
legend;
grid on;
hold off;

% 保存最优参数到文件
save_file_name = 'optimal_fault_rbf_parameters.mat';
save(save_file_name, 'fault_centers', 'fault_sigma', "fault_weights_optimal");
fprintf('最优参数已保存至 %s 文件中。\n', save_file_name);
