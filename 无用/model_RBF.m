% RBF神经网络选择最优节点数量示例，并绘制损失曲线和拟合结果（多维输入）
clear;

% 加载数据
load('rbf_training_data.mat');



% 2. 设置交叉验证参数
max_centers = 50; % 最大节点数量
num_folds = 5; % 交叉验证的折数
num_iterations = 100; % 训练迭代次数
learning_rate = 0.01; % 学习率

% 3. 分层采样选择候选中心点
num_candidates = 50; % 候选中心点数量
% 生成多维输入的候选中心
candidate_centers = datasample(x_train, num_candidates); % 随机选择候选中心

% 4. 分割数据用于交叉验证
indices = crossvalind('Kfold', size(x_train, 1), num_folds);

% 5. 交叉验证，遍历不同的节点数量
validation_errors = zeros(max_centers, 1);
loss_history_all = zeros(max_centers, num_iterations);

for num_centers = 1:max_centers
    fold_errors = zeros(num_folds, 1);
    loss_history = zeros(num_iterations, 1); % 临时变量记录每个节点数量的损失变化
    
    for fold = 1:num_folds
        % 划分训练集和验证集
        test_idx = (indices == fold);
        train_idx = ~test_idx;
        
        x_train_fold = x_train(train_idx, :);
        y_train_model_fold = y_train_model(train_idx, :);
        x_val_fold = x_train(test_idx, :);
        y_val_fold = y_train_model(test_idx, :);
        
        % 正交最小二乘法选择中心点
        model_centers = candidate_centers(1:num_centers, :);
        model_sigma = ones(num_centers, 1) * mean(pdist(model_centers));
        
        % 计算径向基函数输出
        G_train = zeros(length(x_train_fold), num_centers);
        for iter = 1:num_iterations
            for i = 1:num_centers
                % 计算每个样本点到各中心的欧氏距离，并计算径向基函数输出
                G_train(:, i) = exp(-sum((x_train_fold - model_centers(i, :)).^2, 2) / (2 * model_sigma(i)^2));
            end
            
            % 训练输出权重
            weights = pinv(G_train) * y_train_model_fold;
            
            % 验证集预测
            G_val = zeros(length(x_val_fold), num_centers);
            for i = 1:num_centers
                G_val(:, i) = exp(-sum((x_val_fold - model_centers(i, :)).^2, 2) / (2 * model_sigma(i)^2));
            end
            y_val_pred = G_val * weights;
            
            % 计算损失函数（均方误差）
            loss = mean((y_train_model_fold - G_train * weights).^2, 'all');
            loss_history(iter) = loss; % 记录每次迭代的损失
            
            % 更新宽度参数（梯度下降）
            for i = 1:num_centers
                % 计算径向基函数的导数 dG/dsigma
                dG_dsigma = G_train(:, i) .* (sum((x_train_fold - model_centers(i, :)).^2, 2) / (model_sigma(i)^3));
                % 计算损失函数对 model_sigma 的梯度
                gradient = sum((G_train * weights - y_train_model_fold) .* (weights(i) * dG_dsigma), 'all') / length(x_train_fold);
                % 更新宽度参数
                model_sigma(i) = model_sigma(i) - learning_rate * gradient;
            end
        end
        
        % 计算当前fold的验证误差
        fold_errors(fold) = mean((y_val_fold - y_val_pred).^2, 'all');
    end
    
    % 计算当前节点数量的平均验证误差
    validation_errors(num_centers) = mean(fold_errors);
    loss_history_all(num_centers, :) = loss_history; % 保存每个节点数量对应的损失历史
end
%% 

% 6. 找到使验证误差最小的节点数量
[~, optimal_num_centers] = min(validation_errors);
fprintf('最优节点数量为：%d\n', optimal_num_centers);

% 7. 绘制每个节点数量的损失随迭代次数的变化
figure;
hold on;
for num_centers = 1:max_centers
    plot(1:num_iterations, loss_history_all(num_centers, :), 'LineWidth', 1);
end
xlabel('迭代次数');
ylabel('损失');
title('不同节点数量的损失随迭代次数的变化');
grid on;
hold off;

% 8. 绘制最优节点数量的模型输出与实际数据的结果
% 重新训练最优节点数量的RBF网络
model_centers = candidate_centers(1:optimal_num_centers, :);
model_sigma = ones(optimal_num_centers, 1) * mean(pdist(model_centers));
G_train_optimal = zeros(size(x_train, 1), optimal_num_centers);
for i = 1:optimal_num_centers
    G_train_optimal(:, i) = exp(-sum((x_train - model_centers(i, :)).^2, 2) / (2 * model_sigma(i)^2));
end
model_weights_optimal = pinv(G_train_optimal) * y_train_model;

% 预测最优节点数量下的结果
x_test = x_train; % 使用相同的x范围测试
G_test_optimal = zeros(length(x_test), optimal_num_centers);
for i = 1:optimal_num_centers
    G_test_optimal(:, i) = exp(-sum((x_test - model_centers(i, :)).^2, 2) / (2 * model_sigma(i)^2));
end
y_test_model_pred_optimal = G_test_optimal * model_weights_optimal;

% 绘制模型输出与实际数据的对比结果
figure;
plot(1:length(y_train_model), y_train_model, 'b-', 'LineWidth', 1.5, 'DisplayName', '实际输出');
hold on;
plot(1:length(y_test_model_pred_optimal), y_test_model_pred_optimal, 'r--', 'LineWidth', 1.5, 'DisplayName', '模型预测输出');
xlabel('样本编号');
ylabel('输出值');
title('模型输出与实际输出的对比');
legend;
grid on;
hold off;
% 定义保存文件的名称
save_file_name = 'optimal_model_rbf_parameters.mat';

% 保存最优参数
save(save_file_name, 'model_centers', 'model_sigma', 'model_weights_optimal');

% 显示保存成功的提示
fprintf('最优参数已保存至 %s 文件中。\n', save_file_name);
