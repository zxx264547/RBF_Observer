function [centers, sigma, weights] = rbf_train(x, y, min_centers, max_centers)
    % 将数据分为训练集和测试集
    num_samples = size(x, 1);
    train_ratio = 0.8;
    n_train = round(num_samples * train_ratio);

    % 随机打乱数据
    rand_indices = randperm(num_samples);
    x = x(rand_indices, :);
    y = y(rand_indices, :);

    % 划分训练集和测试集
    x_train = x(1:n_train, :);
    y_train_fault = y(1:n_train, :);
    x_test = x(n_train+1:end, :);
    y_test_fault = y(n_train+1:end, :);

    % RBF神经网络训练代码
    % 输入为五个变量，每个变量有n个样本点，中心点数量为num_centers，输出维度为1

    % 参数设置
    best_num_centers = min_centers;
    best_mse = inf;

    for num_centers = min_centers:max_centers
        % K均值聚类用于确定中心点
        [centers, ~] = kmeans(x_train, num_centers);

        % 计算RBF核函数的宽度参数（例如，使用中心之间的平均距离）
        d_max = pdist(centers);
        sigma = mean(d_max) / sqrt(2 * num_centers);
        
        % 计算训练集的RBF层输出
        n_train = size(x_train, 1);
        Phi_train = zeros(n_train, num_centers);
        for i = 1:n_train
            for j = 1:num_centers
                Phi_train(i, j) = exp(-norm(x_train(i, :) - centers(j, :))^2 / (2 * sigma^2));
            end
        end

        % 通过最小二乘法计算权重
        weights = pinv(Phi_train' * Phi_train) * Phi_train' * y_train_fault;

        % 计算测试集的RBF层输出
        n_test = size(x_test, 1);
        Phi_test = zeros(n_test, num_centers);
        for i = 1:n_test
            for j = 1:num_centers
                Phi_test(i, j) = exp(-norm(x_test(i, :) - centers(j, :))^2 / (2 * sigma^2));
            end
        end

        % 预测测试集输出
        y_pred = Phi_test * weights;

        % 计算均方误差 (MSE)
        mse = mean((y_test_fault - y_pred).^2, 'all');

        % 更新最佳中心点数量
        if mse < best_mse
            best_mse = mse;
            best_num_centers = num_centers;
        end
    end

    % 使用最佳中心点数量重新训练模型
    num_centers = best_num_centers;
    [centers, ~] = kmeans(x_train, num_centers);

    % 计算RBF核函数的宽度参数（例如，使用中心之间的平均距离）
    d_max = pdist(centers);
    sigma = mean(d_max) / (3 * sqrt(num_centers));

    % 计算训练集的RBF层输出
    n_train = size(x_train, 1);
    Phi_train = zeros(n_train, num_centers);
    for i = 1:n_train
        for j = 1:num_centers
            Phi_train(i, j) = exp(-norm(x_train(i, :) - centers(j, :))^2 / (2 * sigma^2));
        end
    end

    % 通过最小二乘法计算权重
    weights = pinv(Phi_train' * Phi_train) * Phi_train' * y_train_fault;

    % 计算测试集的RBF层输出
    n_test = size(x_test, 1);
    Phi_test = zeros(n_test, num_centers);
    for i = 1:n_test
        for j = 1:num_centers
            Phi_test(i, j) = exp(-norm(x_test(i, :) - centers(j, :))^2 / (2 * sigma^2));
        end
    end

    % 预测测试集输出
    y_pred = Phi_test * weights;

    % 显示结果
    num_outputs = size(y_test_fault, 2);
    for i = 1:num_outputs
        figure;
        plot(1:n_test, y_test_fault(:, i), 'b', 1:n_test, y_pred(:, i), 'r');
        xlabel('样本点');
        ylabel(['输出 ', num2str(i)]);
        legend('真实值', '预测值');
        title(['RBF神经网络测试结果 - 输出 ', num2str(i), ' (中心点数量: ', num2str(num_centers), ')']);
    end
end