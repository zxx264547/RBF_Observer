function result = uncertain(x)
    x_dim = size(x,1);
    result = zeros(x_dim,1);
    for i = 1:x_dim
        result(i) = 0.1 * sin(x(i)) * cos(x(i));
    end
end