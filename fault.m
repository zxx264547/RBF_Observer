function result = fault(x)
    z = 0;
    for i = 1:size(x,1)
        z = z + x(i);
    end
    result = 1/(exp(-z) + 1);
end