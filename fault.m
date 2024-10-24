function result = fault(x)
    result = 1;
    for i = 1:size(x,1)
        result = result + 0.1 * sin(x(i) * 10);
    end
    
end