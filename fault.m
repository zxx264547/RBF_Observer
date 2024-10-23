function result = fault(x)
    result = 2;
    for i = 1:size(x,1)
        result = result + 0.1 * sin(x(i));
    end
    
end