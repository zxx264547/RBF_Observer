function u = sys_input(t)
    u = zeros(1,length(t));
    for i = 1:length(t)
    u(1,i) = 0.5 * sin(2 * pi * t(i)) + 0.5 * cos(0.008 * pi * t(i));
    end
end