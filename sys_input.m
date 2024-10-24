function u = sys_input(t)
    u = zeros(length(t),1);
    for i = 1:length(t)
    u(i,1) = 0.5 * sin(2 * pi * t(i)) + 0.5 * cos(0.008 * pi * t(i));
    end
end