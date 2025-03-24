function dydt = eg2_backstepping_excitation(t,y)
    global params a1 a2 b1 b2 c1 c2 am freq th;
    u = controller(y', params) + am * sin(2*pi*freq*t + th);
    x1 = y(1);
    x2 = y(2);
    x3 = y(3);
    x1_dot = a1 * x1 + a2 * x2;
    x2_dot = b1 * x2 + b2 * x3;
    x3_dot = c1 * x1^2 + c2 *u;
    dydt = [x1_dot; x2_dot; x3_dot];
end