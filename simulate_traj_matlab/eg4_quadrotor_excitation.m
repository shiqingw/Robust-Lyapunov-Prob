function dsdt = eg4_quadrotor_excitation(t,s)
    global params m r I g am freq th;
    u = controller(s', params);
    u1 = u(1) + am * sin(2*pi*freq*t + th);
    u2 = u(2);
    
    x = s(1);
    y = s(2);
    theta = s(3);
    vx = s(4);
    vy = s(5);
    omega = s(6);

    x_dot = vx;
    y_dot = vy;
    theta_dot = omega;
    vx_dot = -(u1 + u2)*sin(theta)/m - g*sin(theta);
    vy_dot = (u1 + u2)*cos(theta)/m + g*(cos(theta)-1);
    omega_dot = r*(-u1+u2)/I;
    dsdt = [x_dot; y_dot; theta_dot; vx_dot; vy_dot; omega_dot];
end