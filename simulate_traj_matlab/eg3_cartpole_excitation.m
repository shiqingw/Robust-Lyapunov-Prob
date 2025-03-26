function dydt = eg3_cartpole_excitation(t,y)
    global params m M L g b am freq th;
    u = controller(y', params) + am * sin(2*pi*freq*t + th);
    x = y(1);
    theta = y(2);
    v = y(3);
    omega = y(4);

    x_dot = v;
    theta_dot = omega;
    det = M + m*sin(theta)^2;
    v_dot = (u - b * v - m * L * omega^2 * sin(theta)  + 0.5 * m * g * sin(2 * theta)) / det;
    omega_dot = (u * cos(theta) - 0.5 * m * L * omega^2 * sin(2 * theta)...
        - b * v * cos(theta) + (m + M) * g * sin(theta)) / (det * L);
    dydt = [x_dot; theta_dot; v_dot; omega_dot];
end