function dydt = eg1_inv_pend(t,y)
    global params m L b g inertia d0 d1 d2;
    u = controller(y', params);
    angle = y(1);
    angular_velocity = y(2);
    x_ddot = g / L * sin(angle) + u / inertia - b / inertia * angular_velocity;
    x_ddot = x_ddot + d0*sin(t) - d1*angular_velocity - d2*angular_velocity^2;
    dydt = [angular_velocity; x_ddot];
end