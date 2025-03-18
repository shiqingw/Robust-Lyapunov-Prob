function dydt = eg1_inv_pend_excitation(t,y)
    global params m L b g inertia am freq th;
    u = controller(y', params) + am * sin(2*pi*freq*t + th);
    angle = y(1);
    angular_velocity = y(2);
    x_ddot = g / L * sin(angle) + u / inertia - b / inertia * angular_velocity;
    dydt = [angular_velocity; x_ddot];
end
