clear all;
rng('default');
filepath = fileparts(mfilename('fullpath'));
parts = strsplit(filepath, filesep);
parent_path = strjoin(parts(1:end-1), filesep);
result_dir = fullfile(parent_path, 'eg1_results', '001');
controller_dir = result_dir;
modelfile = fullfile(controller_dir, 'controller.onnx');

%% 
global params m L b g inertia am freq th;
params = importONNXFunction(modelfile, "controller");
m = 0.8; 
L = 0.35; 
b = 0.0;
g = 9.81;
inertia = m*L^2;
num_sine = 10;
am = 0.0*ones(1,num_sine);
freq = zeros(num_sine,1);
freq(1:4) = [1;3;5;7];
freq(5:end) = rand(num_sine-4, 1)*10;                
th = (rand(num_sine, 1)-0.5)*pi;

%% 
state_space = [-pi, pi;
               -pi, pi];
state_dim = size(state_space,1);

points_per_dim = [30, 2];
x1 = linspace(state_space(1,1), state_space(1,2), points_per_dim(1));
x2 = linspace(state_space(2,1), state_space(2,2), points_per_dim(2));
[X1,X2] = ndgrid(x1,x2);

initial_states = zeros(prod(points_per_dim),state_dim);
initial_states(:,1) = reshape(X1,[],1);
initial_states(:,2) = reshape(X2,[],1);

n_initial_states = size(initial_states,1);
true_or_false = false(n_initial_states,1);
for i = 1:state_dim
    lb = state_space(i,1);
    ub = state_space(i,2);
    cc = initial_states(:,i);
    true_or_false = (cc==lb | true_or_false);
    true_or_false = (cc==ub | true_or_false);
end
initial_states = initial_states(true_or_false, :);
n_initial_states = size(initial_states,1);

%% 
T = 10;
dt = 0.01;
n = T/dt;
cutoff_radius = 0.2;
threshold = 0.2;

fig1 = figure(1);

x = zeros(n*n_initial_states, state_dim);
x_dot = zeros(n*n_initial_states, state_dim);
u = zeros(n*n_initial_states, 1);
t = zeros(n*n_initial_states, 1);
data_size = 0;
num_kept = 0;

for ii = 1:size(initial_states,1)
    if mod(ii+1, 5)== 0
        fprintf("> Simulated %d/%d trajs\n", ii+1, size(initial_states,1));
    end

    t_tmp = linspace(0,T,n+1);
    sol = ode15s(@eg1_inv_pend_excitation,[0 T],initial_states(ii,:)');
    y_tmp = deval(sol,t_tmp);
    x_tmp = y_tmp(:,1:n)';
    x_dot_tmp = diff(y_tmp')/dt;
    t_tmp = t_tmp(1:n)';
    u_tmp = zeros(n,1);
    
    for i = 1:n
        u_tmp(i) = controller(y_tmp(:,i)', params) + am * sin(2*pi*freq*t_tmp(i) + th);
    end

    x_norm = vecnorm(x_tmp, 2, 2);
    true_or_false = x_norm >= cutoff_radius;
    if min(x_norm) >= threshold
        continue 
    end
    x_tmp = x_tmp(true_or_false,:);
    x_dot_tmp = x_dot_tmp(true_or_false,:);
    t_tmp = t_tmp(true_or_false,:);
    u_tmp = u_tmp(true_or_false,:);

    current_data_size = size(x_tmp,1);
    x(data_size+1:data_size+current_data_size,:) = x_tmp;
    x_dot(data_size+1:data_size+current_data_size,:) = x_dot_tmp;
    u(data_size+1:data_size+current_data_size,:) = u_tmp;
    t(data_size+1:data_size+current_data_size,:) = t_tmp;

    data_size = data_size + current_data_size;
    num_kept = num_kept + 1;

    plot(x_tmp(:,1), x_tmp(:,2), 'LineWidth', 1);
    hold on;
end

%% 
x0 = 10;
y0 = 10;
width = 550;
height = 550;
fig1.Position = [x0 y0 width height]; 
xlim([-pi,pi]);
ylim([-pi,pi]);
extraInputs = {'fontsize',30,'FontName','Serif','Interpreter','latex'};
ticksize = 30;

ax = gca;
ax.LineWidth = 1.5;
xaxisproperties= get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex';
xaxisproperties.FontSize = ticksize;
yaxisproperties= get(gca, 'YAxis');
yaxisproperties.TickLabelInterpreter = 'latex';
yaxisproperties.FontSize = ticksize;
xlabel('$\theta$', extraInputs{:});
ylabel('$\omega$', extraInputs{:});
timestamp = datetime;
timestamp.Format = 'yyyy-MM-dd_HH-mm-ss';
fig_name = sprintf('data_mul_traj_n_%04d_am_%0.1e_%s.png',num_kept,mean(am),string(timestamp));
saveas(fig1, fullfile(result_dir,fig_name));

%% 
x = x(1:data_size,:);
x_dot = x_dot(1:data_size,:);
u = u(1:data_size,:);
t = t(1:data_size,:);
file_name = sprintf('data_mul_n_%04d_am_%0.1e_%s.m',num_kept, mean(am),string(timestamp));
save(fullfile(result_dir,file_name),'t','x','x_dot','u', ...
    'am','freq','th','m', 'L', 'b', 'g', 'inertia');