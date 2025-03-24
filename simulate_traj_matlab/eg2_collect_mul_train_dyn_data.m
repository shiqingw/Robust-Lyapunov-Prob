clear all;
rng('default');
filepath = fileparts(mfilename('fullpath'));
parts = strsplit(filepath, filesep);
parent_path = strjoin(parts(1:end-1), filesep);
result_dir = fullfile(parent_path,'eg2_results', '001');
controller_dir = result_dir;
modelfile = fullfile(controller_dir, 'controller.onnx');

%% 
global params a1 a2 b1 b2 c1 c2 am freq th;
params = importONNXFunction(modelfile, "controller");
a1 = 0.0;
a2 = 1.0;
b1 = 0.0;
b2 = 1.0;
c1 = 1.0;
c2 = 0.8;
num_sine = 10;
am = 0.2*ones(1,num_sine);
freq = zeros(num_sine,1);
freq(1:4) = [1;3;5;7];
freq(5:end) = rand(num_sine-4, 1)*10; 
th = (rand(num_sine, 1)-0.5)*pi;

%% 
state_space = [-1, 1;
               -1, 1;
               -1, 1];
state_dim = size(state_space,1);

points_per_dim = [8, 8, 8];
x1 = linspace(state_space(1,1), state_space(1,2), points_per_dim(1));
x2 = linspace(state_space(2,1), state_space(2,2), points_per_dim(2));
x3 = linspace(state_space(3,1), state_space(3,2), points_per_dim(3));
[X1,X2,X3] = ndgrid(x1,x2,x3);

initial_states = zeros(prod(points_per_dim),state_dim);
initial_states(:,1) = reshape(X1,[],1);
initial_states(:,2) = reshape(X2,[],1);
initial_states(:,3) = reshape(X3,[],1);

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
T = 4;
dt = 0.01;
n = T/dt;
cutoff_radius = 0.05;
threshold = 0.5;

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
    sol = ode15s(@eg2_backstepping_excitation,[0 T],initial_states(ii,:)');
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

    plot3(x_tmp(:,1), x_tmp(:,2), x_tmp(:,3),'LineWidth',2);
    hold on;
end

x0 = 10;
y0 = 10;
width = 550;
height = 550;
fig1.Position = [x0 y0 width height]; 
% xlim([-1 1]);
% ylim([-1 1]);
% zlim([-1 1]);
extraInputs = {'fontsize',30,'FontName','Serif','Interpreter','latex'};
ticksize = 30;
grid on;
ax = gca;
ax.LineWidth = 1.5;
xaxisproperties= get(gca, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex';
xaxisproperties.FontSize = ticksize;

yaxisproperties= get(gca, 'YAxis');
yaxisproperties.TickLabelInterpreter = 'latex';
yaxisproperties.FontSize = ticksize;

zaxisproperties= get(gca, 'ZAxis');
zaxisproperties.TickLabelInterpreter = 'latex';
zaxisproperties.FontSize = ticksize;
xlabel('$x_1$', extraInputs{:});
ylabel('$x_2$', extraInputs{:});
zlabel('$x_3$', extraInputs{:});
timestamp = datetime;
timestamp.Format = 'yyyy-MM-dd_HH-mm-ss';
fig_name = sprintf('data_mul_traj_n_%04d_am_%0.1e_%s.png',num_kept,mean(am),string(timestamp));
saveas(fig1, fullfile(result_dir,fig_name));
% 
%% 
x = x(1:data_size,:);
x_dot = x_dot(1:data_size,:);
u = u(1:data_size,:);
t = t(1:data_size,:);
file_name = sprintf('data_mul_n_%04d_am_%0.1e_%s.m',num_kept, mean(am),string(timestamp));
save(fullfile(result_dir,file_name),'t','x','x_dot','u','am','freq','th', ...
    'a1','a2','b1','b2','c1','c2');