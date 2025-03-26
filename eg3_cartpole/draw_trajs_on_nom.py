import json
import sys
import os
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from cores.dynamical_systems.create_system import get_system
from cores.neural_network.models import LyapunovNetwork, FullyConnectedNetwork
from cores.utils.utils import seed_everything, load_dict
from cores.utils.config import Configuration
from scipy.linalg import solve_continuous_are

def draw_unperturbed(exp_num):
    print("==> Exp_num:", exp_num)
    # Create result directory
    results_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), 0)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Exp directory
    print("==> Creating result directory ...")
    exp_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(exp_dir):
        exp_dir = "{}/eg3_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings_{:03d}.json".format(exp_dir, exp_num)

    # Load test settings
    print("==> Loading test settings ...")
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Decide torch device
    print("==> Deciding torch device ...")
    config = Configuration()
    device = torch.device("cpu")
    print('==> torch device: ', device)

    # Seed everything
    print("==> Seeding everything ...")
    seed = test_settings["seed"]
    seed_everything(seed)

    # Build dynamical system
    print("==> Building dynamical system ...")
    system_name = test_settings["nominal_system_name"]
    system = get_system(system_name=system_name, 
                        dtype=config.pt_dtype).to(device)
    system.eval()
    
    # Build Lyapunov network
    print("==> Building Lyapunov neural network ...")
    lyapunov_config = test_settings["lyapunov_nn_config"]
    lyapunov_in_features = lyapunov_config["in_features"]
    lyapunov_out_features = lyapunov_config["out_features"]
    lyapunov_activations = [lyapunov_config["activations"]]*(lyapunov_config["num_layers"]-1)
    lyapunov_widths = [lyapunov_in_features]+[lyapunov_config["width_each_layer"]]*(lyapunov_config["num_layers"]-1)+[lyapunov_out_features]
    lyapunov_beta = lyapunov_config["beta"]
    lyapunov_zero_at_zero = bool(lyapunov_config["zero_at_zero"])
    lyapunov_bias = bool(lyapunov_config["bias"])
    lyapunov_input_bias = np.array(lyapunov_config["input_bias"], dtype=config.np_dtype)
    lyapunov_input_transform = 1.0/np.array(lyapunov_config["input_transform_to_inverse"], dtype=config.np_dtype)
    lyapunov_nn = LyapunovNetwork(in_features=lyapunov_in_features, 
                                    out_features=lyapunov_out_features,
                                    activations=lyapunov_activations,
                                    widths=lyapunov_widths,
                                    zero_at_zero=lyapunov_zero_at_zero,
                                    bias=lyapunov_bias,
                                    input_bias=lyapunov_input_bias,
                                    input_transform=lyapunov_input_transform,
                                    beta=lyapunov_beta,
                                    dtype=config.pt_dtype)
    lyapunov_weights_dir = f"{exp_dir}/lyapunov_weights_best_loss.pt"
    lyapunov_nn.load_state_dict(torch.load(lyapunov_weights_dir, weights_only=True, map_location=device))
    lyapunov_nn.to(device)
    lyapunov_nn.eval()

    # Build controller network
    print("==> Building controller neural network ...")
    controller_config = test_settings["controller_nn_config"]
    controller_in_features = controller_config["in_features"]
    controller_out_features = controller_config["out_features"]
    controller_activations = [controller_config["activations"]]*(controller_config["num_layers"]-1)
    controller_widths = [controller_in_features]+[controller_config["width_each_layer"]]*(controller_config["num_layers"]-1)+[controller_out_features]
    controller_zero_at_zero = bool(controller_config["zero_at_zero"])
    controller_bias = bool(controller_config["bias"])
    controller_input_bias = np.array(controller_config["input_bias"], dtype=config.np_dtype)
    controller_input_transform = 1.0/np.array(controller_config["input_transform_to_inverse"], dtype=config.np_dtype)
    controller_lower_bound = controller_config["lower_bound"]
    controller_upper_bound = controller_config["upper_bound"]
    controller_nn = FullyConnectedNetwork(in_features=controller_in_features, 
                                    out_features=controller_out_features,
                                    activations=controller_activations,
                                    widths=controller_widths,
                                    zero_at_zero=controller_zero_at_zero,
                                    bias=controller_bias,
                                    input_bias=controller_input_bias,
                                    input_transform=controller_input_transform,
                                    lower_bound=controller_lower_bound,
                                    upper_bound=controller_upper_bound,
                                    dtype=config.pt_dtype)
    controller_nn.load_state_dict(torch.load(f"{exp_dir}/controller_weights_best_loss.pt", weights_only=True, map_location=device))
    controller_nn.to(device)
    controller_nn.eval()

    # Sample initial states
    print("==> Sampling initial states ...")
    dataset_config = test_settings["dataset_config"]
    state_lower_bound = np.array(dataset_config["state_lower_bound"], dtype=config.np_dtype)
    state_upper_bound = np.array(dataset_config["state_upper_bound"], dtype=config.np_dtype)
    post_mesh_size = 30
    state_dim = system.state_dim
    meshgrid = np.meshgrid(*[np.linspace(state_lower_bound[i], state_upper_bound[i], post_mesh_size) for i in range(state_dim)])
    state_np = np.concatenate([meshgrid[i].reshape(-1, 1) for i in range(state_dim)], axis=1)
    del meshgrid

    state_torch = torch.tensor(state_np, dtype=config.pt_dtype, device=device)
    dataset = torch.utils.data.TensorDataset(state_torch)
    batch_size = 2**14
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    V_flatten_torch = torch.zeros((state_torch.shape[0], 1), dtype=config.pt_dtype)
    for (batch_idx, (XY_batch,)) in enumerate(dataloader):
        XY_batch = XY_batch.to(device)
        V_batch, _ = lyapunov_nn.forward_with_jacobian(XY_batch)
        V_flatten_torch[batch_idx*batch_size:min((batch_idx+1)*batch_size, V_flatten_torch.shape[0])] = V_batch.detach().cpu()
    
    V_flatten_np = V_flatten_torch.detach().cpu().numpy()

    lower_bound_mask = np.isclose(state_np, state_lower_bound)
    upper_bound_mask = np.isclose(state_np, state_upper_bound)
    boundary_mask = np.any(lower_bound_mask | upper_bound_mask, axis=1)
    V_min_np = np.min(V_flatten_np[boundary_mask])
    print("> V_min_np: ", V_min_np)

    # Find the states where V is equal to V_min
    print("==> Finding the states where V is equal to V_min ...")
    good_idx = np.where((V_flatten_np >= V_min_np) & (V_flatten_np <= V_min_np + 1e-4))[0]
    good_states_np = state_np[good_idx, :]
    print("> Number of good states: ", good_states_np.shape[0])
    num_trajs = min(200, good_states_np.shape[0])
    horizon = 1500
    dt = 0.01
    sample_idx = np.random.choice(good_states_np.shape[0], num_trajs, replace=False)
    initial_states_np = good_states_np[sample_idx, :]
    trajectories_np = np.zeros((num_trajs, horizon, system.state_dim), dtype=config.np_dtype)
    trajectories_np[:, 0, :] = initial_states_np
    controls_np = np.zeros((num_trajs, horizon, system.control_dim), dtype=config.np_dtype)
    for i in range(horizon-1):
        states_torch = torch.tensor(trajectories_np[:, i, :], dtype=config.pt_dtype, device=device)
        controls_torch = controller_nn(states_torch)
        dx_torch = system(states_torch, controls_torch)
        new_states_torch = states_torch + dx_torch*dt
        trajectories_np[:, i+1, :] = new_states_torch.cpu().detach().numpy()
        controls_np[:, i, :] = controls_torch.cpu().detach().numpy()
    
    # Plot trajectories
    print("==> Plotting trajectories ...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})
    fontsize = 50
    ticksize = 50

    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111)
    for i in range(num_trajs):
        ax.plot(np.arange(horizon)*dt, np.linalg.norm(trajectories_np[i, :, :], axis=1))
    ax.set_xlabel(r"Time [s]", fontsize=fontsize)
    ax.set_ylabel(r"$\lVert x \rVert_2$", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    ax.hlines(y=0, xmin=0, xmax=horizon*dt, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/unperturbed_on_nom_{exp_num:03d}_state_norm.pdf", dpi=100)
    plt.close()

    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111)
    for i in range(num_trajs):
        ax.plot(np.arange(horizon)*dt, controls_np[i, :, 0])
    ax.set_xlabel(r"Time [s]", fontsize=fontsize)
    ax.set_ylabel(r"$u$", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    ax.hlines(y=0, xmin=0, xmax=horizon*dt, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/unperturbed_on_nom_{exp_num:03d}_control.pdf", dpi=100)
    plt.close()

def draw_perturbed(exp_num):
    print("==> Exp_num:", exp_num)
    # Create result directory
    results_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), 0)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Exp directory
    print("==> Creating result directory ...")
    exp_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(exp_dir):
        exp_dir = "{}/eg3_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings_{:03d}.json".format(exp_dir, exp_num)

    # Load test settings
    print("==> Loading test settings ...")
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Decide torch device
    print("==> Deciding torch device ...")
    config = Configuration()
    device = torch.device("cpu")
    print('==> torch device: ', device)

    # Seed everything
    print("==> Seeding everything ...")
    seed = test_settings["seed"]
    seed_everything(seed)

    # Build dynamical system
    print("==> Building dynamical system ...")
    system_name = test_settings["nominal_system_name"]
    system = get_system(system_name=system_name, 
                        dtype=config.pt_dtype).to(device)
    system.eval()
    
    # Build Lyapunov network
    print("==> Building Lyapunov neural network ...")
    lyapunov_config = test_settings["lyapunov_nn_config"]
    lyapunov_in_features = lyapunov_config["in_features"]
    lyapunov_out_features = lyapunov_config["out_features"]
    lyapunov_activations = [lyapunov_config["activations"]]*(lyapunov_config["num_layers"]-1)
    lyapunov_widths = [lyapunov_in_features]+[lyapunov_config["width_each_layer"]]*(lyapunov_config["num_layers"]-1)+[lyapunov_out_features]
    lyapunov_beta = lyapunov_config["beta"]
    lyapunov_zero_at_zero = bool(lyapunov_config["zero_at_zero"])
    lyapunov_bias = bool(lyapunov_config["bias"])
    lyapunov_input_bias = np.array(lyapunov_config["input_bias"], dtype=config.np_dtype)
    lyapunov_input_transform = 1.0/np.array(lyapunov_config["input_transform_to_inverse"], dtype=config.np_dtype)
    lyapunov_nn = LyapunovNetwork(in_features=lyapunov_in_features, 
                                    out_features=lyapunov_out_features,
                                    activations=lyapunov_activations,
                                    widths=lyapunov_widths,
                                    zero_at_zero=lyapunov_zero_at_zero,
                                    bias=lyapunov_bias,
                                    input_bias=lyapunov_input_bias,
                                    input_transform=lyapunov_input_transform,
                                    beta=lyapunov_beta,
                                    dtype=config.pt_dtype)
    lyapunov_weights_dir = f"{exp_dir}/lyapunov_weights_best_loss.pt"
    if not os.path.exists(lyapunov_weights_dir):
        print(f"==> Lyapunov weights not found, skipping {exp_num:02d}")
        return
    lyapunov_nn.load_state_dict(torch.load(lyapunov_weights_dir, weights_only=True, map_location=device))
    lyapunov_nn.to(device)
    lyapunov_nn.eval()

    # Build controller network
    print("==> Building controller neural network ...")
    controller_config = test_settings["controller_nn_config"]
    controller_in_features = controller_config["in_features"]
    controller_out_features = controller_config["out_features"]
    controller_activations = [controller_config["activations"]]*(controller_config["num_layers"]-1)
    controller_widths = [controller_in_features]+[controller_config["width_each_layer"]]*(controller_config["num_layers"]-1)+[controller_out_features]
    controller_zero_at_zero = bool(controller_config["zero_at_zero"])
    controller_bias = bool(controller_config["bias"])
    controller_input_bias = np.array(controller_config["input_bias"], dtype=config.np_dtype)
    controller_input_transform = 1.0/np.array(controller_config["input_transform_to_inverse"], dtype=config.np_dtype)
    controller_lower_bound = controller_config["lower_bound"]
    controller_upper_bound = controller_config["upper_bound"]
    controller_nn = FullyConnectedNetwork(in_features=controller_in_features, 
                                    out_features=controller_out_features,
                                    activations=controller_activations,
                                    widths=controller_widths,
                                    zero_at_zero=controller_zero_at_zero,
                                    bias=controller_bias,
                                    input_bias=controller_input_bias,
                                    input_transform=controller_input_transform,
                                    lower_bound=controller_lower_bound,
                                    upper_bound=controller_upper_bound,
                                    dtype=config.pt_dtype)
    controller_nn.load_state_dict(torch.load(f"{exp_dir}/controller_weights_best_loss.pt", weights_only=True, map_location=device))
    controller_nn.to(device)
    controller_nn.eval()

    # Stability config
    stability_config = test_settings["stability_config"]
    disturbance_channel = torch.tensor(stability_config["disturbance_channel"], dtype=config.pt_dtype)
    d0 = stability_config["d0"]
    d1 = stability_config["d1"]
    d2 = stability_config["d2"]
    disturbance_dim = disturbance_channel.shape[1]

    # Sample initial states
    print("==> Sampling initial states ...")
    dataset_config = test_settings["dataset_config"]
    state_lower_bound = np.array(dataset_config["state_lower_bound"], dtype=config.np_dtype)
    state_upper_bound = np.array(dataset_config["state_upper_bound"], dtype=config.np_dtype)
    post_mesh_size = 30
    state_dim = system.state_dim
    meshgrid = np.meshgrid(*[np.linspace(state_lower_bound[i], state_upper_bound[i], post_mesh_size) for i in range(state_dim)])
    state_np = np.concatenate([meshgrid[i].reshape(-1, 1) for i in range(state_dim)], axis=1)
    del meshgrid

    state_torch = torch.tensor(state_np, dtype=config.pt_dtype, device=device)
    dataset = torch.utils.data.TensorDataset(state_torch)
    batch_size = 2**14
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    V_flatten_torch = torch.zeros((state_torch.shape[0], 1), dtype=config.pt_dtype)
    for (batch_idx, (XY_batch,)) in enumerate(dataloader):
        XY_batch = XY_batch.to(device)
        V_batch, _ = lyapunov_nn.forward_with_jacobian(XY_batch)
        V_flatten_torch[batch_idx*batch_size:min((batch_idx+1)*batch_size, V_flatten_torch.shape[0])] = V_batch.detach().cpu()
    
    V_flatten_np = V_flatten_torch.detach().cpu().numpy()

    lower_bound_mask = np.isclose(state_np, state_lower_bound)
    upper_bound_mask = np.isclose(state_np, state_upper_bound)
    boundary_mask = np.any(lower_bound_mask | upper_bound_mask, axis=1)
    V_min_np = np.min(V_flatten_np[boundary_mask])

    # Find the states where V is equal to V_min
    print("==> Finding the states where V is equal to V_min ...")
    good_idx = np.where((V_flatten_np >= V_min_np) & (V_flatten_np <= V_min_np + 1e-4))[0]
    good_states_np = state_np[good_idx, :]
    print("> Number of good states: ", good_states_np.shape[0])
    num_trajs = min(200, good_states_np.shape[0])
    horizon = 1500
    dt = 0.01
    sample_idx = np.random.choice(good_states_np.shape[0], num_trajs, replace=False)
    initial_states_np = good_states_np[sample_idx, :]
    
    trajectories_np = np.zeros((num_trajs, horizon, system.state_dim), dtype=config.np_dtype)
    trajectories_np[:, 0, :] = initial_states_np
    controls_np = np.zeros((num_trajs, horizon, system.control_dim), dtype=config.np_dtype)
    for i in range(horizon-1):
        states_torch = torch.tensor(trajectories_np[:, i, :], dtype=config.pt_dtype, device=device)
        controls_torch = controller_nn(states_torch)
        dx_torch = system(states_torch, controls_torch) # (N, state_dim)
        
        disturbance = torch.zeros((dx_torch.shape[0], disturbance_dim), dtype=config.pt_dtype, device=device)
        disturbance[:,0] = d0*np.sin(2*np.pi*i*dt) + d1*states_torch[:,0] + d2*states_torch[:,2]**2  # d0*sin(2*pi*t) + d1*x + d2*v^2

        matched_disturbance = torch.matmul(disturbance, disturbance_channel.T.to(device))
        
        dx_torch += matched_disturbance

        new_states_torch = states_torch + dx_torch*dt
        trajectories_np[:, i+1, :] = new_states_torch.cpu().detach().numpy()
        controls_np[:, i, :] = controls_torch.cpu().detach().numpy()
    
    # Plot trajectories
    print("==> Plotting trajectories ...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})
    fontsize = 50
    ticksize = 50

    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111)
    for i in range(num_trajs):
        ax.plot(np.arange(horizon)*dt, np.linalg.norm(trajectories_np[i, :, :], axis=1))
    ax.set_xlabel(r"Time [s]", fontsize=fontsize)
    ax.set_ylabel(r"$\lVert x \rVert_2$", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    ax.hlines(y=0, xmin=0, xmax=horizon*dt, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/perturbed_on_nom_{exp_num:03d}_state_norm.pdf", dpi=100)
    plt.close()

    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111)
    for i in range(num_trajs):
        ax.plot(np.arange(horizon)*dt, controls_np[i, :, 0])
    ax.set_xlabel(r"Time [s]", fontsize=fontsize)
    ax.set_ylabel(r"$u$", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, grid_linewidth=10)
    ax.hlines(y=0, xmin=0, xmax=horizon*dt, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/perturbed_on_nom_{exp_num:03d}_control.pdf", dpi=100)
    plt.close()

if __name__ == "__main__":
    exp_nums = list(range(1, 2))
    for exp_num in exp_nums:
        draw_unperturbed(exp_num)
        print("###########################################")

        draw_perturbed(exp_num)
        print("###########################################")

