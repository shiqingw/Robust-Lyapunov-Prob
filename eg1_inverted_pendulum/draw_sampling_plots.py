import json
import sys
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import argparse
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from cores.dynamical_systems.create_system import get_system
from cores.neural_network.models import LyapunovNetwork, ControllerNetwork
from cores.utils.utils import seed_everything, save_dict, format_time
from cores.utils.config import Configuration

def get_lyapunov_ours(lyapunov_nn, state_flatten_np):
    config = Configuration()
    device = torch.device("cpu")

    # Evaluate Lyapunov network
    lyapunov_nn.eval()
    state_torch = torch.tensor(state_flatten_np, dtype=config.pt_dtype)
    dataset = torch.utils.data.TensorDataset(state_torch)
    batch_size = 512
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    V_torch = torch.zeros((state_torch.shape[0], 1), dtype=config.pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        V_batch = lyapunov_nn(state_batch).detach().cpu()
        V_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, V_torch.shape[0])] = V_batch
    V_flatten_np = V_torch.detach().cpu().numpy()

    return V_flatten_np

def get_stability_ours(model, state_flatten_np):
    config = Configuration()
    device = torch.device("cpu")

    # Evaluate Lyapunov network
    state_torch = torch.tensor(state_flatten_np, dtype=config.pt_dtype)
    dataset = torch.utils.data.TensorDataset(state_torch)
    batch_size = 512
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    stability_torch = torch.zeros((state_torch.shape[0], 1), dtype=config.pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        stability_batch = model(state_batch).detach().cpu()
        stability_torch[batch_idx * batch_size:min((batch_idx + 1) * batch_size, stability_torch.shape[0])] = stability_batch
    stability_flatten_np = stability_torch.detach().cpu().numpy()

    return stability_flatten_np

def draw_on_nominal_system(exp_num):
    print("==> Exp_num:", exp_num)
    # Create result directory
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), 0)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Exp directory
    print("==> Creating result directory ...")
    exp_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(exp_dir):
        exp_dir = "{}/eg1_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)
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
    system.load_state_dict(torch.load(f"{exp_dir}/system_params.pt", weights_only=True, map_location=device))
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
    controller_nn = ControllerNetwork(in_features=controller_in_features, 
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

    # Stability config
    stability_config = test_settings["stability_config"]
    stability_cutoff_radius = stability_config["cutoff_radius"]
    gamma = stability_config["gamma"]
    disturbance_channel = torch.tensor(stability_config["disturbance_channel"], dtype=config.pt_dtype)
    d0 = stability_config["d0"]
    d1 = stability_config["d1"]
    d2 = stability_config["d2"]

    def stability_fun(x):
        u = controller_nn(x)
        dx = system(x, u)
        V, V_dx = lyapunov_nn.forward_with_jacobian(x)
        norm_x = torch.linalg.norm(x, ord=2, dim=1).unsqueeze(-1) # (batch_size,1)
        dV = torch.bmm(V_dx, dx.unsqueeze(-1)).squeeze(-1) # (batch_size,1)
        V_dx_G = torch.matmul(V_dx, disturbance_channel.to(device)).squeeze(-2) # (batch_size,disturbance_dim)
        norm_V_dx_G = torch.linalg.norm(V_dx_G, ord=2, dim=1).unsqueeze(-1) # (batch_size,1)
        stability_condition = dV + norm_V_dx_G * (d0 + d1*norm_x + d2*norm_x**2) + gamma*V
        return stability_condition

    # Forward invariant region
    mesh_size = [100,100]
    state_dim = 2
    meshgrid = np.meshgrid(*[np.linspace(state_lower_bound[i], state_upper_bound[i], mesh_size[i]) for i in range(state_dim)])
    state_flatten_np = np.concatenate([meshgrid[i].reshape(-1, 1) for i in range(state_dim)], axis=1)

    # Boundary mask
    lower_bound_mask = np.isclose(state_flatten_np, state_lower_bound)
    upper_bound_mask = np.isclose(state_flatten_np, state_upper_bound)
    boundary_mask = np.any(lower_bound_mask | upper_bound_mask, axis=1)

    # Get the Lyapunov values (ours)
    V_ours_flatten_np = get_lyapunov_ours(lyapunov_nn, state_flatten_np)
    V_ours_min_np = np.min(V_ours_flatten_np[boundary_mask])
    print("==> Ours:")
    print("> Forward invariant set Lyapunov value: ", V_ours_min_np)
    print("> Forward invariant set percentage: ", np.sum(V_ours_flatten_np <= V_ours_min_np)/len(V_ours_flatten_np))

    # Matplotlib settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})
    fontsize = 50
    ticksize = 50
    level_fontsize = 35
    legend_fontsize = 40

    # Plot the level sets
    pairwise_idx = [(0,1)]
    state_labels = [r"$x_1$", r"$x_2$"]
    state_names = ["theta", "dtheta"]
    mesh_size = 100
    for (x_state_idx, y_state_idx) in pairwise_idx:
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111)

        x_np = np.linspace(state_lower_bound[x_state_idx], state_upper_bound[x_state_idx], mesh_size)
        y_np = np.linspace(state_lower_bound[y_state_idx], state_upper_bound[y_state_idx], mesh_size)
        X_np, Y_np = np.meshgrid(x_np, y_np)
        X_flatten_np = X_np.reshape(-1, 1)
        Y_flatten_np = Y_np.reshape(-1, 1)
        state_flatten_np = np.zeros((X_flatten_np.shape[0], len(state_lower_bound)), dtype=config.np_dtype)
        state_flatten_np[:, x_state_idx] = X_flatten_np[:, 0]
        state_flatten_np[:, y_state_idx] = Y_flatten_np[:, 0]

        # Ours
        V_ours_flatten_np = get_lyapunov_ours(lyapunov_nn, state_flatten_np)
        V_ours_np = V_ours_flatten_np.reshape(mesh_size, mesh_size)

        # Sample stability condition
        num_samples = 200000
        sample_states_np = np.zeros((num_samples, state_dim), dtype= config.np_dtype)
        sample_states_np[:, x_state_idx] = np.random.uniform(state_lower_bound[x_state_idx], state_upper_bound[x_state_idx], num_samples)
        sample_states_np[:, y_state_idx] = np.random.uniform(state_lower_bound[y_state_idx], state_upper_bound[y_state_idx], num_samples)

        stability_flatten_np = get_stability_ours(stability_fun, sample_states_np)
        bad_indices = np.where(stability_flatten_np > 0)[0]


        # Ours contour fill and lines
        cf = ax.contourf(X_np, Y_np, V_ours_np, levels=100, cmap='viridis')
        CS_ours_all = ax.contour(X_np, Y_np, V_ours_np, 
                                levels=np.linspace(0, np.max(V_ours_np), 10),
                                colors='k', linewidths=2)
        ax.clabel(CS_ours_all, inline=True, fontsize=level_fontsize)

        CS_ours_min = ax.contour(X_np, Y_np, V_ours_np, levels=[V_ours_min_np], 
                                colors='k', linewidths=6)
        # ax.clabel(CS_ours_min, inline=True, fontsize=level_fontsize, fmt=lambda _: "Ours")

        # Stability samples
        ax.scatter(sample_states_np[bad_indices,0], sample_states_np[bad_indices,1], s=8, color='red')
        uncolored_circle = plt.Circle( (0.0, 0.0 ), stability_cutoff_radius, fill=False, linestyle='--', linewidth=5, color='black')
        ax.add_patch(uncolored_circle)

        # Set labels and formatting
        ax.set_xlabel(state_labels[x_state_idx], fontsize=fontsize)
        ax.set_ylabel(state_labels[y_state_idx], fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=ticksize)

        # Create proxy artists for legend
        proxy_cutoff = mlines.Line2D([], [], linestyle='--', linewidth=5, color='black', label='Cutoff radius')
        proxy_violation = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, label='Violation')

        # Add legend with the proxy artists
        ax.legend(handles=[proxy_cutoff, proxy_violation], loc='upper right', fontsize=legend_fontsize)

        plt.tight_layout()
        plt.savefig(f"{results_dir}/sample_on_nom_{state_names[x_state_idx]}_{state_names[y_state_idx]}_{exp_num:03d}.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    exp_nums = list(range(1, 2))
    for exp_num in exp_nums:
        draw_on_nominal_system(exp_num)
        print("###########################################")