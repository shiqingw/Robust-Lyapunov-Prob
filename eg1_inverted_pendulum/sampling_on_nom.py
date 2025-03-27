import json
import sys
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import argparse
from torch.utils.data import DataLoader
import time

from cores.dynamical_systems.create_system import get_system
from cores.neural_network.models import LyapunovNetwork, FullyConnectedNetwork
from cores.utils.utils import seed_everything, save_dict, format_time
from cores.utils.config import Configuration

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    parser.add_argument('--delta', default=1e-3, type=float, help='dReal precision')
    parser.add_argument('--epsilon', default=1e-3, type=float, help='dReal precision')
    args = parser.parse_args()

    # Create result directory
    print("==> Creating result directory ...")
    exp_num = args.exp_num
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)

    # Load test settings
    print("==> Loading test settings ...")
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Decide torch device
    print("==> Deciding torch device ...")
    config = Configuration()
    user_device = args.device
    if user_device != "None":
        device = torch.device(user_device)
    else:
        device = config.device
    cpu = torch.device("cpu")
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
    lyapunov_weights_dir = f"{results_dir}/lyapunov_weights_best_loss.pt"
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
    controller_nn.load_state_dict(torch.load(f"{results_dir}/controller_weights_best_loss.pt", weights_only=True, map_location=device))
    controller_nn.to(device)
    controller_nn.eval()

    # The state space
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

    # Sample from uniform distribution
    print("==> Start sampling ...")
    delta = args.delta
    epsilon = args.epsilon
    print(f"> Delta: {delta:.2E}")
    print(f"> Epsilon: {epsilon:.2E}")
    required_samples = int(np.log(2/delta)/(2 * epsilon**2))
    buffer_size = 2**15
    estimated_buffer_count = int(np.ceil(required_samples/buffer_size))
    buffer_count = 0
    bad_counter = 0
    good_counter = 0
    total_counter = 0
    print(f"> Required samples: {required_samples}")
    print(f"> Buffer size: {buffer_size}")
    start_time = time.time()

    while total_counter < required_samples:

        buffer_start_time = time.time()

        ranges = state_upper_bound - state_lower_bound
        samples = np.random.rand(buffer_size, state_lower_bound.shape[0])
        states_np = state_lower_bound + samples * ranges

        states_torch = torch.tensor(states_np, dtype=config.pt_dtype)
        dataset = torch.utils.data.TensorDataset(states_torch)
        batch_size = 512
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        stability_torch = torch.zeros((states_np.shape[0], 1), dtype=config.pt_dtype)

        for (batch_idx, (state_batch,)) in enumerate(dataloader):
            state_batch = state_batch.to(device)
            V_batch, V_dx_batch = lyapunov_nn.forward_with_jacobian(state_batch)
            u_batch = controller_nn(state_batch)
            dx_batch = system(state_batch, u_batch)
            dV_batch = torch.bmm(V_dx_batch, dx_batch.unsqueeze(-1)).squeeze(-1) # (N,1)
            V_dx_G_batch = torch.matmul(V_dx_batch, disturbance_channel.to(device)).squeeze(-2) # (N,disturbance_dim)
            norm_V_dx_G_batch = torch.linalg.norm(V_dx_G_batch, ord=2, dim=1).unsqueeze(-1) # (N,1)
            norm_x_batch = torch.linalg.norm(state_batch, ord=2, dim=1).unsqueeze(-1) # (N,1)
            stability_batch = dV_batch + norm_V_dx_G_batch*(d0 + d1*norm_x_batch + d2*norm_x_batch**2) + gamma*V_batch
            stability_torch[batch_idx*batch_size:min((batch_idx+1)*batch_size, stability_torch.shape[0])] = stability_batch.detach().cpu()

        del states_torch, dataset, dataloader

        decrease_condition_np = stability_torch.numpy()
        cutoff_indices = np.linalg.norm(states_np, ord=2, axis=1) >= stability_cutoff_radius
        decrease_condition_cutoff = decrease_condition_np[cutoff_indices]
        good_indices_cutoff = decrease_condition_cutoff <= 0
        bad_indices_cutoff = decrease_condition_cutoff > 0

        bad_counter = bad_counter + np.sum(bad_indices_cutoff)
        good_counter = good_counter + np.sum(good_indices_cutoff)
        total_counter = total_counter + np.sum(cutoff_indices)

        buffer_end_time = time.time()
    
        if (buffer_count+1)%5 == 0:
            print("> Buffer id: {:03d}/{:03d} | Total points: {:.2E}/{:.2E} | Good %: {:.2f} | Buffer time: {}".format(buffer_count+1, estimated_buffer_count,
                total_counter, required_samples, good_counter/total_counter, format_time(buffer_end_time-buffer_start_time)))
            
        buffer_count = buffer_count + 1

    end_time = time.time()
    print("==> Summary: ")
    print("> Required samples: ", required_samples)
    print("> Total points: ", total_counter)
    print(f"> Good points: {good_counter}. Percentage: {good_counter/total_counter:.4f}")
    print(f"> Bad points: {bad_counter}. Percentage: {bad_counter/total_counter:.4f}")
    print(f"> Time: {format_time(end_time-start_time)}. In seconds: {end_time-start_time}")

    checking_result = {
        "total_points": total_counter,
        "good_points": good_counter,
        "time": end_time-start_time,
        "delta": delta,
        "epsilon": epsilon
    }
    platform = config.platform
    save_dict(checking_result, f"{results_dir}/00_sampling_nom_on_nom_{platform}_{device}_delta_{delta:.1E}_epsilon_{epsilon:.1E}.pkl")
    print("==> Dictionary saved!")
    print("==> Done!")