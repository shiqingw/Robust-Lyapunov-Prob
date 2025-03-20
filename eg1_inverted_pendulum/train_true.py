import json
import sys
import os
import argparse
import shutil
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from torchinfo import summary
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

from cores.dynamical_systems.create_system import get_system
from cores.neural_network.models import LyapunovNetwork, FullyConnectedNetwork, ConstantNetwork
from cores.utils.utils import seed_everything, save_nn_weights, save_dict, get_grad_l2_norm, format_time
from cores.utils.config import Configuration
from cores.utils.train_utils import train_dynamics
from cores.utils.draw_utils import draw_curve
from cores.utils.draw_utils import draw_model_diff_l2_contour, draw_positive_condition_contour, draw_stability_condition_contour

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=2, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    # Create result directory
    print("==> Creating result directory ...")
    exp_num = args.exp_num
    results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    shutil.copy(test_settings_path, results_dir)

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
    nominal_system_name = test_settings["nominal_system_name"]
    nominal_system = get_system(system_name=nominal_system_name, 
                        dtype=config.pt_dtype)
    save_nn_weights(nominal_system, f"{results_dir}/nominal_system_params.pt")
    true_system_name = test_settings["true_system_name"]
    true_system = get_system(system_name=true_system_name, 
                        dtype=config.pt_dtype)
    save_nn_weights(true_system, f"{results_dir}/true_system_params.pt")

    # Build dynamics neural network
    print("==> Building dynamics neural network ...")
    drift_config = test_settings["drift_nn_config"]
    drift_in_features = drift_config["in_features"]
    drift_out_features = drift_config["out_features"]
    drift_activations = [drift_config["activations"]]*(drift_config["num_layers"]-1)
    drift_widths = [drift_in_features]+[drift_config["width_each_layer"]]*(drift_config["num_layers"]-1)+[drift_out_features]
    drift_zero_at_zero = bool(drift_config["zero_at_zero"])
    drift_bias = bool(drift_config["bias"])
    drift_input_bias = np.array(drift_config["input_bias"], dtype=config.np_dtype)
    drift_input_transform = 1.0/np.array(drift_config["input_transform_to_inverse"], dtype=config.np_dtype)
    drift_nn = FullyConnectedNetwork(in_features=drift_in_features, 
                                    out_features=drift_out_features,
                                    activations=drift_activations,
                                    widths=drift_widths,
                                    zero_at_zero=drift_zero_at_zero,
                                    bias=drift_bias,
                                    input_bias=drift_input_bias,
                                    input_transform=drift_input_transform,
                                    lower_bound=None,
                                    upper_bound=None,
                                    dtype=config.pt_dtype)
    summary(drift_nn, input_size=(1, drift_in_features), dtypes=[config.pt_dtype])
    save_nn_weights(drift_nn, f"{results_dir}/drift_weights_init.pt")
    drift_nn.to(device)

    actuation_config = test_settings["actuation_nn_config"]
    actuation_out_features = actuation_config["out_features"]
    actuation_nn = ConstantNetwork(out_features=actuation_out_features, dtype=config.pt_dtype)
    save_nn_weights(actuation_nn, f"{results_dir}/actuation_weights_init.pt")
    actuation_nn.to(device)

    # Train dynamics neural network
    print("==> Training dynamics neural network ...")
    
    def estimated_drift(x):
        known_part = nominal_system.get_drift(x) # (batch_size, state_dim)
        learned_part = drift_nn(x) # (batch_size, 1)
        known_part[:, 1] += learned_part.squeeze(1)
        return known_part # (batch_size, state_dim)
    
    def estimated_actuation(x):
        known_part = nominal_system.get_actuation(x) # (batch_size, state_dim, control_dim)
        learned_part = actuation_nn(x) # (batch_size, 1)
        known_part[:, 1, :] += learned_part
        return known_part
    
    def estimated_system(x, u):
        drift = estimated_drift(x) # (batch_size, state_dim)
        actuation = estimated_actuation(x) # (batch_size, state_dim, control_dim)
        dx = drift + torch.bmm(actuation, u.unsqueeze(2)).squeeze(2) # (batch_size, state_dim)
        return dx
    
    # The state space
    dataset_config = test_settings["dataset_config"]
    state_lower_bound = np.array(dataset_config["state_lower_bound"], dtype=config.np_dtype)
    state_upper_bound = np.array(dataset_config["state_upper_bound"], dtype=config.np_dtype)

    # Visualize dynamics error
    print("==> Plot drift errors (before training)")
    pairwise_idx = [(0,1)]
    state_labels = [r"$x_1$", r"$x_2$"]
    state_names = ["theta", "dtheta"]
    for (x_idx, y_idx) in pairwise_idx:
        diff_mean, diff_max = draw_model_diff_l2_contour(model1=true_system.get_drift,
                                        model2=estimated_drift,
                                        state_lower_bound=state_lower_bound, 
                                        state_upper_bound=state_upper_bound, 
                                        mesh_size=400,
                                        x_state_idx=x_idx,
                                        y_state_idx=y_idx,
                                        x_label=state_labels[x_idx],
                                        y_label=state_labels[y_idx],
                                        savepath=f"{results_dir}/00_drift_err_init_contour_{state_names[x_idx]}_{state_names[y_idx]}.png", 
                                        dpi=100,
                                        device=device, 
                                        pt_dtype=config.pt_dtype)
        print(f"> ({state_names[x_idx]}, {state_names[y_idx]}): mean = {diff_mean:.4f}, max = {diff_max:.4f}")
        
    print("==> Plot actuation errors (before training)")
    pairwise_idx = [(0,1)]
    state_labels = [r"$x_1$", r"$x_2$"]
    state_names = ["theta", "dtheta"]
    for (x_idx, y_idx) in pairwise_idx:
        diff_mean, diff_max = draw_model_diff_l2_contour(model1=true_system.get_actuation,
                                        model2=estimated_actuation,
                                        state_lower_bound=state_lower_bound, 
                                        state_upper_bound=state_upper_bound, 
                                        mesh_size=400,
                                        x_state_idx=x_idx,
                                        y_state_idx=y_idx,
                                        x_label=state_labels[x_idx],
                                        y_label=state_labels[y_idx],
                                        savepath=f"{results_dir}/00_actuation_err_init_contour_{state_names[x_idx]}_{state_names[y_idx]}.png", 
                                        dpi=100,
                                        device=device, 
                                        pt_dtype=config.pt_dtype)
        print(f"> ({state_names[x_idx]}, {state_names[y_idx]}): mean = {diff_mean:.4f}, max = {diff_max:.4f}")

    # train dynamics
    print("==> Train dynamics")
    train_dyn_config = test_settings["train_dyn_config"]
    data_file_name = train_dyn_config["data_file_name"]
    data_path = "{}/eg1_results/{}".format(str(Path(__file__).parent.parent), data_file_name)
    drift_nn_best_loss_loc = f"{results_dir}/drift_weights_best.pt"
    actuation_nn_best_loss_loc = f"{results_dir}/actuation_weights_best.pt"
    dyn_train_loss_monitor, dyn_drift_nn_grad_norm_monitor, dyn_actuation_nn_grad_norm_monitor = train_dynamics(data_path, drift_nn, actuation_nn, estimated_system, 
        train_dyn_config['batch_size'], train_dyn_config['num_epochs'], 
        train_dyn_config['drift_lr'], train_dyn_config['drift_wd'], 
        train_dyn_config['actuation_lr'], train_dyn_config['actuation_wd'],
        drift_nn_best_loss_loc, actuation_nn_best_loss_loc,
        config.pt_dtype, device)
    draw_curve(data=dyn_train_loss_monitor,
               ylabel="Train Loss", 
               savepath=f"{results_dir}/00_dyn_train_loss.png", 
               dpi=100)
    draw_curve(data=dyn_drift_nn_grad_norm_monitor,
                ylabel="Drift NN Grad Norm", 
                savepath=f"{results_dir}/00_dyn_drift_nn_grad_norm.png", 
                dpi=100)
    draw_curve(data=dyn_actuation_nn_grad_norm_monitor,
                ylabel="Actuation NN Grad Norm", 
                savepath=f"{results_dir}/00_dyn_actuation_nn_grad_norm.png", 
                dpi=100)
    
    # Visualize dynamics error
    drift_nn.load_state_dict(torch.load(drift_nn_best_loss_loc, weights_only=True, map_location=device))
    actuation_nn.load_state_dict(torch.load(actuation_nn_best_loss_loc, weights_only=True, map_location=device))
    drift_nn.eval()
    actuation_nn.eval()
    print("==> Plot drift errors (after training)")
    pairwise_idx = [(0,1)]
    state_labels = [r"$x_1$", r"$x_2$"]
    state_names = ["theta", "dtheta"]
    for (x_idx, y_idx) in pairwise_idx:
        diff_mean, diff_max = draw_model_diff_l2_contour(model1=true_system.get_drift,
                                        model2=estimated_drift,
                                        state_lower_bound=state_lower_bound, 
                                        state_upper_bound=state_upper_bound, 
                                        mesh_size=400,
                                        x_state_idx=x_idx,
                                        y_state_idx=y_idx,
                                        x_label=state_labels[x_idx],
                                        y_label=state_labels[y_idx],
                                        savepath=f"{results_dir}/00_drift_err_trained_contour_{state_names[x_idx]}_{state_names[y_idx]}.png", 
                                        dpi=100,
                                        device=device, 
                                        pt_dtype=config.pt_dtype)
        print(f"> ({state_names[x_idx]}, {state_names[y_idx]}): mean = {diff_mean:.4f}, max = {diff_max:.4f}")
        
    print("==> Plot actuation errors (after training)")
    pairwise_idx = [(0,1)]
    state_labels = [r"$x_1$", r"$x_2$"]
    state_names = ["theta", "dtheta"]
    for (x_idx, y_idx) in pairwise_idx:
        diff_mean, diff_max = draw_model_diff_l2_contour(model1=true_system.get_actuation,
                                        model2=estimated_actuation,
                                        state_lower_bound=state_lower_bound, 
                                        state_upper_bound=state_upper_bound, 
                                        mesh_size=400,
                                        x_state_idx=x_idx,
                                        y_state_idx=y_idx,
                                        x_label=state_labels[x_idx],
                                        y_label=state_labels[y_idx],
                                        savepath=f"{results_dir}/00_actuation_err_trained_contour_{state_names[x_idx]}_{state_names[y_idx]}.png", 
                                        dpi=100,
                                        device=device, 
                                        pt_dtype=config.pt_dtype)
        print(f"> ({state_names[x_idx]}, {state_names[y_idx]}): mean = {diff_mean:.4f}, max = {diff_max:.4f}")