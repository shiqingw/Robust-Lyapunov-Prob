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
from cores.cosine_annealing_warmup import CosineAnnealingWarmupRestarts

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

    # Nominal results directory
    nominal_results_exp_num = test_settings["nominal_results_exp_num"]
    nominal_results_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), nominal_results_exp_num)

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
    data_path = f"{nominal_results_dir}/{data_file_name}"
    drift_nn_best_loss_loc = f"{results_dir}/drift_weights_best.pt"
    actuation_nn_best_loss_loc = f"{results_dir}/actuation_weights_best.pt"
    dyn_selected_idx = [1]
    dyn_train_loss_monitor, dyn_drift_nn_grad_norm_monitor, dyn_actuation_nn_grad_norm_monitor = train_dynamics(data_path, drift_nn, actuation_nn, estimated_system, 
        dyn_selected_idx, train_dyn_config['batch_size'], train_dyn_config['num_epochs'], train_dyn_config['warmup_steps'], 
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
    lyapunov_nn.load_state_dict(torch.load(f"{nominal_results_dir}/lyapunov_weights_best_loss.pt", weights_only=True, map_location=device))
    save_nn_weights(lyapunov_nn, f"{results_dir}/lyapunov_weights_init.pt")
    summary(lyapunov_nn, input_size=(1,lyapunov_in_features), dtypes=[config.pt_dtype])
    lyapunov_nn.to(device)

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
    controller_nn.load_state_dict(torch.load(f"{nominal_results_dir}/controller_weights_best_loss.pt", weights_only=True, map_location=device))
    save_nn_weights(controller_nn, f"{results_dir}/controller_weights_init.pt")
    summary(controller_nn, input_size=(1,controller_in_features), dtypes=[config.pt_dtype])
    controller_nn.to(device)

    # The dataset on the state space
    mesh_size = dataset_config["mesh_size"]
    state_dim = nominal_system.state_dim
    meshgrid = np.meshgrid(*[np.linspace(state_lower_bound[i], state_upper_bound[i], mesh_size[i]) for i in range(state_dim)])
    state_np = np.concatenate([meshgrid[i].reshape(-1, 1) for i in range(state_dim)], axis=1)
    del meshgrid
    print("==> Amount of training data: ", state_np.shape[0])

    # Stability config
    stability_config = test_settings["stability_config"]
    stability_cutoff_radius = stability_config["cutoff_radius"]
    gamma = stability_config["gamma"]
    disturbance_channel = torch.tensor(stability_config["disturbance_channel"], dtype=config.pt_dtype)
    d0 = stability_config["d0"]
    d1 = stability_config["d1"]
    d2 = stability_config["d2"]

    # Create training and test data loader
    print("==> Creating training data ...")
    train_config = test_settings["train_config"]
    train_state = torch.tensor(state_np, dtype=config.pt_dtype)
    train_dataset = torch.utils.data.TensorDataset(train_state)
    batch_size = train_config["batch_size"]
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = train_dataloader
    del state_np

    # Define optimizer, learning rate scheduler, loss function, and loss monitor
    num_epochs = train_config["num_epochs"]
    optimizer = torch.optim.Adam([
        {'params': lyapunov_nn.parameters(), 'lr': train_config['lyapunov_lr'], 'weight_decay': train_config['lyapunov_wd']},
        {'params': controller_nn.parameters(), 'lr': train_config['controller_lr'], 'weight_decay': train_config['controller_wd']}])
    scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, 
                                              max_lr=[train_config['lyapunov_lr'], train_config['controller_lr']], 
                                              min_lr=[0.0, 0.0], 
                                              first_cycle_steps=num_epochs, 
                                              warmup_steps=train_config["warmup_steps"])
    
    # Start training
    print("==> Start training ...")
    stability_weight = train_config["stability_weight"]
    stability_margin = train_config["stability_margin"]
    train_loss_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    test_loss_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    train_lyapunov_grad_norm_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    train_controller_grad_norm_monitor = np.zeros(num_epochs, dtype=config.np_dtype)
    best_epoch_test_loss = float('inf')
    lyapunov_best_loss_loc = f"{results_dir}/lyapunov_weights_best_loss.pt"
    controller_best_loss_loc = f"{results_dir}/controller_weights_best_loss.pt"
    best_forward_invariant_percentage = 0.0
    best_epoch = None
    lyapunov_nn.to(device)
    controller_nn.to(device)
    drift_nn.eval()
    actuation_nn.eval()
    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        lyapunov_nn.train()
        controller_nn.train()

        epoch_train_loss = 0
        epoch_stability_loss = 0
        epoch_train_lyapunov_grad_norm = 0
        epoch_train_controller_grad_norm = 0
        epoch_train_start_time = time.time()
        for batch_idx, (x,) in enumerate(train_dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            V, V_dx = lyapunov_nn.forward_with_jacobian(x) # (batch_size,1), (batch_size,state_dim)

            # Loss for stability
            u = controller_nn(x)
            dx = estimated_system(x, u)
            norm_x = torch.linalg.norm(x, ord=2, dim=1).unsqueeze(-1) # (batch_size,1)
            dV = torch.bmm(V_dx, dx.unsqueeze(-1)).squeeze(-1) # (batch_size,1)
            V_dx_G = torch.matmul(V_dx, disturbance_channel.to(device)).squeeze(-2) # (batch_size,disturbance_dim)
            norm_V_dx_G = torch.linalg.norm(V_dx_G, ord=2, dim=1).unsqueeze(-1) # (batch_size,1)
            stability_condition = dV + norm_V_dx_G * (d0 + d1*norm_x + d2*norm_x**2) + gamma*V
            outside_ball_stability = torch.linalg.norm(x, ord=2, dim=1) >= stability_cutoff_radius
            stability_condition = stability_condition[outside_ball_stability]
            loss_stability = stability_weight * torch.max(stability_condition + stability_margin, torch.zeros_like(stability_condition)).mean()

            loss = loss_stability
            loss.backward()
            lyapunov_grad_norm = get_grad_l2_norm(lyapunov_nn)
            controller_grad_norm = get_grad_l2_norm(controller_nn)
            torch.nn.utils.clip_grad_norm_(lyapunov_nn.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(controller_nn.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_train_loss += loss.detach().cpu().numpy()
                epoch_stability_loss += loss_stability.detach().cpu().numpy()
                epoch_train_lyapunov_grad_norm += lyapunov_grad_norm
                epoch_train_controller_grad_norm += controller_grad_norm
        epoch_train_end_time = time.time()
        epoch_train_loss = epoch_train_loss/(batch_idx+1)
        epoch_stability_loss = epoch_stability_loss/(batch_idx+1)
        epoch_train_lyapunov_grad_norm = epoch_train_lyapunov_grad_norm/(batch_idx+1)
        epoch_train_controller_grad_norm = epoch_train_controller_grad_norm/(batch_idx+1)
        if epoch % 5 == 0:
            print("Epoch: {:03d} | Train Loss: {:.4E} | Lya GN: {:.4E} | Ctrl GN: {:.4E} | Time: {}".format(
                epoch+1,
                epoch_train_loss, 
                epoch_train_lyapunov_grad_norm, 
                epoch_train_controller_grad_norm,
                format_time(epoch_train_end_time - epoch_train_start_time)))
        train_loss_monitor[epoch] = epoch_train_loss
        train_lyapunov_grad_norm_monitor[epoch] = epoch_train_lyapunov_grad_norm
        train_controller_grad_norm_monitor[epoch] = epoch_train_controller_grad_norm

        # Test
        lyapunov_nn.eval()
        controller_nn.eval()
        epoch_test_loss = 0
        epoch_test_start_time = time.time()
        with torch.no_grad():
            for batch_idx, (x,) in enumerate(test_dataloader):
                x = x.to(device)
                V, V_dx = lyapunov_nn.forward_with_jacobian(x)

                # Loss for stability
                u = controller_nn(x)
                dx = estimated_system(x, u)
                norm_x = torch.linalg.norm(x, ord=2, dim=1).unsqueeze(-1) # (batch_size,1)
                dV = torch.bmm(V_dx, dx.unsqueeze(-1)).squeeze(-1) # (batch_size,1)
                V_dx_G = torch.matmul(V_dx, disturbance_channel.to(device)).squeeze(-2) # (batch_size,disturbance_dim)
                norm_V_dx_G = torch.linalg.norm(V_dx_G, ord=2, dim=1).unsqueeze(-1) # (batch_size,1)
                stability_condition = dV + norm_V_dx_G * (d0 + d1*norm_x + d2*norm_x**2) + gamma*V
                outside_ball_stability = torch.linalg.norm(x, ord=2, dim=1) >= stability_cutoff_radius
                stability_condition = stability_condition[outside_ball_stability]
                loss_stability = torch.max(stability_condition, torch.zeros_like(stability_condition)).mean()

                loss = loss_stability
                epoch_test_loss += loss.detach().cpu().numpy()
        epoch_test_end_time = time.time()
        epoch_test_loss = epoch_test_loss/(batch_idx+1)
        print("Epoch: {:03d} | Test Loss: {:.4E} | Time: {}".format(epoch+1,
                    epoch_test_loss, format_time(epoch_test_end_time - epoch_test_start_time)))
        test_loss_monitor[epoch] = epoch_test_loss

        # Save the model if the test loss is the best
        if epoch_test_loss <= best_epoch_test_loss:
            best_epoch_test_loss = epoch_test_loss
            torch.save(lyapunov_nn.state_dict(), lyapunov_best_loss_loc)
            torch.save(controller_nn.state_dict(), controller_best_loss_loc)
            print("> Save at epoch {:03d} | Test loss {:.4E}".format(epoch+1, best_epoch_test_loss))

        scheduler.step()
    
    end_time = time.time()
    print("Total time: {}".format(format_time(end_time - start_time)))

    del train_state, train_dataset, train_dataloader, test_dataloader, 
    
    # Visualize training loss
    print("==> Visualizing training and test losses ...")
    draw_curve(data=train_loss_monitor,
               ylabel="Train Loss", 
               savepath=f"{results_dir}/00_train_train_loss.png", 
               dpi=100)
    draw_curve(data=train_lyapunov_grad_norm_monitor,
               ylabel="Lyapunov Train Grad Norm",
               savepath=f"{results_dir}/00_train_lyapunov_grad_norm.png",
               dpi=100)
    draw_curve(data=train_controller_grad_norm_monitor,
               ylabel="Controller Train Grad Norm",
               savepath=f"{results_dir}/00_train_controller_grad_norm.png",
               dpi=100)
    draw_curve(data=test_loss_monitor,
               ylabel="Test Loss",
               savepath=f"{results_dir}/00_train_test_loss.png",
               dpi=100)
    
    # Load the best weights
    if best_epoch is None:
        best_epoch = -1
        lyapunov_nn.load_state_dict(torch.load(lyapunov_best_loss_loc, weights_only=True, map_location=device))
        controller_nn.load_state_dict(torch.load(controller_best_loss_loc, weights_only=True, map_location=device))
    lyapunov_nn.eval()
    controller_nn.eval()

    # Save the training results
    print("==> Saving the training results ...")
    train_results = {
        "time": end_time - start_time,
        "train_loss": train_loss_monitor,
        "test_loss": test_loss_monitor,
        "train_lyapunov_grad_norm": train_lyapunov_grad_norm_monitor,
        "train_controller_grad_norm": train_controller_grad_norm_monitor
    }
    save_dict(train_results, f"{results_dir}/train_results.pkl")

    # Check the Lyapunov function and stability condition
    print("==> Computing the Lyapunov function and stability condition ...")
    post_mesh_size = dataset_config["post_mesh_size"]
    meshgrid = np.meshgrid(*[np.linspace(state_lower_bound[i], state_upper_bound[i], post_mesh_size[i]) for i in range(state_dim)])
    state_flatten_np = np.concatenate([meshgrid[i].reshape(-1, 1) for i in range(state_dim)], axis=1)
    del meshgrid
    state_flatten_torch = torch.tensor(state_flatten_np, dtype=config.pt_dtype)
    dataset = torch.utils.data.TensorDataset(state_flatten_torch)
    batch_size = 512
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    V_flatten_torch = torch.zeros((state_flatten_np.shape[0], 1), dtype=config.pt_dtype)
    stability_flatten_torch = torch.zeros((state_flatten_np.shape[0], 1), dtype=config.pt_dtype)
    for (batch_idx, (state_batch,)) in enumerate(dataloader):
        state_batch = state_batch.to(device)
        V_batch, V_dx_batch = lyapunov_nn.forward_with_jacobian(state_batch)
        V_flatten_torch[batch_idx*batch_size:min((batch_idx+1)*batch_size, V_flatten_torch.shape[0])] = V_batch.detach().cpu()
        u_batch = controller_nn(state_batch)
        dx_batch = true_system(state_batch, u_batch)
        dV_batch = torch.bmm(V_dx_batch, dx_batch.unsqueeze(-1)).squeeze(-1) # (N,1)
        V_dx_G_batch = torch.matmul(V_dx_batch, disturbance_channel.to(device)).squeeze(-2) # (N,disturbance_dim)
        norm_V_dx_G_batch = torch.linalg.norm(V_dx_G_batch, ord=2, dim=1).unsqueeze(-1) # (N,1)
        norm_x_batch = torch.linalg.norm(state_batch, ord=2, dim=1).unsqueeze(-1) # (N,1)
        stability_batch = dV_batch + norm_V_dx_G_batch*(d0 + d1*norm_x_batch + d2*norm_x_batch**2) + gamma*V_batch
        stability_flatten_torch[batch_idx*batch_size:min((batch_idx+1)*batch_size, stability_flatten_torch.shape[0])] = stability_batch.detach().cpu()

    del state_flatten_torch, dataset, dataloader

    print("==> Checking the forward invariant set ...")
    V_flatten_np = V_flatten_torch.detach().cpu().numpy()
    lower_bound_mask = np.isclose(state_flatten_np, state_lower_bound)
    upper_bound_mask = np.isclose(state_flatten_np, state_upper_bound)
    boundary_mask = np.any(lower_bound_mask | upper_bound_mask, axis=1)
    V_min_np = np.min(V_flatten_np[boundary_mask])
    del lower_bound_mask, upper_bound_mask, boundary_mask

    # Determine success 
    print("==> Checking the Lyapunov function and stability condition ...")
    stability_success = True
    stability_flatten_np = stability_flatten_torch.detach().cpu().numpy()
    stability_bad_idx = stability_flatten_np.squeeze() > 0
    stability_outside_ball_idx = np.linalg.norm(state_flatten_np, ord=2, axis=1) >= stability_cutoff_radius
    stability_total_bad_idx = np.logical_and(stability_bad_idx, stability_outside_ball_idx)
    stability_bad_count = np.sum(stability_total_bad_idx)

    if stability_bad_count > 0:
        print("> The stability condition is not satisfied at some points outside the ball.")
        print("> The number of bad points: ", stability_bad_count)
        stability_success = False
        print(state_flatten_np[stability_total_bad_idx])
    print("> Success: ", stability_success)

    # Save the forward invariant set
    print("==> Saving the forward invariant set ...")
    print("> Forward invariant set Lyapunov value: ", V_min_np)
    print("> Forward invariant set percentage: ", np.sum(V_flatten_np <= V_min_np)/len(V_flatten_np))
    forward_invariant_results = {
        "best_epoch": best_epoch,
        "success": stability_success,
        "stability_success": stability_success,
        "lyapunov_value": V_min_np,
        "percentage": np.sum(V_flatten_np <= V_min_np)/len(V_flatten_np)
    }
    save_dict(forward_invariant_results, f"{results_dir}/forward_invariant_results.pkl")

    del state_flatten_np
    del V_flatten_torch
    del stability_flatten_torch, stability_flatten_np, stability_bad_idx, stability_outside_ball_idx, stability_total_bad_idx

    # Plots
    print("==> Visualizing the Lyapunov function and stability condition ...")
    pairwise_idx = [(0,1)]
    state_labels = [r"$x_1$", r"$x_2$"]
    state_names = ["theta", "dtheta"]
    for (x_idx, y_idx) in pairwise_idx:
        draw_positive_condition_contour(lyapunov_nn=lyapunov_nn,
                                        state_lower_bound=state_lower_bound, 
                                        state_upper_bound=state_upper_bound, 
                                        mesh_size=400,
                                        x_state_idx=x_idx,
                                        y_state_idx=y_idx,
                                        x_label=state_labels[x_idx],
                                        y_label=state_labels[y_idx],
                                        positive_cutoff_radius=None,
                                        stability_cutoff_radius=stability_cutoff_radius,
                                        particular_level=None,
                                        savepath=f"{results_dir}/00_cond_positive_contour_{state_names[x_idx]}_{state_names[y_idx]}.png", 
                                        dpi=100,
                                        device=device, 
                                        pt_dtype=config.pt_dtype)

    def stability_fun(x):
        u = controller_nn(x)
        dx = true_system(x, u)
        V, V_dx = lyapunov_nn.forward_with_jacobian(x)
        norm_x = torch.linalg.norm(x, ord=2, dim=1).unsqueeze(-1) # (batch_size,1)
        dV = torch.bmm(V_dx, dx.unsqueeze(-1)).squeeze(-1) # (batch_size,1)
        V_dx_G = torch.matmul(V_dx, disturbance_channel.to(device)).squeeze(-2) # (batch_size,disturbance_dim)
        norm_V_dx_G = torch.linalg.norm(V_dx_G, ord=2, dim=1).unsqueeze(-1) # (batch_size,1)
        stability_condition = dV + norm_V_dx_G * (d0 + d1*norm_x + d2*norm_x**2) + gamma*V
        return stability_condition
    
    pairwise_idx = [(0,1)]
    state_labels = [r"$x_1$", r"$x_2$"]
    state_names = ["theta", "dtheta"]
    for (x_idx, y_idx) in pairwise_idx:
        draw_stability_condition_contour(model=stability_fun,
                                        state_lower_bound=state_lower_bound, 
                                        state_upper_bound=state_upper_bound,
                                        mesh_size=400, 
                                        x_state_idx=x_idx,
                                        y_state_idx=y_idx,
                                        x_label=state_labels[x_idx],
                                        y_label=state_labels[y_idx],
                                        stability_cutoff_radius=stability_cutoff_radius,
                                        savepath=f"{results_dir}/00_cond_stability_contour_{state_names[x_idx]}_{state_names[y_idx]}.png", 
                                        dpi=100, 
                                        device=device, 
                                        pt_dtype=config.pt_dtype)

    print("==> Done!")