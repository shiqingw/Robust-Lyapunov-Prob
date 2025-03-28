import time
import torch
from .utils import format_time, get_grad_l2_norm
import numpy as np
from .dyn_dataset import DynDataset
from ..cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def train_nn_mse_loss_no_test(model, optimizer, scheduler, num_epochs, train_dataloader, best_loc, device, threshold=1e-6):
    loss_monitor = np.zeros(num_epochs, dtype=np.float32)
    grad_norm_monitor = np.zeros(num_epochs, dtype=np.float32)
    model.to(device)
    best_epoch_loss = float('inf')
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_grad_norm = 0
        epoch_start_time = time.time()
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, y)
            loss.backward()
            grad_norm = get_grad_l2_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_grad_norm += grad_norm
                epoch_loss += loss.detach().cpu().numpy()
        epoch_end_time = time.time()
        epoch_loss = epoch_loss/(batch_idx+1)
        epoch_grad_norm = epoch_grad_norm/(batch_idx+1)
        if epoch % 5 == 0:
            print("Epoch: {:03d} | Loss: {:.4E} | Grad Norm: {:.4E} | Time: {}".format(
                epoch+1,
                epoch_loss,
                epoch_grad_norm,
                format_time(epoch_end_time - epoch_start_time)))
        loss_monitor[epoch] = epoch_loss
        grad_norm_monitor[epoch] = epoch_grad_norm
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            torch.save(model.state_dict(), best_loc)
            print("> Save model at epoch {:03d} with loss {:.4E}".format(epoch+1, best_epoch_loss))
        scheduler.step()
        if best_epoch_loss < threshold:
            print("> Threshold reached. Stop training.")
            break
    
    end_time = time.time()
    print("Total time: {}".format(format_time(end_time - start_time)))
    
    return loss_monitor, grad_norm_monitor


def train_dynamics(data_path, drift_nn, actuation_nn, estimated_system, dyn_selected_idx, batchsize, num_epoch, warmup_steps, drift_lr, drift_wd, actuation_lr, actuation_wd, 
                   drift_nn_best_loss_loc, actuation_nn_best_loss_loc, pt_dtype, device):

    optimizer = torch.optim.Adam([
                    {'params': drift_nn.parameters(), 'lr': drift_lr, 'weight_decay': drift_wd},
                    {'params': actuation_nn.parameters(), 'lr': actuation_lr, 'weight_decay': actuation_wd}
                ])

    dataset = DynDataset(data_path, pt_dtype)
    print("Dataset size: {}".format(len(dataset)))
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, 
                                              max_lr=[drift_lr, actuation_lr], 
                                              min_lr=[0.0, 0.0], 
                                              first_cycle_steps=num_epoch, 
                                              warmup_steps=warmup_steps)
    criterion = torch.nn.MSELoss()
    train_loss_monitor = []
    drift_nn_grad_norm_monitor = []
    actuation_nn_grad_norm_monitor = [] 

    drift_nn.train()
    actuation_nn.train()

    best_epoch_loss = float('inf')

    start_time = time.time()
    for epoch in range(num_epoch):
        epoch_train_loss = 0
        epoch_drift_nn_grad_norm = 0
        epoch_actuation_nn_grad_norm = 0
        epoch_train_start_time = time.time()
        for batch_idx, (_, x, u, x_dot) in enumerate(trainloader):
            x = x.to(device)
            u = u.to(device)
            x_dot = x_dot.to(device)
            optimizer.zero_grad()
            output_nn = estimated_system(x, u)
            loss = criterion(x_dot[:, dyn_selected_idx], output_nn[:, dyn_selected_idx])
            loss.backward()
            drift_nn_grad_norm = get_grad_l2_norm(drift_nn)
            actuation_nn_grad_norm = get_grad_l2_norm(actuation_nn)
            torch.nn.utils.clip_grad_norm_(drift_nn.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(actuation_nn.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_train_loss += loss.detach().cpu().numpy()
                epoch_drift_nn_grad_norm += drift_nn_grad_norm
                epoch_actuation_nn_grad_norm += actuation_nn_grad_norm
        epoch_train_end_time = time.time()
        scheduler.step()
        epoch_train_loss = epoch_train_loss/(batch_idx+1)
        epoch_drift_nn_grad_norm = epoch_drift_nn_grad_norm/(batch_idx+1)
        epoch_actuation_nn_grad_norm = epoch_actuation_nn_grad_norm/(batch_idx+1)
        if epoch % 5 == 0:
            print("Epoch: {:03d} | Train Loss: {:.4E} | Drift GN: {:.4E} | Actu. GN: {:.4E} | Time: {}".format(
                epoch+1,
                epoch_train_loss, 
                epoch_drift_nn_grad_norm, 
                epoch_actuation_nn_grad_norm,
                format_time(epoch_train_end_time - epoch_train_start_time)))
        train_loss_monitor.append(epoch_train_loss)
        drift_nn_grad_norm_monitor.append(epoch_drift_nn_grad_norm)
        actuation_nn_grad_norm_monitor.append(epoch_actuation_nn_grad_norm)
        if epoch_train_loss < best_epoch_loss:
            best_epoch_loss = epoch_train_loss
            torch.save(drift_nn.state_dict(), drift_nn_best_loss_loc)
            torch.save(actuation_nn.state_dict(), actuation_nn_best_loss_loc)
            print("> Save model at epoch {:03d} with loss {:.4E}".format(epoch+1, best_epoch_loss))

    end_time = time.time()
    print("Total time: {}".format(format_time(end_time - start_time)))
    return train_loss_monitor, drift_nn_grad_norm_monitor, actuation_nn_grad_norm_monitor

def train_nn_mse_loss_no_test(model, optimizer, scheduler, num_epochs, train_dataloader, best_loc, device, threshold=1e-6):
    loss_monitor = np.zeros(num_epochs, dtype=np.float32)
    grad_norm_monitor = np.zeros(num_epochs, dtype=np.float32)
    model.to(device)
    best_epoch_loss = float('inf')
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_grad_norm = 0
        epoch_start_time = time.time()
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = torch.nn.functional.mse_loss(out, y)
            loss.backward()
            grad_norm = get_grad_l2_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_grad_norm += grad_norm
                epoch_loss += loss.detach().cpu().numpy()
        epoch_end_time = time.time()
        epoch_loss = epoch_loss/(batch_idx+1)
        epoch_grad_norm = epoch_grad_norm/(batch_idx+1)
        if epoch % 5 == 0:
            print("Epoch: {:03d} | Loss: {:.4E} | Grad Norm: {:.4E} | Time: {}".format(
                epoch+1,
                epoch_loss,
                epoch_grad_norm,
                format_time(epoch_end_time - epoch_start_time)))
        loss_monitor[epoch] = epoch_loss
        grad_norm_monitor[epoch] = epoch_grad_norm
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            torch.save(model.state_dict(), best_loc)
            print("> Save model at epoch {:03d} with loss {:.4E}".format(epoch+1, best_epoch_loss))
        scheduler.step()
        if best_epoch_loss < threshold:
            print("> Threshold reached. Stop training.")
            break
    
    end_time = time.time()
    print("Total time: {}".format(format_time(end_time - start_time)))
    
    return loss_monitor, grad_norm_monitor

def pretrain_lyapunov_nn(lyapunov_nn, optimizer, scheduler, num_epochs, train_dataloader, boundary_state_torch, positive_weight, positive_margin, 
                         boundary_weight, boundary_margin, forward_inv_weight, lip_lyapunov_nn, stability_cutoff_radius,
                         best_loc, device, threshold=1e-6):
    loss_monitor = np.zeros(num_epochs, dtype=np.float32)
    grad_norm_monitor = np.zeros(num_epochs, dtype=np.float32)
    lyapunov_nn.to(device)
    best_epoch_loss = float('inf')
    start_time = time.time()
    for epoch in range(num_epochs):
        lyapunov_nn.train()
        epoch_loss = 0
        epoch_grad_norm = 0
        epoch_start_time = time.time()
        for batch_idx, (x,) in enumerate(train_dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            V = lyapunov_nn(x)

            loss_positive = positive_weight * torch.max(-V + positive_margin, torch.zeros_like(V)).mean()

            V_boundary = lyapunov_nn(boundary_state_torch.to(device))
            loss_forward_inv = forward_inv_weight * torch.std(V_boundary)

            loss_boundary = boundary_weight * torch.max(-V_boundary + lip_lyapunov_nn*stability_cutoff_radius + boundary_margin, torch.zeros_like(V_boundary)).mean()

            loss = loss_positive + loss_forward_inv + loss_boundary

            loss.backward()
            grad_norm = get_grad_l2_norm(lyapunov_nn)
            torch.nn.utils.clip_grad_norm_(lyapunov_nn.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                epoch_grad_norm += grad_norm
                epoch_loss += loss.detach().cpu().numpy()
        epoch_end_time = time.time()
        epoch_loss = epoch_loss/(batch_idx+1)
        epoch_grad_norm = epoch_grad_norm/(batch_idx+1)
        if epoch % 5 == 0:
            print("Epoch: {:03d} | Loss: {:.4E} | Grad Norm: {:.4E} | Time: {}".format(
                epoch+1,
                epoch_loss,
                epoch_grad_norm,
                format_time(epoch_end_time - epoch_start_time)))
        loss_monitor[epoch] = epoch_loss
        grad_norm_monitor[epoch] = epoch_grad_norm
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            torch.save(lyapunov_nn.state_dict(), best_loc)
            print("> Save model at epoch {:03d} with loss {:.4E}".format(epoch+1, best_epoch_loss))
        scheduler.step()
        if best_epoch_loss < threshold:
            print("> Threshold reached. Stop training.")
            break
    
    end_time = time.time()
    print("Total time: {}".format(format_time(end_time - start_time)))
    
    return loss_monitor, grad_norm_monitor