import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from torchinfo import summary
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

from cores.neural_network.models import LyapunovNetwork

np.random.seed(0)
torch.manual_seed(0)

in_features = 3
out_features = 2
n_layers = 7
activations = ['tanh'] * n_layers
widths = [in_features] + [10]*n_layers + [out_features]
zero_at_zero = True

input_bias = np.array([0.0, 0.2, 0.3])
input_transform = np.array([1.0, .5, 0.7])
model = LyapunovNetwork(in_features, 
                         out_features, 
                         activations, 
                         widths, 
                         zero_at_zero, 
                         input_bias = input_bias, 
                         input_transform = input_transform)
print('==> Evaluating model...')
summary(model, input_size=(1, in_features))

x = torch.rand((3, in_features))
model.eval()
print(model(x))

out, jac = model.forward_with_jacobian(x)
print(out)
print(jac)

jac_torch_raw = torch.autograd.functional.jacobian(model, x)
jac_torch = torch.zeros((x.shape[0], 1, in_features))
for i in range(x.shape[0]):
    for j in range(1):
        jac_torch[i, j] = jac_torch_raw[i][j][i]
print(jac_torch)

# # Calculate the Jacobian using finite differences
# eps = 1e-3
# jac_fd = torch.zeros((x.shape[0], out_features, in_features))
# for i in range(x.shape[0]):
#     for j in range(out_features):
#         for k in range(in_features):
#             x_plus = x.clone()
#             x_plus[i, k] += eps
#             out_plus = model(x_plus)
#             x_minus = x.clone()
#             x_minus[i, k] -= eps
#             out_minus = model(x_minus)
#             jac_fd[i, j, k] = (out_plus[i, j] - out_minus[i, j]) / (2 * eps)
# print(jac_fd)