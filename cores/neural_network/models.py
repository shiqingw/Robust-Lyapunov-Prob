import torch 
import torch.nn as nn 
from .layers import LinearLayer

def get_activation(activation_name):
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'tanh':
        return nn.Tanh()
    elif activation_name == 'softplus':
        return nn.Softplus()
    elif activation_name == 'identity':
        return nn.Identity()
    elif activation_name == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")
    
class FullyConnectedNetwork(nn.Module):
    def __init__(self, in_features, out_features, activations, widths, zero_at_zero=False, 
                 input_bias=None, input_transform=None, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activations = activations  # List of activation function names
        self.widths = widths  # List of widths for each layer
        self.zero_at_zero = zero_at_zero
        self.dtype = dtype
        if input_bias is None:
            input_bias = torch.zeros(in_features, dtype=self.dtype)
        else:
            input_bias = torch.tensor(input_bias, dtype=self.dtype)
        if input_transform is None:
            input_transform = torch.ones(in_features, dtype=self.dtype)
        else:
            input_transform = torch.tensor(input_transform, dtype=self.dtype)
        self.register_buffer('input_bias', input_bias)
        self.register_buffer('input_transform', input_transform)
        
        if len(self.activations) != len(self.widths) - 2:
            raise ValueError("Number of activations must be two less than number of widths. The last layer has no activation.")
        if self.widths[-1] != self.out_features:
            raise ValueError("Last width must match number of output channels.")
        if self.widths[0] != self.in_features:
            raise ValueError("First width must match number of input channels.")
        
        layers = []
        for i in range(len(self.activations)):
            layer = LinearLayer(in_features=self.widths[i],
                                out_features=self.widths[i+1],
                                bias=True,
                                activation=self.activations[i],
                                dtype=self.dtype)
            
            layers.append(layer)

        layer = LinearLayer(in_features=self.widths[-2],
                            out_features=self.widths[-1],
                            bias=True,
                            activation='identity',
                            dtype=self.dtype)
        layers.append(layer)
    
        self.model = nn.Sequential(*layers)
        self.layers = layers
            
    def forward(self, x):
        x = (x-self.input_bias) * self.input_transform
        out = self.model(x)

        if self.zero_at_zero:
            zeros = torch.zeros_like(x)
            zeros = (zeros-self.input_bias) * self.input_transform
            zero_values = self.model(zeros)
            out = out - zero_values
            
        return out
    
    def forward_with_jacobian(self, x):
        out = (x-self.input_bias) * self.input_transform
        input_transform_expanded = self.input_transform.expand(x.shape[0], -1)
        J = torch.diag_embed(input_transform_expanded).to(self.dtype)

        for layer in self.layers:
            out, jac = layer.forward_with_jacobian(out)
            J = torch.bmm(jac, J)
        
        if self.zero_at_zero:
            zeros = torch.zeros_like(x)
            zeros = (zeros-self.input_bias) * self.input_transform
            zero_values = self.model(zeros)
            out = out - zero_values

        return out, J
    
class LyapunovNetwork(nn.Module): 
    def __init__(self, in_features, out_features, activations, widths, zero_at_zero=False, 
                 input_bias=None, input_transform=None, beta=0.0, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activations = activations  # List of activation function names
        self.widths = widths  # List of widths for each layer
        self.zero_at_zero = zero_at_zero
        self.dtype = dtype
        if input_bias is None:
            input_bias = torch.zeros(in_features, dtype=self.dtype)
        else:
            input_bias = torch.tensor(input_bias, dtype=self.dtype)
        if input_transform is None:
            input_transform = torch.ones(in_features, dtype=self.dtype)
        else:
            input_transform = torch.tensor(input_transform, dtype=self.dtype)
        self.register_buffer('input_bias', input_bias)
        self.register_buffer('input_transform', input_transform)
        self.register_buffer('beta', torch.tensor(beta, dtype=self.dtype))
        
        if len(self.activations) != len(self.widths) - 2:
            raise ValueError("Number of activations must be two less than number of widths. The last layer has no activation.")
        if self.widths[-1] != self.out_features:
            raise ValueError("Last width must match number of output channels.")
        if self.widths[0] != self.in_features:
            raise ValueError("First width must match number of input channels.")
        
        layers = []
        for i in range(len(self.activations)):
            layer = LinearLayer(in_features=self.widths[i],
                                out_features=self.widths[i+1],
                                bias=True,
                                activation=self.activations[i],
                                dtype=self.dtype)
            
            layers.append(layer)

        layer = LinearLayer(in_features=self.widths[-2],
                            out_features=self.widths[-1],
                            bias=True,
                            activation='identity',
                            dtype=self.dtype)
        layers.append(layer)
    
        self.phi = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        x = (x-self.input_bias) * self.input_transform
        out1 = self.phi(x)

        if self.zero_at_zero:
            zeros = torch.zeros_like(x)
            zeros = (zeros-self.input_bias) * self.input_transform
            zero_values = self.phi(zeros)
            out1 = out1 - zero_values
        
        out1 = 0.5 * torch.sum(out1**2, dim=1).unsqueeze(1)
        out2 = 0.5 * self.beta * torch.sum(x**2, dim=1).unsqueeze(1)
        out = out1 + out2
        return out
    
    def forward_with_jacobian(self, x):
        out1 = (x-self.input_bias) * self.input_transform
        input_transform_expanded = self.input_transform.expand(x.shape[0], -1)
        J1 = torch.diag_embed(input_transform_expanded).to(self.dtype)

        for layer in self.layers:
            out1, jac = layer.forward_with_jacobian(out1)
            J1 = torch.bmm(jac, J1)
        
        if self.zero_at_zero:
            zeros = torch.zeros_like(x)
            zeros = (zeros-self.input_bias) * self.input_transform
            zero_values = self.phi(zeros)
            out1 = out1 - zero_values # shape (batch_size, out_features)

        J1 = torch.bmm(out1.unsqueeze(1), J1)

        out1 = 0.5 * torch.sum(out1**2, dim=1).unsqueeze(1)
        out2 = 0.5 * self.beta * torch.sum(x**2, dim=1).unsqueeze(1)
        out = out1 + out2

        J2 = self.beta * x.unsqueeze(1)
        J = J1 + J2

        return out, J