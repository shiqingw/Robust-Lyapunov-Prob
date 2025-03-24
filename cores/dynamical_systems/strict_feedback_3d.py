import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union

def max_alpha_beta_expression(a1, b1, a2, b2, alpha, beta):
    """
    Computes the maximum of |alpha * x1 + beta * x2|
    where x1 is in [a1, b1] and x2 is in [a2, b2].

    Parameters:
    a1 (float): Lower bound for x1
    b1 (float): Upper bound for x1
    a2 (float): Lower bound for x2
    b2 (float): Upper bound for x2
    alpha (float): Coefficient for x1
    beta (float): Coefficient for x2

    Returns:
    float: The maximum absolute value of (alpha * x1 + beta * x2)
    """
    # Define the four corner points
    points = [
        (a1, a2),
        (a1, b2),
        (b1, a2),
        (b1, b2)
    ]
    
    # Compute |alpha * x1 + beta * x2| for each corner
    max_val = max(abs(alpha * x1 + beta * x2) for x1, x2 in points)
    
    return max_val

class StrictFeedback3D(nn.Module):
    """
    A neural network module representing a strict feedback 3D system.

    This module defines a dynamical system with state dimension 3 and control dimension 1.
    The system dynamics are defined by parameters `a1`, `a2`, `b1`, `b2`, `c1`, and `c2`.

    Attributes:
        state_dim (int): Dimension of the state vector (3).
        control_dim (int): Dimension of the control vector (1).
        dtype (torch.dtype): Data type for tensors.
        a1 (torch.Tensor): Parameter `a1`.
        a2 (torch.Tensor): Parameter `a2`.
        b1 (torch.Tensor): Parameter `b1`.
        b2 (torch.Tensor): Parameter `b2`.
        c1 (torch.Tensor): Parameter `c1`.
        c2 (torch.Tensor): Parameter `c2`.
    """

    def __init__(self, a1: float, a2: float, b1: float, b2: float, c1: float, c2: float,
                 dtype: torch.dtype = torch.float32) -> None:
        """
        Initialize the StricFeedback3D module with system parameters.

        Args:
            a1 (float): Parameter `a1`.
            a2 (float): Parameter `a2`.
            b1 (float): Parameter `b1`.
            b2 (float): Parameter `b2`.
            c1 (float): Parameter `c1`.
            c2 (float): Parameter `c2`.
            dtype (torch.dtype, optional): Data type for tensors. Defaults to `torch.float32`.
        """

        super(StrictFeedback3D, self).__init__()

        self.state_dim = 3
        self.control_dim = 1
        self.dtype = dtype

        self.register_buffer('a1', torch.tensor(a1, dtype=self.dtype))
        self.register_buffer('a2', torch.tensor(a2, dtype=self.dtype))
        self.register_buffer('b1', torch.tensor(b1, dtype=self.dtype))
        self.register_buffer('b2', torch.tensor(b2, dtype=self.dtype))
        self.register_buffer('c1', torch.tensor(c1, dtype=self.dtype))
        self.register_buffer('c2', torch.tensor(c2, dtype=self.dtype))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the state derivatives given the current state and control input.

        Args:
            x (torch.Tensor): Current state tensor of shape `(batch_size, 3)`.
            u (torch.Tensor): Control input tensor of shape `(batch_size, 1)`.

        Returns:
            torch.Tensor: State derivatives tensor of shape `(batch_size, 3)`.
        """

        x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        a1 = self.a1
        a2 = self.a2
        b1 = self.b1
        b2 = self.b2
        c1 = self.c1
        c2 = self.c2

        dx1 = a1 * x1 + a2 * x2
        dx2 = b1 * x2 + b2 * x3
        dx3 = c1 * x1**2 + c2 * u

        dx = torch.cat((dx1, dx2, dx3), dim=1)

        return dx
    
    def linearize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearize the system around the origin.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the linearized system matrices `(A, B)`.
                - `A` (np.ndarray): The system matrix of shape `(3, 3)`.
                - `B` (np.ndarray): The input matrix of shape `(3, 1)`.
        """

        a1 = self.a1.item()
        a2 = self.a2.item()
        b1 = self.b1.item()
        b2 = self.b2.item()
        c2 = self.c2.item()

        A = np.array([[a1, a2, 0],
                      [0, b1, b2],
                      [0, 0, 0]], dtype=np.float32)
        
        B = np.array([[0],
                      [0],
                      [c2]], dtype=np.float32)
        
        return A, B
    
    def get_drift(self, x: torch.Tensor) -> torch.Tensor:
        
        drift = torch.zeros(x.shape[0], self.state_dim, dtype=self.dtype, device=x.device)
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]

        a1 = self.a1
        a2 = self.a2
        b1 = self.b1
        b2 = self.b2
        c1 = self.c1

        drift[:, 0] = a1 * x1 + a2 * x2
        drift[:, 1] = b1 * x2 + b2 * x3
        drift[:, 2] = c1 * x1**2

        return drift
    
    def get_actuation(self, x: torch.Tensor) -> torch.Tensor:

        actuation = torch.zeros(x.shape[0], self.state_dim, self.control_dim, dtype=self.dtype, device=x.device)
        
        c2 = self.c2
        actuation[:, 2, 0] = c2

        return actuation
    