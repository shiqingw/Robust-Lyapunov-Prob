import json
import sys
import os
from .inverted_pendulum import InvertedPendulum
from .strict_feedback_3d import StrictFeedback3D
from .cartpole import CartPole
from .quadrotor_2d import Quadrotor2D
from pathlib import Path

def get_system(system_name, dtype):
    with open(os.path.join(str(Path(__file__).parent), "system_params.json"), 'r') as f:
        data = json.load(f)
    if system_name not in data:
        raise ValueError("System name not found in system_params.json")
    data = data[system_name]
    if data["type"] == "InvertedPendulum":
        return InvertedPendulum(mass=data["params"]["mass"], 
                                length=data["params"]["length"], 
                                viscous_friction=data["params"]["viscous_friction"],
                                dtype=dtype)
    elif data["type"] == "StrictFeedback3D":
        return StrictFeedback3D(a1=data["params"]["a1"], 
                                a2=data["params"]["a2"], 
                                b1=data["params"]["b1"], 
                                b2=data["params"]["b2"], 
                                c1=data["params"]["c1"], 
                                c2=data["params"]["c2"],
                                dtype=dtype)
    elif data["type"] == "CartPole":
        return CartPole(mass_cart=data["params"]["mass_cart"], 
                        mass_pole=data["params"]["mass_pole"], 
                        length=data["params"]["length"], 
                        friction_coef=data["params"]["friction_coef"],
                        dtype=dtype)
    elif data["type"] == "Quadrotor2D":
        return Quadrotor2D(mass=data["params"]["mass"], 
                           inertia=data["params"]["inertia"], 
                           arm_length=data["params"]["arm_length"], 
                           dtype=dtype)
    else:
        raise ValueError("System type not found in systems.json")


