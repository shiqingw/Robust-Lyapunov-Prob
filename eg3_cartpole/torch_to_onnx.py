import json
import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cores.neural_network.models import FullyConnectedNetwork
from cores.utils.config import Configuration

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    parser.add_argument('--device', default="None", type=str, help='device number')
    args = parser.parse_args()

    # Create result directory
    print("==> Creating result directory ...")
    exp_num = args.exp_num
    results_dir = "{}/eg3_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
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

    x = torch.tensor([[0,0,0,0],
                      [1,1,1,1],
                      [2,2,2,2],
                      [3,3,3,3],
                      [-3,-3,-3,-3]], dtype=config.pt_dtype).to(device)
    print(controller_nn(x).cpu().detach().numpy())

    torch_model = controller_nn
    torch_model.eval()
    batch_size = 10
    x = torch.randn(batch_size, controller_in_features, dtype=config.pt_dtype, requires_grad=True).to(device)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    os.path.join(results_dir,"controller.onnx"),   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes = {'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    print("==> Controller saved!")