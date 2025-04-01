import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def generate_sh_script_nominal(filename, exp_nums, device):
    with open(filename, "w") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg3_results/{exp_num:03}\n"
            command2 = f"python -u eg3_cartpole/train_nominal.py --exp_num {exp_num} --device {device} > eg3_results/{exp_num:03}/output.out\n"
            file.write(command1)
            file.write(command1)
            file.write(command2)

def generate_sh_script_true(filename, exp_nums, device):
    with open(filename, "a") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg3_results/{exp_num:03}\n"
            command2 = f"python -u eg3_cartpole/train_true.py --exp_num {exp_num} --device {device} > eg3_results/{exp_num:03}/output.out\n"
            file.write(command1)
            file.write(command1)
            file.write(command2)

device = "cuda"
file = os.path.join(str(Path(__file__).parent.parent), f"run_cuda_train_one.sh")

exp_nums = [3]
generate_sh_script_nominal(file, exp_nums, device)

# exp_nums = [2]
# generate_sh_script_true(file, exp_nums, device)