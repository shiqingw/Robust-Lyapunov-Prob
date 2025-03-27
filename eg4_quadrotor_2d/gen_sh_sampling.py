import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def generate_sh_script_sampling_solve_on_nom(filename, exp_nums, device, delta, epsilon):
    with open(filename, "a") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg4_results/{exp_num:03d}\n"
            command2 = f"python -u eg4_quadrotor_2d/sampling_on_nom.py --exp_num {exp_num} " + \
                f"--device {device} " + \
                f"--delta {delta:.1E} " + \
                f"--epsilon {epsilon:.1E} " + \
                f"> eg4_results/{exp_num:03d}/output_sampling_on_nom_delta_{delta:.1E}_epsilon_{epsilon:.1E}.out\n"
            file.write(command1)
            file.write(command2)

def generate_sh_script_sampling_solve_on_true(filename, exp_nums, device, delta, epsilon):
    with open(filename, "a") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg4_results/{exp_num:03d}\n"
            command2 = f"python -u eg4_quadrotor_2d/sampling_on_true.py --exp_num {exp_num} " + \
                f"--device {device} " + \
                f"--delta {delta:.1E} " + \
                f"--epsilon {epsilon:.1E} " + \
                f"> eg4_results/{exp_num:03d}/output_sampling_on_true_delta_{delta:.1E}_epsilon_{epsilon:.1E}.out\n"
            file.write(command1)
            file.write(command2)


file = os.path.join(str(Path(__file__).parent.parent), f"run_sampling_solve.sh")
delta = 1e-3
epsilon = 1e-3

device = "cuda"
exp_nums = [1]
generate_sh_script_sampling_solve_on_nom(file, exp_nums, device, delta, epsilon)

exp_nums = [1,2]
generate_sh_script_sampling_solve_on_true(file, exp_nums, device, delta, epsilon)

device = "cpu"
exp_nums = [1]
generate_sh_script_sampling_solve_on_nom(file, exp_nums, device, delta, epsilon)

exp_nums = [1,2]
generate_sh_script_sampling_solve_on_true(file, exp_nums, device, delta, epsilon)
