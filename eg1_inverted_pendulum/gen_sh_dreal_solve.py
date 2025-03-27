import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def generate_sh_script_dreal_positive_solve(filename, exp_nums, dreal_precision):
    with open(filename, "w") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg1_results/{exp_num:03d}\n"
            command2 = f"python -u eg1_inverted_pendulum/dreal_positive_solve.py --exp_num {exp_num} " + \
                f"--dreal_precision {dreal_precision:.1E} " + \
                f"> eg1_results/{exp_num:03d}/output_dreal_positive_solve_{dreal_precision:.1E}.out\n"
            file.write(command1)
            file.write(command2)

def generate_sh_script_dreal_stability_solve(filename, exp_nums, dreal_precision):
    with open(filename, "w") as file:
        for exp_num in exp_nums:
            command1 = f"mkdir eg1_results/{exp_num:03d}\n"
            command2 = f"python -u eg1_inverted_pendulum/dreal_stability_solve.py --exp_num {exp_num} " + \
                f"--dreal_precision {dreal_precision:.1E} " + \
                f"> eg1_results/{exp_num:03d}/output_dreal_stability_solve_{dreal_precision:.1E}.out\n"
            file.write(command1)
            file.write(command2)

exp_nums = list(range(3, 4))

dreal_precision = 1e-2

file = os.path.join(str(Path(__file__).parent.parent), f"run_dreal_stability_solve.sh")
generate_sh_script_dreal_stability_solve(file, exp_nums, dreal_precision)