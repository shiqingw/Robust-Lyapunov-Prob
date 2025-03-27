import os
import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np

from cores.utils.utils import load_dict

def diagnosis(exp_num):
    print("==> Exp Num:", exp_num)
    results_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    if not os.path.exists(results_dir):
        results_dir = "{}/eg2_results/{:03d}_keep".format(str(Path(__file__).parent.parent), exp_num)

    all_info = []
    # Step 1
    all_info.append(exp_num)

    # Step 2a:
    N = 4
    times = []
    for ii in range(N):
        filepath = f"{results_dir}/00_sampling_on_nom_linux_cpu_delta_1.0E-03_epsilon_1.0E-03_{ii:02d}.pkl"
        if os.path.exists(filepath):
            results = load_dict(filepath)
            times.append(results["time"])
        else:
            times.append(0)
    assert len(times) == N
    all_info.append(np.mean(times))
    all_info.append(np.std(times))

    # Step 2b:
    times = []
    for ii in range(N):
        filepath = f"{results_dir}/00_sampling_on_nom_linux_cuda_delta_1.0E-03_epsilon_1.0E-03_{ii:02d}.pkl"
        if os.path.exists(filepath):
            results = load_dict(filepath)
            times.append(results["time"])
        else:
            times.append(0)
    assert len(times) == N
    all_info.append(np.mean(times))
    all_info.append(np.std(times))

    # Step 2c:
    times = []
    for ii in range(N):
        filepath = f"{results_dir}/00_sampling_on_true_linux_cpu_delta_1.0E-03_epsilon_1.0E-03_{ii:02d}.pkl"
        if os.path.exists(filepath):
            results = load_dict(filepath)
            times.append(results["time"])
        else:
            times.append(0)
    assert len(times) == N
    all_info.append(np.mean(times))
    all_info.append(np.std(times))

    # Step 2d:
    times = []
    for ii in range(N):
        filepath = f"{results_dir}/00_sampling_on_true_linux_cuda_delta_1.0E-03_epsilon_1.0E-03_{ii:02d}.pkl"
        if os.path.exists(filepath):
            results = load_dict(filepath)
            times.append(results["time"])
        else:
            times.append(0)
    assert len(times) == N
    all_info.append(np.mean(times))
    all_info.append(np.std(times))

    # Step 3a:
    percentages = []
    for ii in range(N):
        filepath = f"{results_dir}/00_sampling_on_nom_linux_cpu_delta_1.0E-03_epsilon_1.0E-03_{ii:02d}.pkl"
        if os.path.exists(filepath):
            results = load_dict(filepath)
            percentages.append(1 - results["good_points"] / results["total_points"])
        else:
            percentages.append(0)
    assert len(percentages) == N
    all_info.append(np.mean(percentages))
    all_info.append(np.std(percentages))

    # Step 3b:
    percentages = []
    for ii in range(N):
        filepath = f"{results_dir}/00_sampling_on_nom_linux_cuda_delta_1.0E-03_epsilon_1.0E-03_{ii:02d}.pkl"
        if os.path.exists(filepath):
            results = load_dict(filepath)
            percentages.append(1 - results["good_points"] / results["total_points"])
        else:
            percentages.append(0)
    assert len(percentages) == N
    all_info.append(np.mean(percentages))
    all_info.append(np.std(percentages))

    # Step 3c:
    percentages = []
    for ii in range(N):
        filepath = f"{results_dir}/00_sampling_on_true_linux_cpu_delta_1.0E-03_epsilon_1.0E-03_{ii:02d}.pkl"
        if os.path.exists(filepath):
            results = load_dict(filepath)
            percentages.append(1 - results["good_points"] / results["total_points"])
        else:
            percentages.append(0)
    assert len(percentages) == N
    all_info.append(np.mean(percentages))
    all_info.append(np.std(percentages))

    # Step 3d:
    percentages = []
    for ii in range(N):
        filepath = f"{results_dir}/00_sampling_on_true_linux_cuda_delta_1.0E-03_epsilon_1.0E-03_{ii:02d}.pkl"
        if os.path.exists(filepath):
            results = load_dict(filepath)
            percentages.append(1 - results["good_points"] / results["total_points"])
        else:
            percentages.append(0)
    assert len(percentages) == N
    all_info.append(np.mean(percentages))
    all_info.append(np.std(percentages))
    
    return all_info

if __name__ == "__main__":
    # save to a txt file with separator that can be directly copy pasted to excel-
    with open("text.txt", "w") as file:
        exp_nums = list(range(1, 3))
        for exp_num in exp_nums:
            # try:
            out = diagnosis(exp_num)
            print("#############################################")
            for ii in out:
                file.write(f"{ii}\t")
            file.write(f"\n")
            # except:
            #     file.write(f"\n")