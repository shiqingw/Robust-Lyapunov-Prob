import os

exp_num = 2
result_dir = f"/Users/shiqing/Desktop/Robust-Lyapunov-Prob/eg4_results/{exp_num:03d}"

for i in range(4):
    old_name = f"00_sampling_nom_on_nom_linux_cpu_delta_1.0E-03_epsilon_1.0E-03_{i:02}.pkl"
    new_name = f"00_sampling_on_nom_linux_cpu_delta_1.0E-03_epsilon_1.0E-03_{i:02}.pkl"
    old_path = os.path.join(result_dir, old_name)
    new_path = os.path.join(result_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")
    else:
        print(f"{old_name} does not exist.")

for i in range(4):
    old_name = f"00_sampling_nom_on_nom_linux_cuda_delta_1.0E-03_epsilon_1.0E-03_{i:02}.pkl"
    new_name = f"00_sampling_on_nom_linux_cuda_delta_1.0E-03_epsilon_1.0E-03_{i:02}.pkl"
    old_path = os.path.join(result_dir, old_name)
    new_path = os.path.join(result_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")
    else:
        print(f"{old_name} does not exist.")

for i in range(4):
    old_name = f"00_sampling_nom_on_true_linux_cpu_delta_1.0E-03_epsilon_1.0E-03_{i:02}.pkl"
    new_name = f"00_sampling_on_true_linux_cpu_delta_1.0E-03_epsilon_1.0E-03_{i:02}.pkl"
    old_path = os.path.join(result_dir, old_name)
    new_path = os.path.join(result_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")
    else:
        print(f"{old_name} does not exist.")

for i in range(4):
    old_name = f"00_sampling_nom_on_true_linux_cuda_delta_1.0E-03_epsilon_1.0E-03_{i:02}.pkl"
    new_name = f"00_sampling_on_true_linux_cuda_delta_1.0E-03_epsilon_1.0E-03_{i:02}.pkl"
    old_path = os.path.join(result_dir, old_name)
    new_path = os.path.join(result_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")
    else:
        print(f"{old_name} does not exist.")