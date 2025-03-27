mkdir eg1_results/001
python -u eg1_inverted_pendulum/sampling_on_nom.py --exp_num 1 --device cuda --delta 1.0E-03 --epsilon 1.0E-03 > eg1_results/001/output_sampling_on_nom_cuda_delta_1.0E-03_epsilon_1.0E-03.out
mkdir eg1_results/001
python -u eg1_inverted_pendulum/sampling_on_true.py --exp_num 1 --device cuda --delta 1.0E-03 --epsilon 1.0E-03 > eg1_results/001/output_sampling_on_true_cuda_delta_1.0E-03_epsilon_1.0E-03.out
mkdir eg1_results/002
python -u eg1_inverted_pendulum/sampling_on_true.py --exp_num 2 --device cuda --delta 1.0E-03 --epsilon 1.0E-03 > eg1_results/002/output_sampling_on_true_cuda_delta_1.0E-03_epsilon_1.0E-03.out
mkdir eg1_results/001
python -u eg1_inverted_pendulum/sampling_on_nom.py --exp_num 1 --device cpu --delta 1.0E-03 --epsilon 1.0E-03 > eg1_results/001/output_sampling_on_nom_cpu_delta_1.0E-03_epsilon_1.0E-03.out
mkdir eg1_results/001
python -u eg1_inverted_pendulum/sampling_on_true.py --exp_num 1 --device cpu --delta 1.0E-03 --epsilon 1.0E-03 > eg1_results/001/output_sampling_on_true_cpu_delta_1.0E-03_epsilon_1.0E-03.out
mkdir eg1_results/002
python -u eg1_inverted_pendulum/sampling_on_true.py --exp_num 2 --device cpu --delta 1.0E-03 --epsilon 1.0E-03 > eg1_results/002/output_sampling_on_true_cpu_delta_1.0E-03_epsilon_1.0E-03.out
