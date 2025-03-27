mkdir eg1_results/001
python -u eg1_inverted_pendulum/sampling_on_nom.py --exp_num 1 --delta 1.0E-03 --epsilon 1.0E-03 > eg1_results/001/output_sampling_on_nom_delta_1.0E-03_epsilon_1.0E-03.out
mkdir eg1_results/001
python -u eg1_inverted_pendulum/sampling_on_true.py --exp_num 1 --delta 1.0E-03 --epsilon 1.0E-03 > eg1_results/001/output_sampling_on_true_delta_1.0E-03_epsilon_1.0E-03.out
mkdir eg1_results/002
python -u eg1_inverted_pendulum/sampling_on_true.py --exp_num 2 --delta 1.0E-03 --epsilon 1.0E-03 > eg1_results/002/output_sampling_on_true_delta_1.0E-03_epsilon_1.0E-03.out
