mkdir eg1_results/001
mkdir eg1_results/001
python -u eg1_inverted_pendulum/train.py --exp_num 1 --device cuda > eg1_results/001/output.out
mkdir eg1_results/002
mkdir eg1_results/002
python -u eg1_inverted_pendulum/train.py --exp_num 2 --device cuda > eg1_results/002/output.out
mkdir eg1_results/003
mkdir eg1_results/003
python -u eg1_inverted_pendulum/train.py --exp_num 3 --device cuda > eg1_results/003/output.out
