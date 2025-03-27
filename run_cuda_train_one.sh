mkdir eg4_results/001
mkdir eg4_results/001
python -u eg4_quadrotor_2d/train_nominal.py --exp_num 1 --device cuda > eg4_results/001/output.out
mkdir eg4_results/002
mkdir eg4_results/002
python -u eg4_quadrotor_2d/train_true.py --exp_num 2 --device cuda > eg4_results/002/output.out
