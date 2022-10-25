import numpy as np

data_init = np.loadtxt('test_data_tf_20_quadratic_slack/EDGE_init_guess_random_move_lowered_slack.csv', delimiter=';')
data_no_init = np.loadtxt('test_data_tf_20_quadratic_slack/RANDOM_init_guess_random_move_lowered_slack.csv', delimiter=';')
data_init1 = np.loadtxt('test_data_tf_20_quadratic_slack/EDGE_init_guess_random_move.csv', delimiter=';')
data_no_init1 = np.loadtxt('test_data_tf_20_quadratic_slack/RANDOM_init_guess_random_move.csv', delimiter=';')
# data_no_init = np.loadtxt('test_data_tf_20/RANDOM_no_slack_adapt_init_guess.csv', delimiter=';')
error_ratio = np.sum(data_init, 0)[0] / 100
error_ratio_no_init = np.sum(data_no_init, 0)[0] / 100
error_ratio1 = np.sum(data_init1, 0)[0] / 100
error_ratio_no_init1 = np.sum(data_no_init1, 0)[0] / 100

print(error_ratio)
print(error_ratio_no_init)
print(error_ratio1)
print(error_ratio_no_init1)