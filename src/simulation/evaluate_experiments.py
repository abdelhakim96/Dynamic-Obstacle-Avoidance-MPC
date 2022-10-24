import numpy as np

data_1 = np.loadtxt('test_data_tf_1/EDGE_init_guess.csv', delimiter=';')
data_15 = np.loadtxt('test_data_tf_15/EDGE_init_guess.csv', delimiter=';')
error_ratio = np.sum(data_1, 0)[0] / 100
error_ratio = np.sum(data_15, 0)[0] / 100

print(error_ratio)