from fileinput import close, filename
import numpy as np
import json
import os

# data_init = np.loadtxt('test_data_tf_20_quadratic_slack/EDGE_init_guess_random_move_lowered_slack.csv', delimiter=';')
# data_no_init = np.loadtxt('test_data_tf_20_quadratic_slack/RANDOM_init_guess_random_move_lowered_slack.csv', delimiter=';')
# data_init1 = np.loadtxt('test_data_tf_20_quadratic_slack/EDGE_init_guess_random_move.csv', delimiter=';')
# data_no_init1 = np.loadtxt('test_data_tf_20_quadratic_slack/RANDOM_init_guess_random_move.csv', delimiter=';')
# # data_no_init = np.loadtxt('test_data_tf_20/RANDOM_no_slack_adapt_init_guess.csv', delimiter=';')
# error_ratio = np.sum(data_init, 0)[0] / 100
# error_ratio_no_init = np.sum(data_no_init, 0)[0] / 100
# error_ratio1 = np.sum(data_init1, 0)[0] / 100
# error_ratio_no_init1 = np.sum(data_no_init1, 0)[0] / 100

# print(error_ratio)
# print(error_ratio_no_init)
# print(error_ratio1)
# print(error_ratio_no_init1)

# read out all json files in directory
# and store them together with the np.arrays in a list of tuples
def load_experiment_data():
    data = []
    for file in os.listdir('test_data'):
        if file.endswith('.json'):
            fpath = 'test_data/' + file
            f = open(fpath)
            data += [(json.load(f), np.loadtxt(fpath[:-len('_spec.json')] + '_data.csv', delimiter=';'))]
            # print(data)
            f.close()
    return data

def print_ratio_collision_avoidance():
    for spec, data in load_experiment_data():
        print(spec)
        print(np.sum(data, axis = 0)[0] / data.shape[0])
        
if __name__ == '__main__':
    print_ratio_collision_avoidance()