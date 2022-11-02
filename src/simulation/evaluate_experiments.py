from fileinput import close, filename
import numpy as np
import json
import os
import matplotlib.pyplot as plt

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
    for file in os.listdir('test_data/multiple/'):
        if file.endswith('.json'):
            fpath = 'test_data/multiple/' + file
            f = open(fpath)
            data += [(json.load(f), np.loadtxt(fpath[:-len('_spec.json')] + '_data.csv', delimiter=';'))]
            # print(data)
            f.close()
    return data

def print_ratio_collision_avoidance():
    for spec, data in load_experiment_data():
        print(spec)
        print(np.sum(data, axis = 0)[0] / data.shape[0])


def plot_graph():
    # Store data based on the TF and N_OBST
    data_dict = {}
    for spec, data in load_experiment_data():
        data_dict[(spec['TF'], spec['N_OBST'])] = np.sum(data, axis = 0)[0] / data.shape[0]
    plt.figure(1)
    # plot data
    for key in data_dict:
        plt.scatter(key[0], data_dict[key], c=key[1], cmap='inferno', vmin=5, vmax=30)
    plt.grid()
    plt.xlabel('TF')
    plt.ylabel('Ratio of collision')
    plt.colorbar(label='N_OBST')
    plt.title('Collision of the robot with the obstacles with respect to horizon')
    plt.savefig("plot.png")

    # plot goal reached average
    goal_reached_dict = {}
    for spec, data in load_experiment_data():
        goal_reached_dict[(spec['TF'], spec['N_OBST'])] = np.sum(data, axis = 0)[1] / data.shape[0]
    plt.figure(2)
    for key in data_dict:
        plt.scatter(key[0], goal_reached_dict[key], c=key[1], cmap='inferno', vmin=5, vmax=30)
    plt.grid()
    plt.xlabel('TF')
    plt.ylabel('Ratio of goal reached')
    plt.colorbar(label='N_OBST')
    plt.title('Goal reached with respect to horizon')
    plt.savefig("plot_goal_reached.png")

if __name__ == '__main__':
    plot_graph()