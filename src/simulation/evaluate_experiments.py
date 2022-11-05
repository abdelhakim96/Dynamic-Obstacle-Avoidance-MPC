from matplotlib.cm import ScalarMappable
import numpy as np
import json
import os
import matplotlib.pyplot as plt


def load_experiment_data():
    data = []
    # Load data from json file, adjust your data path here
    for file in os.listdir('test_data/new2/'):
        if file.endswith('.json'):
            fpath = 'test_data/new2/' + file
            f = open(fpath)
            data += [(json.load(f), np.loadtxt(fpath[:-
                      len('_spec.json')] + '_data.csv', delimiter=';'))]
            f.close()
    return data


def print_ratio_collision_avoidance():
    for spec, data in load_experiment_data():
        print(spec)
        print(np.sum(data, axis=0)[0] / data.shape[0])


def mask_data(data):
    # Calculate the average of the ratio of reaching goal by excluding collision cases
    temp_data = data
    for i in range(data.shape[0]):
        if data[i][0] != 0:
            temp_data[i][1] = 0
    return 100*np.sum(temp_data, axis=0)[1] / data.shape[0]


def plot_graph():
    # Store data based on the TF and N_OBST
    data_dict = {}
    for spec, data in load_experiment_data():
        data_dict[(spec['TF'], spec['N_OBST'], spec['scenario'])
                  ] = 100 * np.sum(data, axis=0)[0] / data.shape[0]
    fig, ax = plt.subplots(1, 2, constrained_layout=True,
                           sharey=True, figsize=(8, 5))
    fig.supxlabel('Horizon')
    fig.supylabel('Ratio of collision (%)')
    # plot data
    for key in data_dict:
        if key[2] == "EDGE":
            ax[0].scatter(key[0], data_dict[key], c=key[1],
                          cmap='brg', vmin=5, vmax=30)
        else:
            ax[1].scatter(key[0], data_dict[key], c=key[1],
                          cmap='brg', vmin=5, vmax=30)
    norm = plt.Normalize(5, 30)
    ax[0].set_axisbelow(True)
    ax[0].grid(color='gray', linestyle='dashed')
    ax[1].set_axisbelow(True)
    ax[1].grid(color='gray', linestyle='dashed')
    ax[0].set_title('EDGE')
    ax[1].set_title('RANDOM')
    sm = ScalarMappable(norm=norm, cmap="brg")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax[1])
    cbar.ax.set_title('N_OBST')
    fig.savefig("plot_collision_rate_seperate.svg")

    # plot goal reached average
    goal_reached_dict = {}
    for spec, data in load_experiment_data():
        goal_reached_dict[(spec['TF'], spec['N_OBST'],
                           spec['scenario'])] = mask_data(data)
    fig, ax = plt.subplots(1, 2, constrained_layout=True,
                           sharey=True, figsize=(8, 5))
    fig.supxlabel('Horizon')
    fig.supylabel('Ratio of goal reached (%)')
    # plot data
    for key in goal_reached_dict:
        if key[2] == "EDGE":
            ax[0].scatter(key[0], goal_reached_dict[key],
                          c=key[1], cmap='brg', vmin=5, vmax=30)
        else:
            ax[1].scatter(key[0], goal_reached_dict[key],
                          c=key[1], cmap='brg', vmin=5, vmax=30)
    norm = plt.Normalize(5, 30)
    ax[0].set_axisbelow(True)
    ax[0].grid(color='gray', linestyle='dashed')
    ax[1].set_axisbelow(True)
    ax[1].grid(color='gray', linestyle='dashed')
    ax[0].set_title('EDGE')
    ax[1].set_title('RANDOM')
    sm = ScalarMappable(norm=norm, cmap="brg")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax[1])
    cbar.ax.set_title('N_OBST')
    fig.savefig("plot_goal_reached_rate_seperate.svg")


def plot_graph_qp_solver():
    # Store data based on the QP_ITER
    data_dict = {}
    for spec, data in load_experiment_data():
        data_dict[(spec['QP_ITER'])] = 100 * \
            np.sum(data, axis=0)[0] / data.shape[0]
    fig, ax = plt.subplots(2)
    ax[0].scatter(data_dict.keys(), data_dict.values())
    ax[0].set_ylabel('Ratio of collision (%)')
    ax[0].set_axisbelow(True)
    ax[0].grid(color='gray', linestyle='dashed')
    ax[0].set_ylim([0, 20])
    # plot goal reached average
    goal_reached_dict = {}
    for spec, data in load_experiment_data():
        goal_reached_dict[(spec['QP_ITER'])] = 100 * \
            np.sum(data, axis=0)[1] / data.shape[0]
    ax[1].scatter(goal_reached_dict.keys(), goal_reached_dict.values())
    ax[1].set_xlabel('QP_ITER')
    ax[1].set_ylabel('Ratio of goal reached (%)')
    ax[1].set_axisbelow(True)
    ax[1].grid(color='gray', linestyle='dashed')
    ax[1].set_ylim([80, 100])
    plt.savefig("plot_qp_iter.svg")


if __name__ == '__main__':
    plot_graph_qp_solver()
    # plot_graph()
