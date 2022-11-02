import sys
import os


import numpy as np
from models.world_specification import X_MAX, X_MIN, Y_MAX, Y_MIN, QP_ITER, TF, N_OBST, N_SOLV
from robot_ocp_problem import RobotOcpProblem
from datetime import datetime
import json


def run_experiment():
    # scenarios = ['RANDOM', 'EDGE', 'CENTER']
    scenarios = ['RANDOM', 'EDGE']
    # init_guess = [True, False]
    init_strategy = [True]

    slack = True
    random_move = True
    ocp = RobotOcpProblem(np.array([X_MIN + 1, Y_MIN + 1, np.pi / 4, 0, 0]), np.array([X_MAX - 1, Y_MAX - 1]))

    for s in scenarios:
        for ini in init_strategy:
            # get timestamp for file to save
            time = datetime.now()
            timestamp = f"{time.year:04d}{time.month:02d}{time.day:02d}_{time.hour:02d}{time.minute:02d}{time.second:02d}"
            timestamp2 = f"{time.year:04d}{time.month:02d}{time.day:02d}"
            data = np.ndarray((100, 6))
            init_guess_str = 'init_guess' if ini else 'no_init_guess'
            # store experimental data in dictionary and write it to json file
            experiment_dict = {"slack": slack, "random_move": random_move, "init_guess": ini, "scenario": s, "TF": TF, "N_SOLV": N_SOLV, "N_OBST": N_OBST, "QP_ITER": QP_ITER}
            for i in range(100):
                np.random.seed(i)
                print(f'{s}, {init_guess_str} solving problem: {i}')
                ocp.set_up_new_experiment(s, ini, random_move=True)
                data[i] = ocp.step(400)[1:]
            # store the data to a .csv
            if not os.path.exists(f'test_data/multiple/'):
                os.makedirs(f'test_data/multiple/')
            np.savetxt(f'test_data/multiple/{timestamp}_experiment_data.csv', data, delimiter=';')
            # store the experimental parameters to a json file
            with open(f'test_data/multiple/{timestamp}_experiment_spec.json', 'w') as json_file:
                json.dump(experiment_dict, json_file)

if __name__ == '__main__':
    run_experiment()