import sys
sys.path.append('../')

import numpy as np
from models.world_specification import X_MAX, X_MIN, Y_MAX, Y_MIN
from robot_ocp_problem import RobotOcpProblem

# scenarios = ['RANDOM', 'EDGE', 'CENTER']
scenarios = ['RANDOM', 'EDGE']
# init_guess = [True, False]
init_guess = [True]

ocp = RobotOcpProblem(np.array([X_MIN + 1, Y_MIN + 1, np.pi / 4, 0, 0]), np.array([X_MAX - 1, Y_MAX - 1]), seed=0)

for s in scenarios:
    for ini in init_guess:
        data = np.ndarray((100, 6))
        init_guess_str = 'init_guess' if ini else 'no_init_guess'
        for i in range(100):
            print(f'{s}, {init_guess_str} solving problem: {i}')
            ocp.set_up_new_experiment(i, s, ini, random_move=True)
            data[i] = ocp.step(400)[1:]
        np.savetxt(f'test_data_tf_20_quadratic_slack/{s}_{init_guess_str}_random_move_lowered_slack.csv', data, delimiter=';')
