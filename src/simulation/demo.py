import sys
sys.path.append('../')
import numpy as np
from robot_ocp_problem import RobotOcpProblem
from models.world_specification import X_MIN_ROBOT, X_MAX_ROBOT, Y_MIN_ROBOT, Y_MAX_ROBOT



ocp = RobotOcpProblem(np.array([X_MIN_ROBOT, Y_MIN_ROBOT, np.pi/4, 0, 0]), np.array([X_MAX_ROBOT, Y_MAX_ROBOT]), 0, slack=True)

# random examples
configurations = [('RANDOM', 0), ('RANDOM', 1), ('RANDOM', 5), ('RANDOM', 9), ('EDGE', 4), ('EDGE', 6), ('EDGE', 1), ('EDGE', 9)]
for scenario, seed in configurations:
    ocp.set_up_new_experiment(seed, scenario, init_guess_when_error=True, random_move=True)
    ocp.step(400, True)
    
ocp.set_up_new_experiment(9, 'EDGE', init_guess_when_error=True)
ocp.step(400, True)