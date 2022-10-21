"""
    Here we collect all variables that should be fixed for setting up a simulation environment.
"""


# define boundaries of the grid world
Y_MIN = -7
Y_MAX = - Y_MIN
X_MIN = Y_MIN
X_MAX = Y_MAX

# define limits for start and end position of robot
R_ROBOT = 0.2
MARGIN = 0.05
Y_MIN_ROBOT = Y_MIN + 1
Y_MAX_ROBOT = - Y_MIN_ROBOT
X_MIN_ROBOT = Y_MIN_ROBOT
X_MAX_ROBOT = Y_MAX_ROBOT

# fix number and shape of obstacles in the grid
N_OBST = 25
R_MIN_OBST = 0.6
R_MAX_OBST = 1.0
# define limits where obstacles can be placed
Y_MIN_OBST = Y_MIN_ROBOT + R_MAX_OBST + 3 * R_ROBOT
Y_MAX_OBST = - Y_MIN_ROBOT
X_MIN_OBST = Y_MIN_OBST
X_MAX_OBST = Y_MAX_ROBOT

# define timeframe and number of steps to simulate
N_SOLV = 10
TF = 1
TOL = 0.1