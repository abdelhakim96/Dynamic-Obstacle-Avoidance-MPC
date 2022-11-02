"""
    Here we collect all variables that should be fixed for setting up a simulation environment.
"""


# define boundaries of the grid world
Y_MIN = - 8
Y_MAX = - Y_MIN
X_MIN = Y_MIN
X_MAX = Y_MAX

# define limits for start and end position of robot
R_ROBOT = 0.2
V_MAX_ROBOT = 10
# MARGIN = 0
Y_MIN_ROBOT = Y_MIN + 2
Y_MAX_ROBOT = - Y_MIN_ROBOT
X_MIN_ROBOT = Y_MIN_ROBOT
X_MAX_ROBOT = Y_MAX_ROBOT

# define limits on controls
C_MAX = 8

# fix number and shape of obstacles in the grid
N_OBST = 5
R_MIN_OBST = 0.6
R_MAX_OBST = 1.0
R_OBST = 1
RANDOMNESS = 0.1
# V_MAX_OBST = V_MAX_ROBOT / 8
V_MAX_OBST = 2
# V_MAX_OBST = 0.5
# MARGIN = 3 * RANDOMNESS * V_MAX_OBST
# MARGIN = 1.5
MARGIN = 1.2
# define limits where obstacles can be placed
Y_MIN_OBST = Y_MIN_ROBOT + R_MAX_OBST + 3 * R_ROBOT
Y_MAX_OBST = - Y_MIN_ROBOT
X_MIN_OBST = Y_MIN_OBST
X_MAX_OBST = Y_MAX_ROBOT

# define timeframe and number of steps to simulate
TF = 2
DT = 10
N_SOLV = int(TF * DT)
TOL = 0.15

# define the number of iterations to solve the QP
QP_ITER = 50
