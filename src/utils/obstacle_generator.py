import sys
sys.path.append('../')

import numpy as np
from utils.visualization import Obstacle
from models.world_specification import *

def generate_random_moving_obstacles(seed=None, scenario='RANDOM'):
    np.random.seed(seed)
    if scenario == 'RANDOM':
        x_coordinates = np.random.uniform(X_MIN_OBST, X_MAX_OBST, (N_OBST, 1))
        y_coordinates = np.random.uniform(Y_MIN_OBST, Y_MAX_OBST, (N_OBST, 1))
    elif scenario == 'CENTER':
        x_coordinates = np.zeros((N_OBST, 1))
        y_coordinates = np.zeros((N_OBST, 1))
    elif scenario == 'EDGE':
        x_coordinates = 7 * np.ones((N_OBST, 1))
        y_coordinates = 7 * np.ones((N_OBST, 1))
    
    
    vx = np.random.uniform(-V_MAX_OBST, V_MAX_OBST, (N_OBST, 1))
    vy = np.random.uniform(-V_MAX_OBST, V_MAX_OBST, (N_OBST, 1))
    
    spec = np.hstack((x_coordinates, y_coordinates, vx, vy))
    obstacles = []
    for s in spec:
        obstacles += [Obstacle(*s)]
    return obstacles
