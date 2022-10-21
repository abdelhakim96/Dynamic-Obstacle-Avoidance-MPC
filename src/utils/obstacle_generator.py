print("entered obstacle_generator")
import numpy as np
print("imported numpy to obstacle_generator")
from utils.visualization import Obstacle
print("imported Obstacle to obstacle_generator")
from models.world_specification import *

def generate_random_obstacles(seed=None):
    # sample a grid
    if seed is not None:
        np.random.seed(seed)
    x_coordinates = np.random.uniform(X_MIN_OBST, X_MAX_OBST, (N_OBST, 1))
    y_coordinates = np.random.uniform(Y_MIN_OBST, X_MAX_OBST, (N_OBST, 1))
    radius = np.random.uniform(R_MIN_OBST, R_MAX_OBST, (N_OBST, 1))
    coordinates = np.hstack((x_coordinates, y_coordinates, radius))
    obstacles = []
    for coord in coordinates:
        obstacles += [Obstacle(coord[0], coord[1], coord[2])]
    return obstacles

def generate_regular_obstacle_grid(N, M, x_min, x_max, y_min, y_max, r_min, r_max, seed):
    np.random.seed(seed)
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, M)
    xv, yv = np.meshgrid(x, y)
    obstacles = []
    for i in range(N):
        for j in range(M):
            obstacles += [Obstacle(xv[j, i], yv[j, i], np.random.uniform(r_min, r_max))]
    return obstacles
