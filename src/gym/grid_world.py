from cmath import sqrt
from typing import Tuple
import gym
from simulation.robot_ocp import solve_robot_ocp_closed_loop
from utils.visualization import Obstacle
import numpy as np

def obstacle_creator(num_obstacles, obstacle_radius, x_min, y_min, x_max, y_max):
    x_range= x_max - x_min
    y_range= y_max - y_min
    N = sqrt(num_obstacles)
    obs = []
    for i in range(N):
        for j in range(N):
            x = x_min + (i+0.5) * (x_range) / (N)
            y = y_min + (j+0.5) * (y_range) / (N)
            obs.append(Obstacle(x, y, obstacle_radius))
    return obs

class GridWorld(gym.Env):

    def __init__(self, n_obstacles=4, obstacle_radius=0.4, temperature=0.99):
        self.x_init = -2
        self.y_init = -2
        self.x_goal = 2
        self.y_goal = 2

        #Hyperparameter to decide if the goal is reached
        self.threshold = 0.1
        self.temp = temperature

        self.x = self.x_init
        self.y = self.y_init
        self.state= np.array([self.x, self.y])
        self.r = obstacle_radius
        self.obstacles = obstacle_creator(n_obstacles, obstacle_radius, self.x_init, self.y_init, self.x_goal, self.y_goal)
        

    def step(self, action):

        # We need this function to output the current position of the robot after giving x_subgoal and y_subgoal
        next_state = solve_robot_ocp_closed_loop(self.state, action, self.r, self.obstacles)
        reward, terminal_state = self.reward_function(next_state)

        return next_state, reward, terminal_state, {}

    def reward_function(self, state) -> Tuple[float, bool]:
        """
        This method returns the reward for the current state, and if the goal is reached/terminal state
        """
        distance_to_goal = np.linalg.norm(state - np.array([self.x_goal, self.y_goal]))
        normalization_factor =  np.linalg.norm(np.array([self.x_init, self.y_init]) - np.array([self.x_goal, self.y_goal]))
        normalized_reward = 1 - distance_to_goal / normalization_factor
        normalized_reward = normalized_reward * self.temp
        check_obstacles_hit = self.check_obstacles_hit(state)
        if check_obstacles_hit:
            return -100, True
        else:
            if distance_to_goal < self.threshold:
                return 100, True
            else:
                return normalized_reward, False
    
    def check_if_obstacles_hit(self, state):
        for obs in self.obstacles:
            if np.linalg.norm(obs.center - state) < obs.radius:
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = self.x_init
        self.y = self.y_init


    # This is the part for visiualization, not urgent
    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass