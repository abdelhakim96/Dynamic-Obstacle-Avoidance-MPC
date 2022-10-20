from cmath import sqrt
from typing import Tuple
import gym
from simulation.robot_ocp import solve_robot_ocp_closed_loop
import numpy as np
import torch

class GridWorld(gym.Env):

    def __init__(self, obstacles, x_init, y_init, x_goal, y_goal, robot_radius=0.2, temperature=0.99):
        self.x_init = x_init
        self.y_init = y_init
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.psi = np.pi / 4
        self.v= 0
        self.omega = 0

        #Hyperparameter to decide if the goal is reached
        self.threshold = 0.1
        self.temp = temperature
        self.x = self.x_init
        self.y = self.y_init
        self.obstacles = obstacles
        self.position= np.array([self.x, self.y])
        self.robot_rad = robot_radius

        self.state= self.get_state(self.position)
                
        
    
    def get_state(self, position):
            state =  np.zeros(shape=(len(self.obstacles)+1, 3))
            state[0] = np.array([*position, self.robot_rad])
            for i in range(1, len(self.obstacles)+1):
                state[i] = np.array([self.obstacles[i-1].x , self.obstacles[i-1].y, self.obstacles[i-1].r])
            return state

    def step(self, action):

        # We need this function to output the current position of the robot after giving x_subgoal and y_subgoal
        solver_output= solve_robot_ocp_closed_loop(np.array([*self.position, self.psi, self.v, self.omega]),
                                                    np.array(action.cpu()).ravel(),
                                                    self.obstacles, 100)
        self.position = solver_output[0][0:2]
        self.psi = solver_output[0][2]
        self.v = solver_output[0][3]
        self.omega = solver_output[0][4]
        object_hit, done = solver_output[1], solver_output[2]
        reward, terminal_state = self.reward_function(self.position, object_hit, done)
        next_state = self.get_state(self.position)
        next_state = torch.Tensor(next_state).reshape(1, -1)
        return next_state, torch.Tensor([reward]), torch.Tensor([terminal_state]), {}

    def reward_function(self, state, object_hit, done) -> Tuple[float, bool]:
        """
        This method returns the reward for the current state, and if the goal is reached/terminal state
        """
        if object_hit:
            return -100, True
        elif done:
            return 100, True
        else:
            distance_to_goal = np.linalg.norm(state - np.array([self.x_goal, self.y_goal]))
            normalization_factor =  np.linalg.norm(np.array([self.x_init, self.y_init]) - np.array([self.x_goal, self.y_goal]))
            normalized_reward = 1 - distance_to_goal / normalization_factor
            normalized_reward = normalized_reward * self.temp
            return normalized_reward, False
    
    def check_if_obstacles_hit(self, state):
        for obs in self.obstacles:
            if np.linalg.norm(obs.center - state) < (obs.radius+self.robot_rad):
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = np.array([self.x_init, self.y_init])
        return torch.Tensor(self.get_state(self.position))


    # This is the part for visiualization, not urgent
    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass