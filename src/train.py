import torch
import torch.nn as nn
from gym_examples.gym_examples.grid_world import GridWorld
import numpy as np
from utils.obstacle_generator import generate_random_obstacles, generate_regular_obstacle_grid
import argparse
from agent.ddpg_agent.ddpg_agent import DDPG
from models.world_specification import *

def train(env, agent: DDPG, num_episodes):
    for i in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            state = torch.flatten(state).unsqueeze(0)
            action = agent.calc_action(state)
            next_state, reward, done, _ = env.step(action)
            value_loss, policy_loss = agent.update_params(state, action, reward, done, next_state)
            state = next_state
            print(f"Epsiode = {i}, Action = {action}, Reward = {reward}, Value Loss = {value_loss}, Policy Loss = {policy_loss}")

def main(args):
    obstacles = generate_random_obstacles(args.seed)
    env = GridWorld(obstacles, X_MIN_ROBOT, Y_MIN_ROBOT, X_MAX_ROBOT, Y_MAX_ROBOT)

    # In order => Gamma, Tau, Number of Hidden Layers, Number of Input( Number of Obstacles + Robot), Number of Actions (next subgoal)
    hidden_layers = [128, 128]
    agent = DDPG(args.gamma, args.tau, hidden_layers, 3*(len(obstacles)+1), 2)
    train(env, agent, args.num_episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_obstacles', type=int, default=4)
    parser.add_argument('--obstacle_radius_max', type=float, default=0.8)
    parser.add_argument('--obstacle_radius_min', type=float, default=0.6)
    parser.add_argument('--robot_radius', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.99)
    parser.add_argument('--x_init', type=float, default=-2)
    parser.add_argument('--y_init', type=float, default=-2)
    parser.add_argument('--x_goal', type=float, default=2)
    parser.add_argument('--y_goal', type=float, default=2)
    parser.add_argument('--x_min', type=float, default=-3)
    parser.add_argument('--x_max', type=float, default=3)
    parser.add_argument('--y_min', type=float, default=-3)
    parser.add_argument('--y_max', type=float, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--num_test_episodes', type=int, default=100)
    parser.add_argument('--num_test_steps', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.95)
    args = parser.parse_args()
    main(args)