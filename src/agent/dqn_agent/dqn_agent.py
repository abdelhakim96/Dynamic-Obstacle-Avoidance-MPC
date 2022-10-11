import numpy as np
import torch
import torch.optim as optim
from utils import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent(object):

    def __init__(self, Q, Q_target, num_actions, env_name, gamma=0.99, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4,
                 history_length=0):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.env_name = env_name

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # Add Replay Buffer
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        # Sample Next Batch
        bs, ba, bn, br, bd = self.replay_buffer.next_batch(batch_size=self.batch_size)
        bs = torch.Tensor(bs).float().cuda()
        ba = torch.Tensor(ba).long().cuda()
        bn = torch.Tensor(bn).float().cuda()
        br = torch.Tensor(br).cuda()
        bd = torch.Tensor(bd).cuda()

        if self.env_name == 'CarRacing-v0':
            bs = bs.permute(0, 3, 1, 2)
            bn = bn.permute(0, 3, 1, 2)

        # TD target(Bellman Eq) and Loss
        q = self.Q(bs).gather(1, ba.unsqueeze(-1)).squeeze(-1)
        new_acc = br + self.gamma * self.Q_target(bn).max(1)[0]
        new_acc = new_acc * (1 - bd)
        loss = self.loss_function(q, new_acc)
        # Update NN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Soft update for TN
        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            if self.env_name == "CarRacing-v0":
                state = state.transpose(2, 0, 1)
                if len(state.shape) == 3:
                    state = state.reshape(1, 1, 96, 96)
            action_id = torch.argmax(self.Q(torch.Tensor(state).cuda())).item()
        else:
            action_id = np.random.randint(self.num_actions)

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
