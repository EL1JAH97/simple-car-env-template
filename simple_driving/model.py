import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math

# Configuration for the neural network
LEARNING_RATE = 0.00025
FC1_DIMS = 128
FC2_DIMS = 128
MEM_SIZE = 50000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 4 * MEM_SIZE
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_START_SIZE = 10000

class Network(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(*input_dims, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self, input_dims, n_actions):
        self.learn_step_counter = 0  # Added to track the number of steps taken (for epsilon decay)
        self.action_space = [i for i in range(n_actions)]
        self.memory = ReplayBuffer(MEM_SIZE, input_shape=input_dims)
        self.policy_network = Network(input_dims=input_dims, n_actions=n_actions)
        self.target_network = Network(input_dims=input_dims, n_actions=n_actions)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.epsilon = EPS_START

    def choose_action(self, observation):
        if self.memory.mem_counter > REPLAY_START_SIZE:
            self.epsilon = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.learn_step_counter / EPS_DECAY)
        else:
            self.epsilon = EPS_START

        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float32).to(self.policy_network.device)
            actions = self.policy_network(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_counter < BATCH_SIZE:
            return

        self.policy_network.optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.memory.sample_buffer(BATCH_SIZE)
        states = torch.tensor(states).to(self.policy_network.device)
        actions = torch.tensor(actions).to(self.policy_network.device)
        rewards = torch.tensor(rewards).to(self.policy_network.device)
        states_ = torch.tensor(states_).to(self.policy_network.device)
        dones = torch.tensor(dones).to(self.policy_network.device)

        indices = np.arange(BATCH_SIZE)
        q_pred = self.policy_network(states)[indices, actions]
        q_next = self.target_network(states_).max(dim=1)[0]
        q_next[dones] = 0.0

        q_target = rewards + GAMMA * q_next

        loss = self.policy_network.loss(q_pred, q_target)
        loss.backward()
        self.policy_network.optimizer.step()
        self.learn_step_counter += 1  # Increment the step counter after each learning step
