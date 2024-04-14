import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from simple_driving.envs.simple_driving_env import SimpleDrivingEnv  # Ensure you import your specific environment
from model import Network, DQN_Solver

EPISODES = 3500
REPLAY_START_SIZE = 10000

# Seed setup for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Initialize environment
env = SimpleDrivingEnv(isDiscrete=True, renders=False)
env.action_space.seed(0)

# Initialize DQN agent
agent = DQN_Solver(env)

episode_rewards = []
episode_batch_score = 0

# Training loop
for i in range(EPISODES):
    state = env.reset()
    episode_reward = 0

    while True:
        action = agent.choose_action(state)
        state_, reward, done, _ = env.step(action)
        agent.memory.add(state, action, reward, state_, done)

        if agent.memory.mem_count > REPLAY_START_SIZE:
            agent.learn()

        state = state_
        episode_reward += reward
        print(f"Current Episode {i}: Current Reward: {episode_reward}")

        if done:
            break

    episode_rewards.append(episode_reward)

    if i % 100 == 0 and i > 0:
        # Saving the model locally, not to Google Collab
        torch.save(agent.policy_network.state_dict(), f'policy_network_{i}.pth')
        print(f"Average reward at episode {i}: {np.mean(episode_rewards[-100:])}")

# Plotting results
plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.title('Reward Trends Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

