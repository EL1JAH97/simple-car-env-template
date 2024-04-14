import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from simple_driving.envs.simple_driving_env import SimpleDrivingEnv  # Ensure you import your specific environment
from model import Network, DQN_Solver

# Hyperparameters
EPISODES = 2500
REPLAY_START_SIZE = 10000
BATCH_SIZE = 32

# Seed setup for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Initialize environment
env = SimpleDrivingEnv(isDiscrete=True, renders=False)  # Initialize with appropriate arguments
env.action_space.seed(0)  # Assuming your environment has a similar method

# Initialize DQN agent
input_dims = env.observation_space.shape  # This needs to match your environment's specifics
n_actions = env.action_space.n
agent = DQN_Solver(input_dims=input_dims, n_actions=n_actions)

episode_rewards = []
episode_batch_score = 0

# Training loop
for i in range(EPISODES):
    state = env.reset()
    episode_reward = 0

    while True:
        action = agent.choose_action(state)
        state_, reward, done, _ = env.step(action)
        agent.memory.store_transition(state, action, reward, state_, done)

        if agent.memory.mem_counter > REPLAY_START_SIZE:
            agent.learn()

        state = state_
        episode_reward += reward
        print(f"Current Episode {i}: Current Reward: {episode_reward}")
        if done:
            if agent.memory.mem_counter > REPLAY_START_SIZE:
                print(f"Episode {i}: Reward: {episode_reward}")
            break

    episode_rewards.append(episode_reward)

    # Save and print every 100 episodes
    if i % 100 == 0 and i != 0:
        torch.save(agent.policy_network.state_dict(), 'policy_network.pth')
        print(f"Average reward at episode {i}: {np.mean(episode_rewards[-100:])}")

# Plotting results
plt.plot(np.arange(len(episode_rewards)), episode_rewards)
plt.title('Reward Trends Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()


