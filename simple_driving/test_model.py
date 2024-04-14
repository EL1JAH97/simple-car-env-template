import torch
import gym
from simple_driving.envs.simple_driving_env import SimpleDrivingEnv
from model import Network, DQN_Solver  # Make sure these are correctly imported

# Initialize the environment with rendering
env = SimpleDrivingEnv(isDiscrete=True, renders=True)

# Load the trained model
model_path = 'policy_network_2400.pth'  
model = Network(env)
model.load_state_dict(torch.load(model_path))
model.eval()

# Simulation loop
state = env.reset()
done = False
while not done:
    state_tensor = torch.tensor([state], dtype=torch.float32)  # Convert state to tensor
    with torch.no_grad():
        action = torch.argmax(model(state_tensor)).item()  # Predict action from the model
    state, reward, done, _ = env.step(action)  # Take action in the environment

env.close()  # Close the environment
