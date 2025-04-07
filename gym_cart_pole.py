"""
    Based on: https://www.gymlibrary.dev/environments/classic_control/cart_pole/
    The goal of the CartPole environment is to balance the pole on the cart by moving the cart left or right.
    The environment is considered solved if the pole is balanced for 200 time steps.
    The environment is considered failed if the pole falls down or the cart moves out of bounds.
    The CartPole environment is considered solved if the average reward is greater than or equal to 195.0 over 100 consecutive episodes.
    The average reward is calculated as the sum of rewards in 100 consecutive episodes divided by 100.

    The code will open a window and you will see the cartpole environment in action.
    The cartpole environment will be rendered for 200 time steps.
    The agent will choose an action based on the custom agent's policy.
    The agent will take the chosen action and the environment will return the next state, reward, and done flag.

    The CartPole environment has the following 4 observations:
        1. Cart Position
        2. Cart Velocity
        3. Pole Angle
        4. Pole Angular Velocity

    The CartPole environment has 2 actions:
        1. Move the cart to the left - Action 0
        2. Move the cart to the right - Action 1

"""
import gym
import numpy as np
import matplotlib.pyplot as plt

# Define bool8 manually if missing
if not hasattr(np, 'bool8'):
    np.bool8 = bool

class CustomAgent:
    def __init__(self, observation_space):
        self.observation_space = observation_space
        self.weights = np.random.uniform(-1, 1, size=self.observation_space.shape)

    def get_action(self, observation):
        observation_weight_product = np.dot(observation, self.weights)
        return 1 if observation_weight_product >= 0 else 0
    def get_action(self, observation):
        observation_weight_product = np.dot(observation, self.weights)
        return 1 if observation_weight_product >= 0 else 0

def run_episode(env, agent):
    observation, info = env.reset(seed=42)
    total_reward = 0

    for i in range(200):
        action = agent.get_action(observation)
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        observation = next_state
        if done or truncated:
            break

    return total_reward

def random_search(env, num_samples=100):
    best_total_reward = -float('inf')
    best_weights = None

    agent = CustomAgent(env.observation_space)
    agent.weights = np.random.uniform(-1, 1, size=agent.observation_space.shape)

    for i in range(num_samples):
        agent.weights = np.random.uniform(-1, 1, size=agent.observation_space.shape)
        total_reward = run_episode(env, agent)
        print(f"Sample {i + 1}/{num_samples}: Total Reward = {total_reward}")
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_weights = agent.weights.copy()

    return best_weights, best_total_reward

def evaluate_random_search(env, num_searches=100, max_episodes=1000):
    episodes_to_solve = []

    for i in range(num_searches):
        best_weights,best_total_reward = random_search(env)
        agent = CustomAgent(env.observation_space)
        agent.weights = best_weights
        print(f"Best weights from search {i + 1}: {best_weights}")
        print(f"Best total reward from search {i + 1}: {best_total_reward}")
        for episode in range(max_episodes):
            total_reward = run_episode(env, agent)
            if total_reward == 200:
                episodes_to_solve.append(episode + 1)
                break
        else:
            episodes_to_solve.append(max_episodes)

    return episodes_to_solve

# Load CartPole's environment
env = gym.make('CartPole-v1', render_mode='human')
env.action_space.seed(42)

# Evaluate the random search scheme
episodes_to_solve = evaluate_random_search(env)

# Plot histogram
plt.hist(episodes_to_solve, bins=30, edgecolor='black')
plt.xlabel('Number of Episodes')
plt.ylabel('Frequency')
plt.title('Histogram of Episodes Required to Reach Score 200')
plt.show()

# Report average number of episodes
average_episodes = np.mean(episodes_to_solve)
print(f"Average number of episodes required to reach score 200: {average_episodes}")

env.close()