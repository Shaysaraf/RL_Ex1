import gym
import numpy as np
import matplotlib.pyplot as plt

# Define bool8 manually if missing
if not hasattr(np, 'bool8'):
    np.bool8 = bool

# Custom agent using a simple linear policy
class CustomAgent:
    def __init__(self, observation_space):
        self.observation_space = observation_space
        self.weights = np.random.uniform(-1, 1, size=self.observation_space.shape)

    def get_action(self, observation):
        return 1 if np.dot(observation, self.weights) >= 0 else 0

def run_episode(env, agent, render=False):
    observation, info = env.reset(seed=42)
    total_reward = 0

    for _ in range(200):
        if render:
            env.render()
        action = agent.get_action(observation)
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    return total_reward

def random_search(env, num_samples=100):
    best_total_reward = -float('inf')
    best_weights = None
    agent = CustomAgent(env.observation_space)

    for i in range(num_samples):
        agent.weights = np.random.uniform(-1, 1, size=agent.observation_space.shape)
        total_reward = run_episode(env, agent)
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_weights = agent.weights.copy()

    return best_weights, best_total_reward

def evaluate_random_search(env, num_searches=100, max_episodes=1000):
    episodes_to_solve = []
    failures = 0

    for i in range(num_searches):
        best_weights, best_reward = random_search(env)
        agent = CustomAgent(env.observation_space)
        agent.weights = best_weights

        for episode in range(max_episodes):
            total_reward = run_episode(env, agent)
            if total_reward == 200:
                episodes_to_solve.append(episode + 1)
                break
        else:
            episodes_to_solve.append(max_episodes)
            failures += 1

    return episodes_to_solve, failures

# Main script
env = gym.make('CartPole-v1', render_mode='human')
env.action_space.seed(42)

# Run evaluation
episodes_to_solve, failures = evaluate_random_search(env, num_searches=100)

# Plot histogram
plt.figure(figsize=(10,6))
plt.hist(episodes_to_solve, bins=30, edgecolor='black')
plt.xlabel('Episodes Required to Reach Score 200')
plt.ylabel('Frequency')
plt.title('Histogram of Episodes Required (Random Search Agent)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Summary stats
average_episodes = np.mean(episodes_to_solve)
print(f" Average episodes to reach score 200: {average_episodes:.2f}")
print(f"Failed to reach score 200 in 1000 episodes: {failures} out of 100 searches")

env.close()
