import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")


episodes = 1000
# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for ep in range(episodes):

    total_reward = 0

    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
    # print(f"Step info: x={observation[0]:.2f}, y={observation[1]:.2f}, reward={reward:.2f}, done={terminated}")
    print(f"Episode {ep + 1}: Reward = {total_reward}")
env.close()