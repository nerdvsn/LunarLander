# evaluate.py

from stable_baselines3 import PPO
import gymnasium as gym
import time

# Modell und Umgebung laden
model_path = "./lunarlander_sb3/models/ppo_lunarlander_final.zip"
model = PPO.load(model_path)
env = gym.make("LunarLander-v3", render_mode="human")

# Evaluation
episodes = 5
obs, _ = env.reset()
for ep in range(episodes):
    # obs, _ = env.reset()
    terminated = False
    total_reward = 0
    while not terminated:
        env.render()
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step info: x={obs[0]:.2f}, y={obs[1]:.2f}, reward={reward:.2f}, done={terminated}")
        time.sleep(0.02)  # für flüssigere Animation
    print(f"Episode {ep + 1}: Reward = {total_reward}")
env.close()
