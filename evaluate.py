# evaluate.py

import time
import gymnasium as gym
from stable_baselines3 import PPO
import torch

def main():
    # 1) Modell laden (auf CPU)
    model_path = "./lunarlander_sb3/models/ppo_lunarlander_final.zip"
    model = PPO.load(
        model_path, 
        device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # 2) Neue Env erzeugen
    env = gym.make("LunarLander-v3", render_mode="human")

    # 3) Evaluation über N Episoden
    episodes = 2
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0

        print(f"\n=== Episode {ep} ===")
        while not (terminated or truncated):
            # 4) Rendern
            env.render()

            # 5) Aktion vorhersagen
            action, _states = model.predict(obs, deterministic=True)

            # 6) Env-Schritt
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # 7) Debug-Info
            print(f"Step: pos=({obs[0]:.2f},{obs[1]:.2f}), reward={reward:.2f}")

            # kurz warten für flüssigere Darstellung
            time.sleep(0.02)

        print(f"Episode {ep} beendet – kumulativer Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
