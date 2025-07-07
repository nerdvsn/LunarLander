# train.py

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback


# GPU-Nutzung prüfen
import torch
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Ordnerstruktur
MODEL_DIR = "./lunarlander_sb3/models"
LOG_DIR = "./lunarlander_sb3/logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Umgebung erstellen
env = make_vec_env("LunarLander-v3", n_envs=1)
env = VecMonitor(env, filename=os.path.join(LOG_DIR, "monitor.csv"))

# Callback zum regelmäßigen Speichern
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=MODEL_DIR,
    name_prefix="ppo_lunarlander"
)

# Modell initialisieren
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Training starten
model.learn(
    total_timesteps=800_000,
    callback=checkpoint_callback
)

# Modell speichern
model.save(os.path.join(MODEL_DIR, "ppo_lunarlander_final"))
print("Training abgeschlossen und Modell gespeichert.")
