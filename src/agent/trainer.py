import src.env

import os
import pathlib

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

total_timesteps=1000000
save_freq=200000

experiment_dir = "experiments"
experiment_name = "experiment"
learning_rate = 0.0003
save_model_path = "model.zip"

path_checkpoint = os.path.join(experiment_dir, experiment_name + "_checkpoints")
abs_path_checkpoint = os.path.abspath(path_checkpoint)

# Prevent overwriting existing checkpoints when starting a new experiment if checkpoint saving is enabled
# if os.path.isdir(path_checkpoint):
#     raise RuntimeError(
#         abs_path_checkpoint + " folder already exists. "
#                               "Use a different --experiment_dir, or --experiment_name,"
#                               "or if previous checkpoints are not needed anymore, "
#                               "remove the folder containing the checkpoints. "
#     )

env = gym.make("Roller-v1", render_mode="human")
check_env(env)

model = PPO(
    "MultiInputPolicy",
    env,
    ent_coef=0.0001,
    verbose=2,
    n_steps=32,
    batch_size=32,
    tensorboard_log=experiment_dir,
    learning_rate=learning_rate,
)

checkpoint_callback = CheckpointCallback(
    save_freq=save_freq,
    save_path=path_checkpoint,
    name_prefix=experiment_name,
)

learn_arguments = dict(total_timesteps=total_timesteps, callback=checkpoint_callback)

model.learn(**learn_arguments)

zip_save_path = pathlib.Path(save_model_path).with_suffix(".zip")
model.save(zip_save_path)

env.close()
