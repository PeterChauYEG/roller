import src.env

import argparse
import os
import pathlib
from typing import Callable

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

experiment_dir = "experiments"

save_freq = 200000
total_timesteps = 100
save_model_path = "model.zip"
experiment_name = "experiment"
base_learning_rate = 0.0003
batch_size = 64

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument(
    "--save_checkpoint_frequency",
    default=save_freq,
    type=int,
    help=(
        "If set, will save checkpoints every 'frequency' environment steps. "
        "Requires a unique --experiment_name or --experiment_dir for each run. "
        "Does not need --save_model_path to be set. "
    ),
)
parser.add_argument(
    "--linear_lr_schedule",
    default=False,
    action="store_true",
    help="Use a linear LR schedule for training. If set, learning rate will decrease until it reaches 0 at "
    "--timesteps"
    "value. Note: On resuming training, the schedule will reset. If disabled, constant LR will be used.",
)
parser.add_argument(
    "--experiment_name",
    default=experiment_name,
    type=str,
    help="The name of the experiment, which will be displayed in tensorboard and "
    "for checkpoint directory and name (if enabled).",
)
parser.add_argument(
    "--save_model_path",
    default=save_model_path,
    type=str,
    help="The path to use for saving the trained sb3 model after training is complete. Saved model can be used later "
    "to resume training. Extension will be set to .zip",
)
parser.add_argument(
    "--timesteps",
    default=total_timesteps,
    type=int,
    help="The number of environment steps to train for, default is 1_000_000. If resuming from a saved model, "
    "it will continue training for this amount of steps from the saved state without counting previously trained "
    "steps",
)
parser.add_argument(
    "--batch_size",
    default=batch_size,
    type=int,
    help="Size of batch and n_steps for PPO training",
)

args, extras = parser.parse_known_args()

# paths
path_checkpoint = os.path.join(experiment_dir, args.experiment_name + "_checkpoints")
abs_path_checkpoint = os.path.abspath(path_checkpoint)


# LR
# LR schedule code snippet from:
# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


learning_rate = (
    base_learning_rate
    if not args.linear_lr_schedule
    else linear_schedule(base_learning_rate)
)

env = gym.make("Roller-v1", render_mode="human")
check_env(env)

model = PPO(
    "MultiInputPolicy",
    env,
    ent_coef=0.0001,
    verbose=2,
    n_steps=args.batch_size,
    batch_size=args.batch_size,
    tensorboard_log=experiment_dir,
    learning_rate=learning_rate,
)

checkpoint_callback = CheckpointCallback(
    save_freq=args.save_checkpoint_frequency,
    save_path=path_checkpoint,
    name_prefix=args.experiment_name,
)

learn_arguments = dict(total_timesteps=args.timesteps, callback=checkpoint_callback)

model.learn(**learn_arguments)

zip_save_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
model.save(zip_save_path)

env.close()
