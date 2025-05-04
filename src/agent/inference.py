import pathlib
import argparse

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from src.agent.utils.summary import log_summary
from src.env.utils.env import has_damage_been_done, get_damage_diff_percent

n_rerolls = 3
timesteps=100000
n_log_interval=10000

experiment_dir = "experiments"
model_path = "model.zip"

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument(
    "--model_path",
    default=model_path,
    type=str,
    help="The path to use for saving the trained sb3 model after training is complete. Saved model can be used later "
         "to resume training. Extension will be set to .zip",
)
parser.add_argument(
    "--timesteps",
    default=timesteps,
    type=int,
    help="The number of environment steps to train for, default is 1_000_000. If resuming from a saved model, "
         "it will continue training for this amount of steps from the saved state without counting previously trained "
         "steps",
)
parser.add_argument(
    "--render",
    default=False,
    action="store_true",
    help="render the results",
)

args, extras = parser.parse_known_args()

env = gym.make("Roller-v1", render_mode="human")
check_env(env)

path_zip = pathlib.Path(args.model_path)

model = PPO.load(path_zip, env=env, tensorboard_log=experiment_dir)

obs, info = env.reset()
if args.render:
    env.render()

print("\n======== Starting inference ========")
print("Steps", timesteps, "| Rerolls / turn", n_rerolls)

diffs = []
damage_dealt = []
damage_taken = []
wins = 0
losses = 0
rolls = 0
hands = []

for i in range(args.timesteps):
    if i % n_log_interval == 0:
        print("Step", i)

    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        rolls = info["rolls"]

        if info["player_won"]:
            wins += 1
        elif not info["player_won"]:
            losses += 1

    diff = 0
    if has_damage_been_done(obs["damage_done"]):
        diff = get_damage_diff_percent(
            obs["damage_done"],
            obs["player"][0],
            obs["enemy"][0]
        )
        diffs.append(round(diff, 2))

        damage_taken.append(obs["damage_done"][0])
        damage_dealt.append(obs["damage_done"][1])

    if terminated or truncated:
        hands.append(info["hands"])
        obs, info = env.reset()

    if args.render:
        env.render()

env.close()

print("\n======== Inference finished ========")
log_summary(wins, losses, diffs, damage_dealt, damage_taken, rolls, hands)
