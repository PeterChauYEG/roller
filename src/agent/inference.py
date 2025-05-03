import src.env

import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

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
wins = 0
losses = 0

for i in range(args.timesteps):
    if i % n_log_interval == 0:
        print("Step", i)

    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        if info["player_won"]:
            wins += 1
        elif not info["player_won"]:
            losses += 1

    diff = 0
    if obs["damage_done"][0] > 0 or obs["damage_done"][1] > 0:
        damage_to_player = (obs["damage_done"][0] / obs["player"][0]) * 100
        damage_to_enemy = (obs["damage_done"][1] / obs["enemy"][0]) * 100
        diff = round(damage_to_enemy - damage_to_player, 2)
        diffs.append(diff)

    if terminated or truncated:
        obs, info = env.reset()

    if args.render:
        env.render()

env.close()

# turn p_diffs into a np array
diffs = np.array(diffs)

best = diffs.max()
worst = diffs.min()
median = np.median(diffs)
mean = diffs.mean()
mean = round(mean, 2)

std = np.std(diffs)
std = round(std, 2)

print("\n======== Summary of wins - losses ========")
print("Wins: ", wins)
print("Losses: ", losses)
if wins + losses != 0:
    print("Winrate: ", round(wins / (wins + losses) * 100, 2), "%")

print("\n======== Summary of damage dealt - damage taken ========")
print("Best: ", best, "%")
print("Worst: ", worst, "%")
print("Mean: ", mean, "%")
print("Median: ", median, "%")
print("Std: ", std, "%")

# then create a histogram

# set the style
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
plt.title("Distribution of % difference from max roll")
plt.xlabel("% difference")
plt.ylabel("Frequency")
plt.hist(diffs, bins=20, color="blue", alpha=0.7)
plt.axvline(mean, color="red", linestyle="dashed", linewidth=1)
plt.axvline(median, color="orange", linestyle="dashed", linewidth=1)
plt.axvline(best, color="green", linestyle="dashed", linewidth=1)
plt.axvline(worst, color="purple", linestyle="dashed", linewidth=1)
plt.legend({"Mean": mean, "Median": median, "Best": best, "Worst": worst})
plt.show()
