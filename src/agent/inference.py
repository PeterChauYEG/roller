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

def log_win_losses(w, l):
    print("\n======== Summary of wins - losses ========")
    print("Wins: ", w)
    print("Losses: ", l)
    if w + l != 0:
        print("Winrate: ", round(w / (w + l) * 100, 2), "%")
    else:
        print("No games completed")

def log_stats(label, data):
    data = np.array(data)

    best = data.max()
    worst = data.min()
    median = np.median(data)
    mean = data.mean()
    mean = round(mean, 2)

    std = np.std(data)
    std = round(std, 2)

    print("\n======== Summary of {0} ========".format(label))
    print("Best: ", best)
    print("Worst: ", worst)
    print("Mean: ", mean)
    print("Median: ", median)
    print("Std: ", std)

    return mean, median, best, worst

def plot_histogram(title, xlabel, ylabel, mean, median, best, worst, data):
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.hist(data, bins=20, color="blue", alpha=0.7)
    plt.axvline(mean, color="red", linestyle="dashed", linewidth=1)
    plt.axvline(median, color="orange", linestyle="dashed", linewidth=1)
    plt.axvline(best, color="green", linestyle="dashed", linewidth=1)
    plt.axvline(worst, color="purple", linestyle="dashed", linewidth=1)
    plt.legend({"Mean": mean, "Median": median, "Best": best, "Worst": worst})
    plt.show()

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

        damage_dealt.append(obs["damage_done"][0])
        damage_taken.append(obs["damage_done"][1])

    if terminated or truncated:
        obs, info = env.reset()

    if args.render:
        env.render()

env.close()

print("\n======== Inference finished ========")

log_win_losses(wins, losses)

diff_mean, diff_median, diff_best, diff_worst = log_stats("Difference of damage dealt vs taken (%)", diffs)
damage_dealt_mean, damage_dealt_median, damage_dealt_best, damage_dealt_worst = log_stats("Damage dealt (%)", damage_dealt)
damage_taken_mean, damage_taken_median, damage_taken_best, damage_taken_worst = log_stats("Damage taken", damage_taken)

plot_histogram("Difference of damage dealt vs taken (%)", "Difference (%)", "Frequency", diff_mean, diff_median, diff_best, diff_worst, diffs)
plot_histogram("Damage dealt (%)", "Damage dealt (%)", "Frequency", damage_dealt_mean, damage_dealt_median, damage_dealt_best, damage_dealt_worst, damage_dealt)
plot_histogram("Damage taken (%)", "Damage taken (%)", "Frequency", damage_taken_mean, damage_taken_median, damage_taken_best, damage_taken_worst, damage_taken)
