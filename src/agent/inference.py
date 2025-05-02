import src.env

import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

n_rerolls = 3
timesteps=100000
n_log_interval=10000
debug=False

experiment_dir = "experiments"
model_path = "model.zip"

env = gym.make("Roller-v1", render_mode="human")
check_env(env)

path_zip = pathlib.Path(model_path)

model = PPO.load(path_zip, env=env, tensorboard_log=experiment_dir)

obs, info = env.reset()

max_roll = obs["dice_faces"].max(axis=1).sum()

print("\n======== Starting inference ========")
print("Steps", timesteps, "| Rerolls / turn", n_rerolls)

if debug:
    print("Max roll", max_roll)
    print("Dices\n", obs["dice_faces"])

diffs = []
wins = 0
losses = 0

for i in range(timesteps):
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
        damage_to_player = obs["damage_done"][0] / obs["player"][0] * 100
        damage_to_enemy = obs["damage_done"][1] / obs["enemy"][0] * 100
        diff = round(damage_to_enemy - damage_to_player, 2)
        diffs.append(diff)

    if debug:
        print(i, "| Action", action, "| Results", obs["roll_results"], "| Total", obs["roll_results_totals"], "| Remaining rolls", obs["n_remaining_rolls"][0], "| Reward", reward, "| Player HP", obs["player"][1], "| Enemy HP", obs["enemy"][1])

    if terminated or truncated:
        obs, info = env.reset()

        max_roll = obs["dice_faces"].max(axis=1).sum()

        if debug:
            print("\n======== Resetting environment ========\n", "Max roll", max_roll)
            print("Player Max HP", obs["player"][0], "| Enemy Max HP", obs["enemy"][0])
            print("Dices\n", obs["dice_faces"])

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
