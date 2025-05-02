import src.env

import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

n_rerolls = 3
timesteps=1000000
n_log_interval=10000
debug=False

experiment_dir = "experiments"
model_path = "model.zip"

env = gym.make("Roller-v1", render_mode="human")
check_env(env)

path_zip = pathlib.Path(model_path)

model = PPO.load(path_zip, env=env, tensorboard_log=experiment_dir)

obs, info = env.reset()

max_roll = obs["dices"].max(axis=1).sum()

print("\n======== Starting inference ========")
print("Steps", timesteps, "| Rerolls / turn", n_rerolls)

if debug:
    print("Max roll", max_roll)
    print("Dices\n", obs["dices"])

p_diffs = []

for i in range(timesteps):
    if i % n_log_interval == 0:
        print("Step", i)

    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # round to 2 decimal places
    p_diff = (obs["roll_results_totals"][0] - max_roll) / max_roll * 100
    p_diff = round(p_diff, 2)

    if debug:
        print(i, "| Action", action, "| Roll results", obs["roll_results"], "| Roll total", obs["roll_results_totals"][0], "| Remaining rolls", obs["n_remaining_rolls"][0], "| Reward", reward, "| % diff", p_diff)

    if terminated or truncated:
        p_diffs.append(p_diff)

        obs, info = env.reset()

        max_roll = obs["dices"].max(axis=1).sum()

        if debug:
            print("\n======== Resetting environment ========\n", "Max roll", max_roll)
            print("Dices\n", obs["dices"])

env.close()

# turn p_diffs into a np array
p_diffs = np.array(p_diffs)

best = p_diffs.max()
worst = p_diffs.min()
median = np.median(p_diffs)
mean = p_diffs.mean()
mean = round(mean, 2)

std = np.std(p_diffs)
std = round(std, 2)

print("\n======== Summary ========")
print("Best % diff: ", best, "%")
print("Worst % diff: ", worst, "%")
print("Mean % diff: ", mean, "%")
print("Median % diff: ", median, "%")
print("Std % diff: ", std, "%")

# then create a histogram

# set the style
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
plt.title("Distribution of % difference from max roll")
plt.xlabel("% difference")
plt.ylabel("Frequency")
plt.hist(p_diffs, bins=20, color="blue", alpha=0.7)
plt.axvline(mean, color="red", linestyle="dashed", linewidth=1)
plt.axvline(median, color="orange", linestyle="dashed", linewidth=1)
plt.axvline(best, color="green", linestyle="dashed", linewidth=1)
plt.axvline(worst, color="purple", linestyle="dashed", linewidth=1)
plt.legend({"Mean": mean, "Median": median, "Best": best, "Worst": worst})
plt.show()
