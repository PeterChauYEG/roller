import argparse
import pathlib

import gymnasium as gym

from src.agent.utils.summary import log_summary
from src.env.utils.env import get_damage_diff_percent, has_damage_been_done

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


n_rerolls = 3
timesteps = 100000
n_log_interval = 10000

experiment_dir = "experiments"
model_path = "model.zip"

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument(
    "--model_path",
    default=model_path,
    type=str,
    help="Used for saving the trained model",
)
parser.add_argument(
    "--timesteps",
    default=timesteps,
    type=int,
    help="The number of environment steps to train for, default is 1_000_000.",
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
print("Steps", args.timesteps, "| Rerolls / turn", n_rerolls)

diffs = []
damage_dealt = []
damage_taken = []
wins = 0
losses = 0
rolls = 0
hands = []
battles_won = []

for i in range(args.timesteps):
    if i % n_log_interval == 0:
        print("Step", i)

    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if info["player_won"]:
        wins += 1
    if terminated:
        battles_won.append(info["battles_won"])
        if not info["player_won"]:
            losses += 1

    diff = 0
    if has_damage_been_done(obs["damage_done"]):
        diff = get_damage_diff_percent(
            obs["damage_done"], obs["player"][0], obs["enemy"][0]
        )
        diffs.append(round(diff, 2))

        damage_taken.append(obs["damage_done"][0])
        damage_dealt.append(obs["damage_done"][1])

    if args.render:
        env.render()

    if terminated or truncated:
        hands.append(info["hands"])
        rolls += info["rolls"]

        obs, info = env.reset()

        if args.render:
            env.render()

env.close()

print("\n======== Inference finished ========")
log_summary(
    wins, losses, diffs, damage_dealt, damage_taken, rolls, hands, battles_won
)
