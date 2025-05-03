from tabulate import tabulate

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.game import Game, N_ACTIONS, N_DICES, N_DICE_FACES, N_MAX_FACE_VALUE, N_MIN_FACE_VALUE, N_DICE_TYPES, N_MAX_ROLLS, MIN_ROLL, MAX_ROLL, MAX_ENEMY_HP, WinnerType

class RollerEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # render
        self.action = None
        self.obs = None
        self.won = None
        self.reward = None

        self.action_space = spaces.MultiBinary(N_ACTIONS)
        self.observation_space = spaces.Dict({
            "dice_faces": spaces.Box(
                low=N_MIN_FACE_VALUE,
                high=N_MAX_FACE_VALUE,
                shape=(N_DICES, N_DICE_FACES),
                dtype=np.int8
            ),
            "dice_types": spaces.Box(
                low=0,
                high=N_DICE_TYPES,
                shape=(N_DICES,),
                dtype=np.int8
            ),
            "roll_results": spaces.Box(
                low=N_MIN_FACE_VALUE,
                high=N_MAX_FACE_VALUE,
                shape=(N_DICES,),
                dtype=np.int8
            ),
            "roll_results_totals": spaces.Box(
                low=MIN_ROLL,
                high=MAX_ROLL,
                shape=(2,),
                dtype=np.float16
            ),
            "n_remaining_rolls": spaces.Box(
                low=0,
                high=N_MAX_ROLLS,
                shape=(1,),
                dtype=np.int8
            ),
            "enemy": spaces.Box(
                low=0,
                high=MAX_ENEMY_HP,
                shape=(4,),
                dtype=np.int16
            ),
            "player": spaces.Box(
                low=0,
                high=MAX_ENEMY_HP,
                shape=(2,),
                dtype=np.int16
            ),
            "damage_done": spaces.Box(
                low=0,
                high=MAX_ENEMY_HP,
                shape=(2,),
                dtype=np.int16
            )
        })

        self.game = Game()
        self.last_roll_results_totals = [0., 0.]

    def step(self, action):
        self.action = action

        info = {
            "player_won": False,
        }

        truncated = False

        obs, won = self.game.player_turn(action)
        self.obs = obs
        self.won = won

        reward = 0

        if won != WinnerType.NONE:
            if won == WinnerType.PLAYER:
                info["player_won"] = True
                reward += 1000
            elif won == WinnerType.ENEMY:
                info["player_won"] = False
                reward -= 1000

            print("Game over. Player won: ", won == WinnerType.PLAYER)
            self.last_roll_results_totals = [0., 0.]

            self.reward = reward
            return obs, reward, True, truncated, info

        # calc the value of the roll
        diff = (obs["roll_results_totals"] - self.last_roll_results_totals).sum()
        reward = float(diff)

        self.last_roll_results_totals = obs["roll_results_totals"]

        # calc the difference of damage dealt - damage taken as a number [0 - 100]
        if obs["damage_done"][0] > 0 or obs["damage_done"][1] > 0:
            damage_to_player = obs["damage_done"][0] / obs["player"][0] * 100
            damage_to_enemy = obs["damage_done"][1] / obs["enemy"][0] * 100
            diff = (damage_to_enemy - damage_to_player) * 10
            reward += diff

        reward = round(float(reward), 2)
        self.reward = reward
        return obs, reward, False, truncated, info

    def reset(self, seed=None, options=None):
        self.action = None

        super().reset(seed=seed)
        info = {}

        obs, won = self.game.reset()
        self.obs = obs
        self.won = won
        self.reward = 0
        self.last_roll_results_totals = obs["roll_results_totals"]

        return obs, info

    def render(self):
        roll_headers = ["Dice 1", "Dice 2", "Dice 3", "Dice 4", "Dice 5", "Dice 6"]
        unit_headers = ["Name", "HP", "Max HP", "Attack", "Defense"]
        info_headers = ["Reward", "Damage Taken", "Damage Dealt", "Remaining Rolls"]
        dices_headers = ["Dice", "Type", "Face 1", "Face 2", "Face 3", "Face 4", "Face 5", "Face 6"]

        print("=========================================")

        if self.obs is not None:
            reward = self.reward if self.reward is not None else 0
            damage_taken = self.obs["damage_done"][0]
            damage_dealt = self.obs["damage_done"][1]

            info = [
                [
                    reward,
                    damage_taken,
                    damage_dealt,
                    self.obs["n_remaining_rolls"][0]
                ]
            ]

            print(tabulate(info, info_headers, tablefmt="simple_outline"))

        if self.action is not None:
            action = [self.action]
            print("Rerolling dices")
            print(tabulate(action, roll_headers, tablefmt="simple_outline"))

        if self.obs is not None:
            roll_results = [self.obs["roll_results"]]
            print("Roll results")
            print(tabulate(roll_results, roll_headers, tablefmt="simple_outline"))

            units = [
                [
                    "Player",
                    self.obs["player"][1],
                    self.obs["player"][0],
                    self.obs["roll_results_totals"][0],
                    self.obs["roll_results_totals"][1],
                ],
                [
                    "Enemy",
                    self.obs["enemy"][1],
                    self.obs["enemy"][0],
                    self.obs["enemy"][2],
                    self.obs["enemy"][3],
                ]
            ]
            print(tabulate(units, unit_headers, tablefmt="simple_outline"))

            dice_faces = [
                [
                    f"Dice {i + 1}",
                    "Attack" if self.obs["dice_types"][i] == 0 else "Defense",
                    *self.obs["dice_faces"][i]
                ]
                for i in range(N_DICES)
            ]

            print(tabulate(dice_faces, dices_headers, tablefmt="simple_outline"))

    def close(self):
        pass
