import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env.env_utils import has_damage_been_done, get_damage_diff_percent
from src.env.game import Game, WinnerType
from src.env.env_constants import N_DICES, N_MAX_ROLLS, N_DICE_FACES, N_MAX_FACE_VALUE, N_TRAITS, \
    N_ACTIONS, N_DICE_TYPES, DAMAGE_REWARD_MULTIPLIER, LOSE_REWARD, WIN_REWARD, MAX_PLAYER_ATTACK, MAX_ENEMY_ATTACK, \
    MAX_ENEMY_DEFENSE, MAX_PLAYER_DEFENSE

from src.env.render_utils import calculate_traits, TRAITS_HEADERS, calculate_info, INFO_HEADERS, calculate_action, \
    ROLL_HEADERS, calculate_roll_results, calculate_units, UNIT_HEADERS, calculate_dice_faces, DICES_HEADERS, \
    render_table

class RollerEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # for render
        self.action = None
        self.obs = None
        self.reward = None
        self.last_roll_results_totals = 0
        self.hand = 0
        self.rolls = 0

        self.action_space = spaces.MultiBinary(N_ACTIONS)
        self.observation_space = spaces.Dict({
            "dice_faces": spaces.Box(
                low=0,
                high=N_MAX_FACE_VALUE,
                shape=(N_DICES, N_DICE_FACES, 2),
                dtype=np.int16
            ),
            "dice_types": spaces.Box(
                low=0,
                high=N_DICE_TYPES,
                shape=(N_DICES,),
                dtype=np.int16
            ),
            "roll_results": spaces.Box(
                low=0,
                high=N_MAX_FACE_VALUE,
                shape=(N_DICES,2),
                dtype=np.int16
            ),
            "n_remaining_rolls": spaces.Box(
                low=0,
                high=N_MAX_ROLLS,
                shape=(1,),
                dtype=np.int16
            ),
            "enemy": spaces.Box(
                low=0,
                high=MAX_PLAYER_ATTACK,
                shape=(4,),
                dtype=np.int16
            ),
            "player": spaces.Box(
                low=0,
                high=MAX_PLAYER_ATTACK,
                shape=(4,),
                dtype=np.int16
            ),
            "damage_done": spaces.Box(
                low=0,
                high=MAX_PLAYER_ATTACK,
                shape=(2,),
                dtype=np.int16
            ),
            "traits": spaces.Box(
                low=0,
                high=100,
                shape=(N_TRAITS, N_DICES, 4),
                dtype=np.int16
            ),
        })

        self.game = Game()

    def step(self, action):
        self.action = action
        truncated = False

        info = {
            "player_won": False,
            "hands": 0
        }

        obs, won, did_roll = self.game.player_turn(action)
        self.obs = obs

        if not did_roll:
            self.hand += 1
        else:
            self.rolls += 1

        reward, game_over, won = self.calculate_reward(obs, won)
        self.reward = reward

        info["rolls"] = self.rolls

        if game_over:
            self.last_roll_results_totals = 0
            info["player_won"] = won
            info["hands"] = self.hand + 1

            return obs, reward, True, truncated, info

        self.last_roll_results_totals = obs["player"][2] + obs["player"][3]

        return obs, reward, False, truncated, info

    def calculate_reward(self, obs, won):
        reward = 0

        if won != WinnerType.NONE:
            if won == WinnerType.PLAYER:
                reward = WIN_REWARD
            elif won == WinnerType.ENEMY:
                reward = -LOSE_REWARD

            return reward, True, won == WinnerType.PLAYER,

        # calc the value of the roll
        if self.last_roll_results_totals > 0:
            player_total = obs["player"][2] + obs["player"][3]
            diff = player_total - self.last_roll_results_totals
            reward = diff

        # calc the difference of damage dealt - damage taken as a number [0 - 100]
        if has_damage_been_done(obs["damage_done"]):
            diff = get_damage_diff_percent(
                obs["damage_done"],
                obs["player"][0],
                obs["enemy"][0]
            )
            diff = diff * DAMAGE_REWARD_MULTIPLIER
            reward += diff

        reward = float(round(reward, 2))

        return reward, False, False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.action = None
        self.hand = 0
        self.rolls = 0
        self.reward = 0
        info = {}

        obs, won, did_roll = self.game.reset()
        self.obs = obs
        self.last_roll_results_totals = obs["player"][2] + obs["player"][3]

        return obs, info

    def render(self):
        print("================== hand {} ======================".format(self.hand))

        if self.obs is not None:
            info = calculate_info(
                self.obs["damage_done"],
                self.reward,
                self.obs["n_remaining_rolls"]
            )
            units = calculate_units(
                self.obs["player"],
                self.obs["enemy"],
            )
            print("\n> Info")
            render_table(INFO_HEADERS, info)
            render_table(UNIT_HEADERS, units)

        if self.action is not None:
            action = calculate_action(self.action)
            print("\n> Rerolling dices")
            render_table(ROLL_HEADERS, action)

        if self.obs is not None:
            roll_results = calculate_roll_results(self.obs["roll_results"])

            dice_faces = calculate_dice_faces(self.obs["dice_faces"], self.obs["dice_types"])
            traits = calculate_traits(self.obs["traits"])

            print("\n> Roll results")
            render_table(ROLL_HEADERS, roll_results)

            print("\n> Lookup tables")
            render_table(DICES_HEADERS, dice_faces)
            render_table(TRAITS_HEADERS, traits)

    def close(self):
        pass
