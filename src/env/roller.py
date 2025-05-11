import gymnasium as gym
from gymnasium import spaces

import numpy as np

from src.env.data.game import (
    BETTER_DAMAGE_REWARD,
    BETTER_ROLL_REWARD,
    LOSE_REWARD,
    MAX_ENEMY_ATTACK,
    MAX_ENEMY_DEFENSE,
    MAX_ENEMY_HP,
    MAX_PLAYER_ATTACK,
    MAX_PLAYER_DEFENSE,
    MAX_PLAYER_HP,
    MIN_ENEMY_ATTACK,
    MIN_ENEMY_DEFENSE,
    MIN_ENEMY_HP,
    MIN_PLAYER_HP,
    N_ACTIONS,
    N_DICES,
    N_DICE_FACES,
    N_MAX_FACE_VALUE,
    N_MAX_ROLLS,
    N_MIN_FACE_VALUE,
    N_TRAITS,
    WIN_REWARD,
    WORST_DAMAGE_REWARD,
    WORST_ROLL_REWARD,
)
from src.env.game import Game
from src.env.utils.env import (
    get_damage_diff_percent,
    get_number_of_trait_effects,
    has_damage_been_done,
)
from src.env.utils.render import (
    DICES_HEADERS,
    INFO_HEADERS,
    ROLL_HEADERS,
    TRAITS_HEADERS,
    UNIT_HEADERS,
    calculate_action,
    calculate_dice_faces,
    calculate_info,
    calculate_roll_results,
    calculate_traits,
    calculate_units,
    render_table,
)


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
        self.battles_won = 0

        self.action_space = spaces.MultiBinary(N_ACTIONS)

        trait_effects = get_number_of_trait_effects()

        self.observation_space = spaces.Dict(
            {
                "roll_result_traits": spaces.Box(
                    low=0, high=N_TRAITS, shape=(N_DICES,), dtype=np.int16
                ),
                "roll_result_values": spaces.Box(
                    low=N_MIN_FACE_VALUE,
                    high=N_MAX_FACE_VALUE,
                    shape=(N_DICES,),
                    dtype=np.int16,
                ),
                "damage_done": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([MAX_PLAYER_ATTACK, MAX_ENEMY_ATTACK]),
                    shape=(2,),
                    dtype=np.int16,
                ),
                "player": spaces.Box(
                    low=np.array(
                        [
                            MIN_PLAYER_HP,
                            0,
                            N_MIN_FACE_VALUE * 6,
                            N_MIN_FACE_VALUE * 6,
                        ]
                    ),
                    high=np.array(
                        [
                            MAX_PLAYER_HP + 2,
                            MAX_PLAYER_HP + 2,
                            MAX_PLAYER_ATTACK,
                            MAX_PLAYER_DEFENSE,
                        ]
                    ),
                    shape=(4,),
                    dtype=np.int16,
                ),
                "enemy": spaces.Box(
                    low=np.array(
                        [MIN_ENEMY_HP, 0, MIN_ENEMY_ATTACK, MIN_ENEMY_DEFENSE]
                    ),
                    high=np.array(
                        [
                            MAX_ENEMY_HP * 10,
                            MAX_ENEMY_HP * 10,
                            MAX_ENEMY_ATTACK,
                            MAX_ENEMY_DEFENSE,
                        ]
                    ),
                    shape=(4,),
                    dtype=np.int16,
                ),
                "n_remaining_rolls": spaces.Box(
                    low=0, high=N_MAX_ROLLS, shape=(1,), dtype=np.int16
                ),
                "traits": spaces.Box(
                    low=0, high=100, shape=(trait_effects * 4,), dtype=np.int16
                ),
                "all_dice_face_traits": spaces.Box(
                    low=0,
                    high=N_TRAITS,
                    shape=(N_DICES * N_DICE_FACES,),
                    dtype=np.int16,
                ),
                "all_dice_face_values": spaces.Box(
                    low=0,
                    high=N_MAX_FACE_VALUE,
                    shape=(N_DICES * N_DICE_FACES,),
                    dtype=np.int16,
                ),
            }
        )

        self.game = Game()

    def step(self, action):
        self.action = action
        truncated = False

        info = {
            "player_won": False,
            "hands": 0,
            "battles_won": 0,
        }

        obs, game_over, did_roll, hand_played, next_battle = (
            self.game.player_turn(action)
        )
        self.obs = obs

        if did_roll:
            self.rolls += 1

        if hand_played:
            self.hand += 1

        reward = self.calculate_reward(obs, game_over, next_battle)
        self.reward = reward

        info["rolls"] = self.rolls

        if next_battle:
            info["hands"] = self.hand
            info["player_won"] = next_battle
            self.battles_won += 1

        if game_over:
            self.last_roll_results_totals = 0
            info["hands"] = self.hand
            info["battles_won"] = self.battles_won
            self.battles_won = 0

            return obs, reward, True, truncated, info

        self.last_roll_results_totals = obs["player"][2] + obs["player"][3]

        return obs, reward, False, truncated, info

    def calculate_reward(self, obs, game_over, won):
        reward = 0

        if won:
            reward = WIN_REWARD
            return reward

        if game_over:
            if not won:
                reward = -LOSE_REWARD
                return reward

        # calc the value of the roll
        if self.last_roll_results_totals > 0:
            player_total = obs["player"][2] + obs["player"][3]
            diff = player_total - self.last_roll_results_totals

            if diff >= 0:
                reward += BETTER_ROLL_REWARD
            else:
                reward += WORST_ROLL_REWARD

        # calc the difference of damage dealt - damage taken as a number
        # [0 - 100]
        if has_damage_been_done(obs["damage_done"]):
            diff = get_damage_diff_percent(
                obs["damage_done"], obs["player"][0], obs["enemy"][0]
            )

            if diff >= 0:
                reward += diff * BETTER_DAMAGE_REWARD
            else:
                reward += diff * WORST_DAMAGE_REWARD

        reward = float(round(reward, 2))

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.action = None
        self.hand = 0
        self.rolls = 0
        self.reward = 0
        self.battles_won = 0
        self.last_roll_results_totals = 0
        info = {}

        obs, game_over, did_roll, hand_played, next_battle = self.game.reset()
        self.obs = obs
        self.last_roll_results_totals = obs["player"][2] + obs["player"][3]

        return obs, info

    def render(self):
        print(
            "================== hand {} ======================".format(
                self.hand
            )
        )

        if self.obs is not None:
            info = calculate_info(
                self.obs["damage_done"],
                self.reward,
                self.obs["n_remaining_rolls"],
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
            roll_results = calculate_roll_results(
                self.obs["roll_result_traits"], self.obs["roll_result_values"]
            )

            dice_faces = calculate_dice_faces(
                self.obs["all_dice_face_traits"],
                self.obs["all_dice_face_values"],
            )
            traits = calculate_traits(self.obs["traits"])

            print("\n> Roll results")
            render_table(ROLL_HEADERS, roll_results)

            print("\n> Lookup tables")
            render_table(DICES_HEADERS, dice_faces)
            render_table(TRAITS_HEADERS, traits)

    def close(self):
        pass
