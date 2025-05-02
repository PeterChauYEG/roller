import gymnasium as gym
import numpy as np
from gymnasium import spaces
from enum import Enum

N_ACTIONS = 6
N_DICES = 6
N_DICE_FACES = 6
N_DICE_TYPES = 2
N_MAX_FACE_VALUE = 20
N_MIN_FACE_VALUE = 1
N_MAX_ROLLS = 3

MAX_ROLL = N_DICES * N_MAX_FACE_VALUE / 2
MIN_ROLL = N_DICES * N_MIN_FACE_VALUE / 2

class DiceType(Enum):
    ATTACK = 0
    DEFENSE = 1

class Roller():
    n_max_rolls = N_MAX_ROLLS
    n_remaining_rolls = N_MAX_ROLLS
    n_dices = N_DICES
    n_faces = N_DICE_FACES
    n_max_face_value = N_MAX_FACE_VALUE
    n_min_face_value = N_MIN_FACE_VALUE

    dices = dict()
    roll_results = []
    
    def __init__(self):
        self.reset()

    # game flow ===============================================
    def reset(self):
        self.dices = dict()
        self.roll_results = []
        self.generate_dices()
        self.generate_dice_faces()

        return self.new_turn()
    
    def new_turn(self):
        self.n_remaining_rolls = self.n_max_rolls
        self.roll_results = []
        self.roll_all_dices()

        return self.get_observation()

    def roll_all_dices(self):
        for dice_i in range(self.n_dices):
            value = self.roll_dice(dice_i)
            self.roll_results.append(value)

    def roll_dices(self, roll_dices_i):
        if np.sum(roll_dices_i) == 0:
            return self.get_observation(True)

        for i in range(len(roll_dices_i)):
            if roll_dices_i[i] == 0:
                continue

            value = self.roll_dice(i)
            self.roll_results[i] = value

        self.consume_roll()
        
        return self.get_observation()

    def roll_dice(self, dice_i):
        face_i = np.random.randint(0, self.n_faces)
        dice = self.dices[dice_i]
        face = dice["faces"][face_i]

        return face

    def consume_roll(self):
        self.n_remaining_rolls -= 1 
    
    # generators ===============================================
    def generate_dices(self):
        for i in range(self.n_dices):
            new_dice = self.generate_dice(i)
            self.dices[i] = new_dice

    def generate_dice(self, i):
        faces = self.generate_dice_faces()
        dice_type = DiceType.ATTACK if i < self.n_dices // 2 else DiceType.DEFENSE

        return dict(
            faces=faces,
            dice_type=dice_type
        )

    def generate_dice_faces(self):
        faces = []
        for i in range(self.n_faces):
            face = self.generate_dice_face()
            faces.append(face)
        return faces

    def generate_dice_face(self):
        face = np.random.randint(self.n_min_face_value, self.n_max_face_value + 1)
        return face
    
    # getters ===============================================
    def get_roll_results_totals(self):
        attack_total = 0
        defense_total = 0

        for i, face in enumerate(self.roll_results):
            dice_type = self.dices[i]["dice_type"]

            if dice_type == DiceType.ATTACK:
                attack_total += face
            elif dice_type == DiceType.DEFENSE:
                defense_total += face

        return np.array([float(attack_total), float(defense_total)], dtype=np.float16)
    
    def get_dices(self):
        dices = []
        dice_types = []

        for dice in self.dices:
            dice = self.dices[dice]

            dice_type = dice["dice_type"].value
            dice_faces = []

            for face in dice["faces"]:
                dice_faces.append(face)

            dices.append(dice_faces)
            dice_types.append(dice_type)

        return np.array(dices, dtype=np.int8), np.array(dice_types, dtype=np.int8)

    def get_observation(self, play_hand=False):
        roll_results_totals = self.get_roll_results_totals()
        dice_faces, dice_types = self.get_dices()

        return dict(
            dice_faces=dice_faces,
            dice_types=dice_types,
            roll_results=np.array(self.roll_results, dtype=np.int8),
            roll_results_totals=roll_results_totals,
            n_remaining_rolls=np.array([self.n_remaining_rolls], dtype=np.int8),
        ), play_hand

class RollerEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

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
        })

        self.roller = Roller()
        self.last_roll_results_totals = [0., 0.]

    def step(self, action):
        info = {}
        truncated = False
        terminated = False

        obs, play_hand = self.roller.roll_dices(action)

        diff = obs["roll_results_totals"] - self.last_roll_results_totals
        reward = float(diff.sum())

        n_remaining_rolls = obs["n_remaining_rolls"][0]

        self.last_roll_results_totals = obs["roll_results_totals"]

        if play_hand or n_remaining_rolls == 0:
            terminated = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info = {}

        obs, play_hand = self.roller.reset()
        self.last_roll_results_totals = obs["roll_results_totals"]

        return obs, info

    def render(self):
        pass

    def close(self):
        pass
