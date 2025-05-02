import gymnasium as gym
import numpy as np
from gymnasium import spaces

N_ACTIONS = 6
N_DICES = 6
N_DICE_FACES = 6

class Roller():
    n_max_rolls = 3
    n_remaining_rolls = 3
    n_dices = 6
    n_faces = 6
    n_max_face_value = 20
    n_min_face_value = 1

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
        face = self.dices[dice_i][face_i]

        return face

    def consume_roll(self):
        self.n_remaining_rolls -= 1 
    
    # generators ===============================================
    def generate_dices(self):
        for i in range(self.n_dices):
            new_dice = self.generate_dice()
            self.dices[i] = new_dice

    def generate_dice(self):
        faces = self.generate_dice_faces()
        return faces

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
        roll_results_totals = 0

        for face in self.roll_results:
            roll_results_totals += face

        return roll_results_totals
    
    def get_dices(self):
        dices = []

        for dice in self.dices:
            dice_faces = []

            for face in self.dices[dice]:
                dice_faces.append(face)

            dices.append(dice_faces)

        return np.array(dices, dtype=np.int8)

    def get_observation(self, play_hand=False):
        roll_results_totals = self.get_roll_results_totals()
        dices = self.get_dices()

        return dict(
            dices=dices,
            roll_results=np.array(self.roll_results, dtype=np.int8),
            roll_results_totals=np.array([float(roll_results_totals)], dtype=np.float16),
            n_remaining_rolls=np.array([self.n_remaining_rolls], dtype=np.int8),
        ), play_hand

class RollerEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.action_space = spaces.MultiBinary(N_ACTIONS)
        self.observation_space = spaces.Dict({
            "dices": spaces.Box(low=0, high=100, shape=(N_DICES, N_DICE_FACES), dtype=np.int8),
            "roll_results": spaces.Box(low=0, high=100, shape=(N_DICES,), dtype=np.int8),
            "roll_results_totals": spaces.Box(low=0, high=255, shape=(1,), dtype=np.float16),
            "n_remaining_rolls": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int8),
        })

        self.roller = Roller()
        self.last_roll_results_totals = 0.

    def step(self, action):
        info = {}
        truncated = False
        terminated = False

        obs, play_hand = self.roller.roll_dices(action)

        reward = float(obs["roll_results_totals"][0]) - self.last_roll_results_totals

        n_remaining_rolls = obs["n_remaining_rolls"][0]

        self.last_roll_results_totals = obs["roll_results_totals"][0]

        if play_hand or n_remaining_rolls == 0:
            terminated = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info = {}

        obs, play_hand = self.roller.reset()
        self.last_roll_results_totals = float(obs["roll_results_totals"][0])

        return obs, info

    def render(self):
        pass

    def close(self):
        pass
