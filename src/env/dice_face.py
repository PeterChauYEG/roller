import numpy as np

from src.env.data.game import N_MAX_FACE_VALUE, N_MIN_FACE_VALUE, N_TRAITS, TRAIT_DISTRIBUTION
from src.env.data.traits import TRAITS


class DiceFace:
    def __init__(self):
        self.n_max_face_value = N_MAX_FACE_VALUE
        self.n_min_face_value = N_MIN_FACE_VALUE
        self.n_traits = N_TRAITS
        self.trait_keys = [key for key in TRAITS.keys()]

        self.value = 0
        self.trait = 0
        self.generate_value()
        self.generate_trait()

    def generate_value(self):
        self.value = np.random.randint(self.n_min_face_value, self.n_max_face_value + 1)

    def generate_trait(self):
        traits = np.random.choice(self.trait_keys, p=TRAIT_DISTRIBUTION)
        self.trait = traits

    # getters
    def get_trait(self):
        return self.trait

    def get_value(self):
        return self.value
