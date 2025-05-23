import numpy as np

from src.env.data.game import N_MAX_FACE_VALUE, N_MIN_FACE_VALUE, N_TRAITS
from src.env.data.traits import TRAITS
from src.env.trait import Trait


class DiceFace:

    def __init__(self, trait: Trait):
        self.n_max_face_value = N_MAX_FACE_VALUE
        self.n_min_face_value = N_MIN_FACE_VALUE
        self.n_traits = N_TRAITS
        self.trait_keys = list(TRAITS.keys())

        self.trait = trait
        self.value = self.generate_value()

    def generate_value(self) -> int:
        return np.random.randint(
            self.n_min_face_value, self.n_max_face_value + 1
        )

    # getters
    def get_trait(self) -> Trait:
        return self.trait

    def get_value(self) -> int:
        return self.value
