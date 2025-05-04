import numpy as np

from src.env.data.game import N_MAX_FACE_VALUE, N_MIN_FACE_VALUE, N_TRAITS


class DiceFace:
    def __init__(self):
        self.n_max_face_value = N_MAX_FACE_VALUE
        self.n_min_face_value = N_MIN_FACE_VALUE
        self.n_traits = N_TRAITS

        self.value = 0
        self.trait = 0
        self.generate_value()
        self.generate_trait()

    def generate_value(self):
        self.value = np.random.randint(self.n_min_face_value, self.n_max_face_value + 1)

    def generate_trait(self):
        np.random.randint(0, self.n_traits)

    # getters
    def get_trait(self):
        return self.trait

    def get_value(self):
        return self.value
