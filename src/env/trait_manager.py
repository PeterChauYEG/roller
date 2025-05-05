import numpy as np

from src.env.data.game import N_DICES, N_DICE_FACES, N_TRAITS, TRAIT_DISTRIBUTION
from src.env.data.traits import TRAITS


class TraitManager:
    def __init__(self):
        self.n_dices = N_DICES
        self.n_dice_faces = N_DICE_FACES
        self.n_traits = N_TRAITS
        self.trait_keys = [key for key in TRAITS.keys()]
        self.traits = self.generate_dice_face_traits()

    # generators
    def generate_dice_face_traits(self):
        return np.random.choice(
            self.trait_keys,
            p=TRAIT_DISTRIBUTION,
            size=(self.n_dices, self.n_dice_faces)
        )

    # getters
    def get_trait(self, dice_i, face_i):
        return self.traits[dice_i][face_i]
