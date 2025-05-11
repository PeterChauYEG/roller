import numpy as np
from typing import Dict

from src.env.data.game import N_DICES, N_DICE_FACES, N_TRAITS, TRAIT_DISTRIBUTION
from src.env.data.traits import TRAITS
from src.env.dice_face import DiceFace
from src.env.trait_effect import TraitEffect


class TraitManager:
    def __init__(self):
        self.n_dices = N_DICES
        self.n_dice_faces = N_DICE_FACES
        self.n_traits = N_TRAITS
        self.trait_keys = [key for key in TRAITS.keys()]
        self.traits = self.__generate_dice_face_traits()

    # generators
    def __generate_dice_face_traits(self) -> [int]:
        return np.random.choice(
            self.trait_keys,
            p=TRAIT_DISTRIBUTION,
            size=(self.n_dices, self.n_dice_faces),
        )

    # getters
    def get_trait(self, dice_i: int, face_i: int) -> int:
        return self.traits[dice_i][face_i]

    # returns: trait id (key), level
    @staticmethod
    def get_face_traits(roll_results: [DiceFace]) -> Dict[int, int]:
        traits = {}

        for i in range(N_DICES):
            face = roll_results[i]
            trait = face.get_trait()

            if trait not in traits:
                traits[trait] = 1
            else:
                traits[trait] += 1

        return traits

    @staticmethod
    def get_trait_effects(face_traits: Dict[int, int]) -> [TraitEffect]:
        trait_effects = []

        for face_trait in face_traits:
            trait = TRAITS[face_trait]
            level = face_traits[face_trait]

            effects = trait.get_current_effect(level)

            if effects is None:
                continue

            for effect in effects:
                trait_effects.append(effect)

        return trait_effects
