import numpy as np

from src.env.data.game import N_DICES
from src.env.dice import Dice
from src.env.trait_manager import TraitManager


class DiceManager:

    def __init__(self, trait_manager: TraitManager):
        self.trait_manager = trait_manager
        self.dices = self.__generate_dices()

    def __generate_dices(self) -> [Dice]:
        dices = []

        for i in range(N_DICES):
            dices.append(Dice(i, self.trait_manager))

        return dices

    def reset(self) -> None:
        self.dices = self.__generate_dices()

    # getters
    def get_dices(self) -> [Dice]:
        return self.dices

    def get_dice(self, i: int) -> Dice:
        return self.dices[i]

    def get_observation(self) -> ([int], [int]):
        all_dice_face_values = []
        all_dice_face_traits = []

        for dice in self.dices:
            dice_face_values = []
            dice_face_traits = []

            for face in dice.get_faces():
                face_value = face.get_value()
                face_trait = face.get_trait()
                dice_face_values.append(face_value)
                dice_face_traits.append(face_trait)

            all_dice_face_values.append(dice_face_values)
            all_dice_face_traits.append(dice_face_traits)

        return (
            np.array(all_dice_face_values, dtype=np.int16).flatten(),
            np.array(all_dice_face_traits, dtype=np.int16).flatten(),
        )
