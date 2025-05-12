import numpy as np

from src.env.data.game import N_DICES
from src.env.dice_face import DiceFace
from src.env.dice_manager import DiceManager
from src.env.game_enums import DiceType


class RollManager:

    def __init__(self, dice_manager: DiceManager):
        self.dice_manager = dice_manager
        self.roll_results = []

    def __roll_dice(self, dice_i: int) -> DiceFace:
        face_i = np.random.randint(0, N_DICES)
        dice = self.dice_manager.get_dice(dice_i)
        face = dice.get_face(face_i)

        return face

    def roll_all_dices(self) -> None:
        self.roll_results = []

        for dice_i in range(N_DICES):
            face = self.__roll_dice(dice_i)
            self.roll_results.append(face)

    def roll_dice(self, dice_i: int) -> None:
        face = self.__roll_dice(dice_i)

        self.roll_results[dice_i] = face

    # getters
    def get_roll_results_totals_by_dice_type(self) -> list[int]:
        attack_total = 0
        defense_total = 0

        for i, face in enumerate(self.roll_results):
            dice_type = self.dice_manager.get_dice(i).get_type()
            face_value = face.get_value()

            if dice_type == DiceType.ATTACK:
                attack_total += face_value
            elif dice_type == DiceType.DEFENSE:
                defense_total += face_value

        return attack_total, defense_total

    def get_roll_results(self) -> list[DiceFace]:
        return self.roll_results

    def get_observation(self) -> (np.ndarray[int], np.ndarray[int]):
        roll_result_values = []
        roll_result_traits = []

        for i, face in enumerate(self.roll_results):
            face_value = face.get_value()
            face_trait = face.get_trait()

            roll_result_values.append(face_value)
            roll_result_traits.append(face_trait)

        return (
            np.array(roll_result_values, dtype=np.int16),
            np.array(roll_result_traits, dtype=np.int16),
        )
