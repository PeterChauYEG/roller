from typing import Dict

import numpy as np

from src.env.data.game import (
    N_DICES,
    N_DICE_FACES,
    N_TRAITS,
    TRAIT_DISTRIBUTION,
)
from src.env.data.traits import TRAITS
from src.env.dice_face import DiceFace
from src.env.game_enums import EffectType, OperationType
from src.env.trait_effect import TraitEffect


class TraitManager:

    def __init__(self):
        self.n_dices = N_DICES
        self.n_dice_faces = N_DICE_FACES
        self.n_traits = N_TRAITS
        self.trait_keys = list(TRAITS.keys())
        self.traits = self.__generate_dice_face_traits()

    # generators
    def __generate_dice_face_traits(self) -> [int]:
        return np.random.choice(
            self.trait_keys,
            p=TRAIT_DISTRIBUTION,
            size=(self.n_dices, self.n_dice_faces),
        )

    # reset
    def reset(self) -> None:
        self.traits = self.__generate_dice_face_traits()

    # getters
    def get_trait(self, dice_i: int, face_i: int) -> int:
        return self.traits[dice_i][face_i]

    def apply_traits(
        self, attack_total: int, defense_total: int, roll_results: [DiceFace]
    ) -> (int, int):
        face_traits = self.get_face_traits(roll_results)

        effects: [TraitEffect] = self.get_trait_effects(face_traits)

        sorted_effects = self.sort_traits_effects(effects)
        attack_total, defense_total = self.apply_trait_effects(
            sorted_effects, attack_total, defense_total
        )

        return attack_total, defense_total

    @staticmethod
    def get_observation() -> np.ndarray[int]:
        traits = []
        for i in range(N_TRAITS):
            trait = TRAITS[i]
            serialized_trait = trait.get_observation()

            if len(serialized_trait) == 0:
                continue

            # append each element of the serialized trait
            for j in range(len(serialized_trait)):
                traits.append(serialized_trait[j])

        return np.array(traits, dtype=np.int16).flatten()

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

    @staticmethod
    def sort_traits_effects(trait_effects: [TraitEffect]) -> [TraitEffect]:
        sorted_traits = []
        add_effects = []
        subtract_effects = []
        multiply_effects = []
        divide_effects = []

        for effect in trait_effects:
            operation = effect.get_operation()

            if operation == OperationType.ADD:
                add_effects.append(effect)
            if operation == OperationType.SUBTRACT:
                subtract_effects.append(effect)
            if operation == OperationType.MULTIPLY:
                multiply_effects.append(effect)
            if operation == OperationType.DIVIDE:
                divide_effects.append(effect)

        sorted_traits.extend(add_effects)
        sorted_traits.extend(subtract_effects)
        sorted_traits.extend(multiply_effects)
        sorted_traits.extend(divide_effects)

        return sorted_traits

    def apply_trait_effects(
        self, effects: [TraitEffect], attack_total: int, defense_total: int
    ) -> (int, int):
        for effect in effects:
            attack_total, defense_total = self.apply_trait_effect(
                effect, attack_total, defense_total
            )

        return attack_total, defense_total

    @staticmethod
    def apply_trait_effect(
        effect: TraitEffect, attack_total: int, defense_total: int
    ) -> (int, int):
        effect_type = effect.get_type()
        operation = effect.get_operation()
        value = effect.get_value()

        if effect_type == EffectType.ATTACK:
            if operation == OperationType.ADD:
                attack_total += value
            elif operation == OperationType.SUBTRACT:
                attack_total -= value
            elif operation == OperationType.MULTIPLY:
                attack_total *= value
            elif operation == OperationType.DIVIDE:
                attack_total /= value
        elif effect_type == EffectType.DEFENSE:
            if operation == OperationType.ADD:
                defense_total += value
            elif operation == OperationType.SUBTRACT:
                defense_total -= value
            elif operation == OperationType.MULTIPLY:
                defense_total *= value
            elif operation == OperationType.DIVIDE:
                defense_total /= value

        return attack_total, defense_total
