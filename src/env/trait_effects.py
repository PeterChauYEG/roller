from typing import Dict, Union

from src.env.trait_effect import TraitEffect


class TraitEffects:

    def __init__(self, effects: Dict[int, Union[[TraitEffect], None]]):
        self.effects = effects

    # getters
    def get_effects(self) -> [TraitEffect]:
        return self.effects

    def get_effect(self, level: int) -> TraitEffect:
        return self.effects[level]

    def get_observation(self) -> [[int, int, int, int]]:
        serialized_effects = []

        for level, effects in self.effects.items():
            if effects is None:
                continue

            for effect in effects:
                effect = [level, *effect.get_observation()]
                serialized_effects.append(effect)

        return serialized_effects
