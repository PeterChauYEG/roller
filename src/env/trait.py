from typing import Union

from src.env.trait_effect import TraitEffect
from src.env.trait_effects import TraitEffects


class Trait:

    def __init__(self, name: str, effects: TraitEffects):
        self.name = name
        self.effects = effects

    def get_current_effect(self, level: int) -> Union[[TraitEffect], None]:
        if level == 0:
            return None

        effect = self.effects.get_effect(level)

        if effect is not None:
            return effect

        return self.get_current_effect(level - 1)

    def get_obs(self) -> [[int, int, int, int]]:
        return self.effects.get_obs()
