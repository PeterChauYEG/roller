from src.env.data.game import N_DICES

class Trait:
    def __init__(self, name, effect):
        self.name = name
        self.effect = effect

    def get_current_effect(self, level):
        if level == 0:
            return None

        if self.effect[level]:
            return self.effect[level]

        return self.get_current_effect(level - 1)

    def serialize(self):
        return dict(
            name=self.name,
            effect=self.effect
        )

    def serialize_obs(self):
        serialized_effect = []
        for level, effects in self.effect.items():
            if effects is None:
                continue

            for effect in effects:
                effect = [
                    level,
                    effect["type"].value,
                    effect["value"],
                    effect["operation"].value
                ]
                serialized_effect.append(effect)

        return serialized_effect
