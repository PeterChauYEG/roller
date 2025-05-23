from src.env.game_enums import EffectType, OperationType


class TraitEffect:

    def __init__(
        self, effect_type: EffectType, value: int, operation: OperationType
    ):
        self.type = effect_type
        self.value = value
        self.operation = operation

    # getters
    def get_type(self) -> EffectType:
        return self.type

    def get_value(self) -> int:
        return self.value

    def get_operation(self) -> OperationType:
        return self.operation

    def get_observation(self) -> [int, int, int]:
        return [self.type.value, self.value, self.operation.value]
