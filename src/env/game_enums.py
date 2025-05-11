from enum import Enum


class EffectType(Enum):
    ATTACK = 0
    DEFENSE = 1


class OperationType(Enum):
    ADD = 0
    MULTIPLY = 1
    DIVIDE = 2
    SUBTRACT = 3


class DiceType(Enum):
    ATTACK = 0
    DEFENSE = 1


class WinnerType(Enum):
    PLAYER = 0
    ENEMY = 1
    NONE = 2
