from src.env.game_enums import EffectType, OperationType
from src.env.trait import Trait

TRAITS = {
    0: Trait(
        "Attack Boost",
        {
            1: None,
            2: None,
            3: [{
                "type": EffectType.ATTACK,
                "value": 5,
                "operation": OperationType.MULTIPLY
            }],
            4: None,
            5: [{
                "type": EffectType.ATTACK,
                "value": 15,
                "operation": OperationType.MULTIPLY
            }],
            6: [{
                "type": EffectType.ATTACK,
                "value": 30,
                "operation": OperationType.MULTIPLY
            }],
        }
    ),
    1: Trait(
        "Defense Boost",
        {
            1: None,
            2: None,
            3: [{
                "type": EffectType.DEFENSE,
                "value": 5,
                "operation": OperationType.MULTIPLY
            }],
            4: None,
            5: [{
                "type": EffectType.DEFENSE,
                "value": 15,
                "operation": OperationType.MULTIPLY
            }],
            6: [{
                "type": EffectType.DEFENSE,
                "value": 30,
                "operation": OperationType.MULTIPLY
            }],
        }
    ),
    2: Trait(
        "Attack",
        {
            1: None,
            2: [{
                "type": EffectType.ATTACK,
                "value": 10,
                "operation": OperationType.ADD
            }],
            3: [{
                "type": EffectType.ATTACK,
                "value": 15,
                "operation": OperationType.ADD
            }],
            4: [{
                "type": EffectType.ATTACK,
                "value": 20,
                "operation": OperationType.ADD
            }],
            5: [{
                "type": EffectType.ATTACK,
                "value": 30,
                "operation": OperationType.ADD
            }],
            6: [{
                "type": EffectType.ATTACK,
                "value": 40,
                "operation": OperationType.ADD
            }],
        }
    ),
    3: Trait(
        "Defense",
        {
            1: None,
            2: [{
                "type": EffectType.DEFENSE,
                "value": 10,
                "operation": OperationType.ADD
            }],
            3: [{
                "type": EffectType.DEFENSE,
                "value": 15,
                "operation": OperationType.ADD
            }],
            4: [{
                "type": EffectType.DEFENSE,
                "value": 20,
                "operation": OperationType.ADD
            }],
            5: [{
                "type": EffectType.DEFENSE,
                "value": 30,
                "operation": OperationType.ADD
            }],
            6: [{
                "type": EffectType.DEFENSE,
                "value": 40,
                "operation": OperationType.ADD
            }],
        }
    ),
}
