from src.env.game_enums import EffectType, OperationType
from src.env.trait import Trait
from src.env.trait_effect import TraitEffect
from src.env.trait_effects import TraitEffects

TRAITS = {
    0: Trait(
        "None",
        TraitEffects(
            {
                1: None,
                2: None,
                3: None,
                4: None,
                5: None,
                6: None,
            }
        ),
    ),
    1: Trait(
        "Attack Boost",
        TraitEffects(
            {
                1: None,
                2: None,
                3: [
                    TraitEffect(
                        effect_type=EffectType.ATTACK,
                        value=5,
                        operation=OperationType.MULTIPLY,
                    )
                ],
                4: None,
                5: [
                    TraitEffect(
                        effect_type=EffectType.ATTACK,
                        value=15,
                        operation=OperationType.MULTIPLY,
                    )
                ],
                6: [
                    TraitEffect(
                        effect_type=EffectType.ATTACK,
                        value=30,
                        operation=OperationType.MULTIPLY,
                    )
                ],
            }
        ),
    ),
    2: Trait(
        "Defense Boost",
        TraitEffects(
            {
                1: None,
                2: None,
                3: [
                    TraitEffect(
                        effect_type=EffectType.DEFENSE,
                        value=5,
                        operation=OperationType.MULTIPLY,
                    )
                ],
                4: None,
                5: [
                    TraitEffect(
                        effect_type=EffectType.DEFENSE,
                        value=15,
                        operation=OperationType.MULTIPLY,
                    )
                ],
                6: [
                    TraitEffect(
                        effect_type=EffectType.DEFENSE,
                        value=30,
                        operation=OperationType.MULTIPLY,
                    )
                ],
            }
        ),
    ),
    3: Trait(
        "Attack",
        TraitEffects(
            {
                1: None,
                2: [
                    TraitEffect(
                        effect_type=EffectType.ATTACK,
                        value=10,
                        operation=OperationType.ADD,
                    )
                ],
                3: [
                    TraitEffect(
                        effect_type=EffectType.ATTACK,
                        value=15,
                        operation=OperationType.ADD,
                    )
                ],
                4: [
                    TraitEffect(
                        effect_type=EffectType.ATTACK,
                        value=20,
                        operation=OperationType.ADD,
                    )
                ],
                5: [
                    TraitEffect(
                        effect_type=EffectType.ATTACK,
                        value=30,
                        operation=OperationType.ADD,
                    )
                ],
                6: [
                    TraitEffect(
                        effect_type=EffectType.ATTACK,
                        value=40,
                        operation=OperationType.ADD,
                    )
                ],
            }
        ),
    ),
    4: Trait(
        "Defense",
        TraitEffects(
            {
                1: None,
                2: [
                    TraitEffect(
                        effect_type=EffectType.DEFENSE,
                        value=10,
                        operation=OperationType.ADD,
                    )
                ],
                3: [
                    TraitEffect(
                        effect_type=EffectType.DEFENSE,
                        value=15,
                        operation=OperationType.ADD,
                    )
                ],
                4: [
                    TraitEffect(
                        effect_type=EffectType.DEFENSE,
                        value=20,
                        operation=OperationType.ADD,
                    )
                ],
                5: [
                    TraitEffect(
                        effect_type=EffectType.DEFENSE,
                        value=30,
                        operation=OperationType.ADD,
                    )
                ],
                6: [
                    TraitEffect(
                        effect_type=EffectType.DEFENSE,
                        value=40,
                        operation=OperationType.ADD,
                    )
                ],
            }
        ),
    ),
}
