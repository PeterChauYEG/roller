from src.env.data.game import N_DICE_FACES, N_TRAITS
from src.env.data.traits import TRAITS


def has_damage_been_done(damage_done):
    damage_to_player = damage_done[0]
    damage_to_enemy = damage_done[1]

    return damage_to_player > 0 or damage_to_enemy > 0


def get_percent_damage_of_max_hp(damage, max_hp):
    return (damage / max_hp) * 100


def get_damage_diff_percent(damage_done, player, enemy):
    damage_to_player = get_percent_damage_of_max_hp(damage_done[0], player)
    damage_to_enemy = get_percent_damage_of_max_hp(damage_done[1], enemy)
    diff = damage_to_enemy - damage_to_player

    return diff


def get_number_of_trait_effects():
    trait_effects = 0

    for i in range(N_TRAITS):
        trait = TRAITS[i]
        for level in range(1, N_DICE_FACES + 1):
            if trait.effects.get_effect(level) is not None:
                trait_effects += 1

    return trait_effects
