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
