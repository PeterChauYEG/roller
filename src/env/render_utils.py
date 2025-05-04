from tabulate import tabulate

from src.env.game import EffectType, OperationType
from src.env.env_constants import N_DICES, N_DICE_FACES, N_TRAITS

ROLL_HEADERS = ["Dice 1", "Dice 2", "Dice 3", "Dice 4", "Dice 5", "Dice 6"]
UNIT_HEADERS = ["Name", "HP", "Max HP", "Attack", "Defense"]
INFO_HEADERS = ["Reward", "Damage Taken", "Damage Dealt", "Remaining Rolls"]
DICES_HEADERS = ["Dice", "Type", "Face 1", "Face 2", "Face 3", "Face 4", "Face 5", "Face 6"]
TRAITS_HEADERS = ["Trait", "Level", "Attack +", "Attack *", "Defense +", "Defense *"]

def render_table(headers, data):
    print(tabulate(data, headers, tablefmt="simple_outline"))

def calculate_info(damage_done, reward, n_remaining_rolls):
    reward = reward if reward is not None else 0
    damage_taken = damage_done[0]
    damage_dealt = damage_done[1]
    n_remaining_rolls = n_remaining_rolls[0]

    info = [
        reward,
        damage_taken,
        damage_dealt,
        n_remaining_rolls
    ]

    return [info]

def calculate_action(action):
    return [action]

def calculate_roll_results(roll_results):
    res = []
    for i in roll_results:
        res.append(f"{i[0]} (trait {i[1]})")

    return [res]

def calculate_units(player, enemy, roll_results_totals):
    player = [
        "Player",
        player[1],
        player[0],
        roll_results_totals[0],
        roll_results_totals[1],
    ]

    enemy = [
        "Enemy",
        enemy[1],
        enemy[0],
        enemy[2],
        enemy[3],
    ]

    return [player, enemy]

def calculate_dice_faces(dice_faces, dice_types):
    res = []

    for i in range(N_DICES):
        dice_type = dice_types[i]
        dice_type_label = "Attack" if dice_type == 0 else "Defense"

        dice = [
            i + 1,
            dice_type_label
        ]

        for j in range(N_DICE_FACES):
            value = dice_faces[i][j][0]
            trait = dice_faces[i][j][1]
            label = f"{value} (trait {trait})"

            dice.append(label)

        res.append(dice)

    return res

def calculate_traits(traits):
    res = []

    for i in range(N_TRAITS):
        trait = traits[i]
        trait_label = f"Trait {i}"

        for j in range(N_DICES):
            effect = trait[j]
            if effect.sum() == 0:
                continue

            attack_mult = "-"
            attack_add = "-"
            defense_mult = "-"
            defense_add = "-"

            if effect[1] == EffectType.DEFENSE.value:
                if effect[3] == OperationType.ADD.value:
                    defense_add = effect[2]
                elif effect[3] == OperationType.MULTIPLY.value:
                    defense_mult = effect[2]
            elif effect[1] == EffectType.ATTACK.value:
                if effect[3] == OperationType.MULTIPLY.value:
                    attack_mult = effect[2]
                elif effect[3] == OperationType.ADD.value:
                    attack_add = effect[2]

            row = [
                trait_label,
                i,
                attack_add,
                attack_mult,
                defense_add,
                defense_mult,
            ]

            res.append(row)

    return res
