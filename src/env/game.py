import numpy as np

from src.env.data.game import N_DICES, N_MAX_ROLLS, N_DICE_FACES, N_MAX_FACE_VALUE, N_MIN_FACE_VALUE, MIN_ENEMY_HP, \
    MIN_ENEMY_ATTACK, MIN_ENEMY_DEFENSE, N_TRAITS, MAX_PLAYER_HP, MIN_PLAYER_HP, MAX_ENEMY_HP, MAX_ENEMY_ATTACK, \
    MAX_ENEMY_DEFENSE, MIN_PLAYER_ATTACK, MAX_PLAYER_ATTACK, MIN_PLAYER_DEFENSE, MAX_PLAYER_DEFENSE
from src.env.dice import Dice
from src.env.game_enums import WinnerType, DiceType, EffectType, OperationType
from src.env.data.traits import TRAITS
from src.env.unit import Unit


def get_trait_levels(face_traits):
    trait_levels = []

    for face_trait in face_traits:
        trait = TRAITS[face_trait]
        level = trait.get_current_effect(face_trait)
        trait_levels.append(level)

    return trait_levels

def get_trait_effects(face_traits):
    trait_effects = []

    for face_trait in face_traits:
        trait = TRAITS[face_trait]
        level = face_traits[face_trait]

        effects = trait.get_current_effect(level)

        if effects is None:
            continue

        for effect in effects:
            trait_effects.append(effect)

    return trait_effects

def apply_trait_effects(effects, attack_total, defense_total):
    for effect in effects:
        attack_total, defense_total = apply_trait_effect(effect, attack_total, defense_total)

    return attack_total, defense_total

def apply_trait_effect(effect, attack_total, defense_total):
    if effect["type"] == EffectType.ATTACK:
        if effect["operation"] == OperationType.ADD:
            attack_total += effect["value"]
        elif effect["operation"] == OperationType.SUBTRACT:
            attack_total -= effect["value"]
        elif effect["operation"] == OperationType.MULTIPLY:
            attack_total *= effect["value"]
        elif effect["operation"] == OperationType.DIVIDE:
            attack_total /= effect["value"]
    elif effect["type"] == EffectType.DEFENSE:
        if effect["operation"] == OperationType.ADD:
            defense_total += effect["value"]
        elif effect["operation"] == OperationType.SUBTRACT:
            defense_total -= effect["value"]
        elif effect["operation"] == OperationType.MULTIPLY:
            defense_total *= effect["value"]
        elif effect["operation"] == OperationType.DIVIDE:
            defense_total /= effect["value"]

    return attack_total, defense_total


def sort_traits_effects(face_traits):
    sorted_traits = []
    add_effects = []
    subtract_effects = []
    multiply_effects = []
    divide_effects = []

    for face_trait in face_traits:
        if face_trait["operation"] == OperationType.ADD:
            add_effects.append(face_trait)
        if face_trait["operation"] == OperationType.SUBTRACT:
            subtract_effects.append(face_trait)
        if face_trait["operation"] == OperationType.MULTIPLY:
            multiply_effects.append(face_trait)
        if face_trait["operation"] == OperationType.DIVIDE:
            divide_effects.append(face_trait)

    sorted_traits.extend(add_effects)
    sorted_traits.extend(subtract_effects)
    sorted_traits.extend(multiply_effects)
    sorted_traits.extend(divide_effects)

    return sorted_traits


def get_traits_obs():
    traits = []
    for i in range(N_TRAITS):
        trait = TRAITS[i]
        traits.append(trait.serialize_obs())

    return np.array(traits, dtype=np.int8)


class Game():
    def __init__(self):
        self.n_max_rolls = N_MAX_ROLLS
        self.n_remaining_rolls = N_MAX_ROLLS
        self.n_dices = N_DICES
        self.n_faces = N_DICE_FACES
        self.traits = TRAITS

        self.enemy = Unit(
            MIN_ENEMY_HP,
            MAX_ENEMY_HP,
            MIN_ENEMY_ATTACK,
            MAX_ENEMY_ATTACK,
            MIN_ENEMY_DEFENSE,
            MAX_ENEMY_DEFENSE
        )
        self.player = Unit(
            MIN_PLAYER_HP,
            MAX_PLAYER_HP,
            MIN_PLAYER_ATTACK,
            MAX_PLAYER_ATTACK,
            MIN_PLAYER_DEFENSE,
            MAX_PLAYER_DEFENSE
        )

        self.dices = []
        self.roll_results = []
        self.damage_done = [0, 0]

        self.reset()

    # game flow ===============================================
    def reset(self):
        self.dices = []
        self.roll_results = []
        self.damage_done = [0, 0]
        self.generate_dices()
        self.player = Unit(
            MIN_PLAYER_HP,
            MAX_PLAYER_HP,
            MIN_PLAYER_ATTACK,
            MAX_PLAYER_ATTACK,
            MIN_PLAYER_DEFENSE,
            MAX_PLAYER_DEFENSE
        )
        self.enemy = Unit(
            MIN_ENEMY_HP,
            MAX_ENEMY_HP,
            MIN_ENEMY_ATTACK,
            MAX_ENEMY_ATTACK,
            MIN_ENEMY_DEFENSE,
            MAX_ENEMY_DEFENSE
        )

        return self.new_turn()

    def new_turn(self):
        self.n_remaining_rolls = self.n_max_rolls
        self.roll_results = []
        self.roll_all_dices()
        self.reset_rolls()

        prev_damage_done = self.damage_done
        self.damage_done = [0, 0]

        self.enemy.turn_start()

        return self.get_observation(prev_damage_done), WinnerType.NONE, False

    def roll_all_dices(self):
        for dice_i in range(self.n_dices):
            face = self.roll_dice(dice_i)
            self.roll_results.append(face)

    def player_turn(self, roll_dices_i):
        should_roll = np.sum(roll_dices_i) > 0

        if should_roll:
            for i in range(len(roll_dices_i)):
                if roll_dices_i[i] == 0:
                    continue

                face = self.roll_dice(i)
                self.roll_results[i] = face

            self.consume_roll()

        if self.n_remaining_rolls == 0 or not should_roll:
            self.handle_fight()
            winner = self.get_winner()

            if winner != WinnerType.NONE:
                return self.get_observation(), winner, False

            return self.new_turn()

        return self.get_observation(), WinnerType.NONE, True

    def roll_dice(self, dice_i):
        face_i = np.random.randint(0, self.n_faces)
        dice = self.dices[dice_i]
        face = dice.get_face(face_i)

        return face

    def consume_roll(self):
        self.n_remaining_rolls -= 1

    def reset_rolls(self):
        self.n_remaining_rolls = self.n_max_rolls

    def handle_fight(self):
        attack, defense = self.get_roll_results_totals()

        self.player.set_attack(attack)
        self.player.set_defense(defense)

        damage_to_player = self.player.apply_damage(self.enemy.get_attack())
        damage_to_enemy = self.enemy.apply_damage(attack)

        self.set_damage_done(damage_to_player, damage_to_enemy)

    def set_damage_done(self, damage_to_player, damage_to_enemy):
        self.damage_done = [
            damage_to_player,
            damage_to_enemy
        ]

    def get_winner(self):
        if self.player.get_hp() <= 0:
            return WinnerType.ENEMY

        if self.enemy.get_hp() <= 0:
            return WinnerType.PLAYER

        return WinnerType.NONE

    # generators ===============================================
    def generate_dices(self):
        self.dices = []

        for i in range(self.n_dices):
            self.dices.append(Dice(i))

    # getters ===============================================
    def get_roll_results_totals_obs(self):
        attack_total, defense_total = self.get_roll_results_totals()

        # attack_total, defense_total = self.apply_traits(attack_total, defense_total)

        return np.array([float(attack_total), float(defense_total)], dtype=np.float16)

    def get_face_traits(self):
        # trait id (key), level
        traits = {}

        for i in range(self.n_dices):
            face = self.roll_results[i]
            trait = face.get_trait()

            if trait not in traits:
                traits[trait] = 1
            else:
                traits[trait] += 1

        return traits

    def apply_traits(self, attack_total, defense_total):
        face_traits = self.get_face_traits()

        effects = get_trait_effects(face_traits)

        sorted_effects = sort_traits_effects(effects)
        attack_total, defense_total = apply_trait_effects(sorted_effects, attack_total, defense_total)

        return attack_total, defense_total

    def get_roll_results_totals(self):
        attack_total = 0
        defense_total = 0

        for i, face in enumerate(self.roll_results):
            dice_type = self.dices[i].get_type()
            face_value = face.get_value()

            if dice_type == DiceType.ATTACK:
                attack_total += face_value
            elif dice_type == DiceType.DEFENSE:
                defense_total += face_value

        attack_total, defense_total = self.apply_traits(attack_total, defense_total)
        return attack_total, defense_total

    def get_dices_obs(self):
        dices = []
        dice_types = []

        for dice in self.dices:
            dice_type = dice.get_type().value
            dice_faces = []

            for face in dice.get_faces():
                face_value = face.get_value()
                face_trait = face.get_trait()
                dice_faces.append([face_value, face_trait])

            dices.append(dice_faces)
            dice_types.append(dice_type)

        return np.array(dices, dtype=np.int8), np.array(dice_types, dtype=np.int8)

    def get_damage_done_obs(self):
        return np.array(self.damage_done, dtype=np.int16)

    def get_roll_results_obs(self):
        roll_results = []

        for i, face in enumerate(self.roll_results):
            face_value = face.get_value()
            face_trait = face.get_trait()

            roll_results.append([face_value, face_trait])

        return np.array(roll_results, dtype=np.int8)

    def get_observation(self, prev_damage_done = [0, 0]):
        roll_results_totals = self.get_roll_results_totals_obs()

        self.player.set_attack(roll_results_totals[0])
        self.player.set_defense(roll_results_totals[1])

        enemy = self.enemy.get_obs()
        player = self.player.get_obs()

        roll_results = self.get_roll_results_obs()
        dice_faces, dice_types = self.get_dices_obs()
        traits = get_traits_obs()

        damage_done = self.get_damage_done_obs()
        if prev_damage_done[0] != 0 or prev_damage_done[1] != 0:
            damage_done = np.array(prev_damage_done, dtype=np.int16)

        return dict(
            dice_faces=dice_faces,
            dice_types=dice_types,
            roll_results=roll_results,
            n_remaining_rolls=np.array([self.n_remaining_rolls], dtype=np.int8),
            enemy=enemy,
            player=player,
            damage_done=damage_done,
            traits=traits,
        )
