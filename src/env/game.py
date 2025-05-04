import numpy as np

from src.env.env_constants import N_DICES, N_MAX_ROLLS, N_DICE_FACES, N_MAX_FACE_VALUE, N_MIN_FACE_VALUE, MIN_ENEMY_HP, \
    MIN_ENEMY_ATTACK, MIN_ENEMY_DEFENSE, N_TRAITS, MAX_PLAYER_HP, MIN_PLAYER_HP, MAX_ENEMY_HP, MAX_ENEMY_ATTACK, \
    MAX_ENEMY_DEFENSE
from src.env.game_enums import WinnerType, DiceType, EffectType, OperationType
from src.env.traits_data import TRAITS


class Game():
    n_max_rolls = N_MAX_ROLLS
    n_remaining_rolls = N_MAX_ROLLS
    n_dices = N_DICES
    n_faces = N_DICE_FACES
    n_max_face_value = N_MAX_FACE_VALUE
    n_min_face_value = N_MIN_FACE_VALUE
    traits = TRAITS

    enemy = dict(
        hp=MIN_ENEMY_HP,
        attack=MIN_ENEMY_ATTACK,
        defense=MIN_ENEMY_DEFENSE,
    )

    player = dict(
        hp=MIN_ENEMY_HP
    )

    dices = dict()
    roll_results = []
    damage_done = [0, 0]

    def __init__(self):
        self.reset()

    # game flow ===============================================
    def reset(self):
        self.dices = dict()
        self.roll_results = []
        self.damage_done = [0, 0]
        self.generate_dices()
        self.generate_dice_faces()
        self.enemy = self.generate_enemy()
        self.player = self.generate_player()

        return self.new_turn()

    def new_turn(self):
        self.n_remaining_rolls = self.n_max_rolls
        self.roll_results = []
        self.roll_all_dices()
        self.reset_rolls()

        prev_damage_done = self.damage_done
        self.damage_done = [0, 0]

        self.enemy = self.generate_enemy_turn_start()

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
        face = dice["faces"][face_i]

        return face

    def consume_roll(self):
        self.n_remaining_rolls -= 1

    def reset_rolls(self):
        self.n_remaining_rolls = self.n_max_rolls

    def handle_fight(self):
        attack, defense = self.get_roll_results_totals()

        damage_to_player = max(self.enemy["attack"] - defense, 0)
        damage_to_enemy = max(attack - self.enemy["defense"], 0)

        self.set_damage_done(damage_to_player, damage_to_enemy)
        self.player["hp"] -= damage_to_player
        self.enemy["hp"] -= damage_to_enemy

    def set_damage_done(self, damage_to_player, damage_to_enemy):
        self.damage_done = [
            damage_to_player,
            damage_to_enemy
        ]

    def get_winner(self):
        if self.player["hp"] <= 0:
            return WinnerType.ENEMY

        if self.enemy["hp"] <= 0:
            return WinnerType.PLAYER

        return WinnerType.NONE

    # generators ===============================================
    def generate_dices(self):
        for i in range(self.n_dices):
            new_dice = self.generate_dice(i)
            self.dices[i] = new_dice

    def generate_dice(self, i):
        faces = self.generate_dice_faces()
        dice_type = DiceType.ATTACK if i < self.n_dices // 2 else DiceType.DEFENSE

        return dict(
            faces=faces,
            dice_type=dice_type
        )

    def generate_dice_faces(self):
        faces = []
        for i in range(self.n_faces):
            face = self.generate_dice_face()
            faces.append(face)
        return faces

    def generate_dice_face(self):
        value = np.random.randint(self.n_min_face_value, self.n_max_face_value + 1)
        trait = np.random.randint(0, N_TRAITS)
        return dict(
            value=value,
            trait=trait
        )

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
            trait = face["trait"]

            if trait not in traits:
                traits[trait] = 1
            else:
                traits[trait] += 1

        return traits

    def get_trait_levels(self, face_traits):
        trait_levels = []

        for face_trait in face_traits:
            trait = TRAITS[face_trait]
            level = trait.get_current_effect(face_trait)
            trait_levels.append(level)

        return trait_levels

    def get_trait_effects(self, face_traits):
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

    def apply_trait_effects(self, effects, attack_total, defense_total):
        for effect in effects:
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

    def sort_traits_effects(self, face_traits):
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

    def apply_traits(self, attack_total, defense_total):
        face_traits = self.get_face_traits()

        # get the trait effects that apply
        effects = self.get_trait_effects(face_traits)

        sorted_effects = self.sort_traits_effects(effects)
        attack_total, defense_total = self.apply_trait_effects(sorted_effects, attack_total, defense_total)

        return attack_total, defense_total

    def get_roll_results_totals(self):
        attack_total = 0
        defense_total = 0

        for i, face in enumerate(self.roll_results):
            dice_type = self.dices[i]["dice_type"]

            if dice_type == DiceType.ATTACK:
                attack_total += face["value"]
            elif dice_type == DiceType.DEFENSE:
                defense_total += face["value"]

        attack_total, defense_total = self.apply_traits(attack_total, defense_total)
        return attack_total, defense_total

    def get_dices(self):
        dices = []
        dice_types = []

        for dice in self.dices:
            dice = self.dices[dice]

            dice_type = dice["dice_type"].value
            dice_faces = []

            for face in dice["faces"]:
                dice_faces.append([face["value"], face["trait"]])

            dices.append(dice_faces)
            dice_types.append(dice_type)

        return np.array(dices, dtype=np.int8), np.array(dice_types, dtype=np.int8)

    def get_damage_done_obs(self):
        return np.array(self.damage_done, dtype=np.int16)

    def get_traits_obs(self):
        traits = []
        for i in range(N_TRAITS):
            trait = TRAITS[i]
            traits.append(trait.serialize_obs())

        return np.array(traits, dtype=np.int8)

    def get_roll_results_obs(self):
        roll_results = []

        for i, face in enumerate(self.roll_results):
            roll_results.append([face["value"], face["trait"]])

        return np.array(roll_results, dtype=np.int8)

    def get_observation(self, prev_damage_done = [0, 0]):
        roll_results_totals = self.get_roll_results_totals_obs()
        roll_results = self.get_roll_results_obs()
        dice_faces, dice_types = self.get_dices()
        enemy = self.get_enemy_obs()
        player = self.get_player_obs()
        traits = self.get_traits_obs()

        damage_done = self.get_damage_done_obs()
        if prev_damage_done[0] != 0 or prev_damage_done[1] != 0:
            damage_done = np.array(prev_damage_done, dtype=np.int16)

        return dict(
            dice_faces=dice_faces,
            dice_types=dice_types,
            roll_results=roll_results,
            roll_results_totals=roll_results_totals,
            n_remaining_rolls=np.array([self.n_remaining_rolls], dtype=np.int8),
            enemy=enemy,
            player=player,
            damage_done=damage_done,
            traits=traits,
        )

    # enemy ===============================================
    def generate_enemy_hp(self):
        hp = np.random.randint(MIN_ENEMY_HP, MAX_ENEMY_HP + 1)

        return hp

    def generate_enemy_attack(self):
        attack = np.random.randint(MIN_ENEMY_ATTACK, MAX_ENEMY_ATTACK + 1)
        return attack

    def generate_enemy_defense(self):
        defense = np.random.randint(MIN_ENEMY_DEFENSE, MAX_ENEMY_DEFENSE + 1)
        return defense

    def generate_enemy(self):
        hp = self.generate_enemy_hp()

        enemy = dict(
            max_hp=hp,
            hp=hp,
            attack=MIN_ENEMY_ATTACK,
            defense=MIN_ENEMY_DEFENSE
        )
        return enemy

    def generate_enemy_turn_start(self):
        enemy = dict(
            max_hp=self.enemy["max_hp"],
            hp=self.enemy["hp"],
            attack=self.generate_enemy_attack(),
            defense=self.generate_enemy_defense()
        )
        return enemy

    def get_enemy_obs(self):
        arr = [
            self.enemy["max_hp"],
            self.enemy["hp"],
            self.enemy["attack"],
            self.enemy["defense"]
        ]
        return np.array(arr, dtype=np.int16)

    # player ===============================================
    def generate_player_hp(self):
        hp = np.random.randint(MIN_PLAYER_HP, MAX_PLAYER_HP + 1)
        return hp

    def generate_player(self):
        hp = self.generate_player_hp()

        player = dict(
            max_hp=hp,
            hp=hp
        )
        return player

    def get_player_obs(self):
        arr = [
            self.player["max_hp"],
            self.player["hp"]
        ]

        return np.array(arr, dtype=np.int16)
