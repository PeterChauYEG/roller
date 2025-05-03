import numpy as np
from enum import Enum

N_ACTIONS = 6
N_DICES = 6
N_DICE_FACES = 6
N_DICE_TYPES = 2
N_MAX_FACE_VALUE = 20
N_MIN_FACE_VALUE = 1
N_MAX_ROLLS = 3

MAX_ROLL = N_DICES * N_MAX_FACE_VALUE / 2
MIN_ROLL = N_DICES * N_MIN_FACE_VALUE / 2

MIN_PLAYER_HP = 100
MAX_PLAYER_HP = 100

MIN_ENEMY_HP = 100
MAX_ENEMY_HP = 200
MAX_ENEMY_ATTACK = 50
MIN_ENEMY_ATTACK = 10
MAX_ENEMY_DEFENSE = 50
MIN_ENEMY_DEFENSE = 10

class DiceType(Enum):
    ATTACK = 0
    DEFENSE = 1

class WinnerType(Enum):
    PLAYER = 0
    ENEMY = 1
    NONE = 2

class Game():
    n_max_rolls = N_MAX_ROLLS
    n_remaining_rolls = N_MAX_ROLLS
    n_dices = N_DICES
    n_faces = N_DICE_FACES
    n_max_face_value = N_MAX_FACE_VALUE
    n_min_face_value = N_MIN_FACE_VALUE

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

        return self.get_observation(prev_damage_done), WinnerType.NONE

    def roll_all_dices(self):
        for dice_i in range(self.n_dices):
            value = self.roll_dice(dice_i)
            self.roll_results.append(value)

    def player_turn(self, roll_dices_i):
        should_roll = np.sum(roll_dices_i) > 0

        if should_roll:
            for i in range(len(roll_dices_i)):
                if roll_dices_i[i] == 0:
                    continue

                value = self.roll_dice(i)
                self.roll_results[i] = value

            self.consume_roll()

        if self.n_remaining_rolls == 0 or not should_roll:
            self.handle_fight()
            winner = self.get_winner()

            if winner != WinnerType.NONE:
                return self.get_observation(), winner

            return self.new_turn()

        return self.get_observation(), WinnerType.NONE

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
        face = np.random.randint(self.n_min_face_value, self.n_max_face_value + 1)
        return face
    
    # getters ===============================================
    def get_roll_results_totals_obs(self):
        attack_total, defense_total = self.get_roll_results_totals()

        return np.array([float(attack_total), float(defense_total)], dtype=np.float16)

    def get_roll_results_totals(self):
        attack_total = 0
        defense_total = 0

        for i, face in enumerate(self.roll_results):
            dice_type = self.dices[i]["dice_type"]

            if dice_type == DiceType.ATTACK:
                attack_total += face
            elif dice_type == DiceType.DEFENSE:
                defense_total += face

        return attack_total, defense_total
    
    def get_dices(self):
        dices = []
        dice_types = []

        for dice in self.dices:
            dice = self.dices[dice]

            dice_type = dice["dice_type"].value
            dice_faces = []

            for face in dice["faces"]:
                dice_faces.append(face)

            dices.append(dice_faces)
            dice_types.append(dice_type)

        return np.array(dices, dtype=np.int8), np.array(dice_types, dtype=np.int8)

    def get_damage_done_obs(self):
        return np.array(self.damage_done, dtype=np.int16)

    def get_observation(self, prev_damage_done = [0, 0]):
        roll_results_totals = self.get_roll_results_totals_obs()
        dice_faces, dice_types = self.get_dices()
        enemy = self.get_enemy_obs()
        player = self.get_player_obs()

        damage_done = self.get_damage_done_obs()
        if prev_damage_done[0] != 0 or prev_damage_done[1] != 0:
            damage_done = np.array(prev_damage_done, dtype=np.int16)

        return dict(
            dice_faces=dice_faces,
            dice_types=dice_types,
            roll_results=np.array(self.roll_results, dtype=np.int8),
            roll_results_totals=roll_results_totals,
            n_remaining_rolls=np.array([self.n_remaining_rolls], dtype=np.int8),
            enemy=enemy,
            player=player,
            damage_done=damage_done
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
