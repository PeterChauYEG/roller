import numpy as np

class Unit:
    def __init__(
            self,
            min_hp,
            max_hp,
            min_attack,
            max_attack,
            min_defense,
            max_defense
    ):
        self.hp_range = dict(
            min=min_hp,
            max=max_hp
        )
        self.attack_range = dict(
            min=min_attack,
            max=max_attack
        )
        self.defense_range = dict(
            min=min_defense,
            max=max_defense
        )

        hp = self.generate_player_hp()
        self.hp = hp
        self.max_hp = hp

        self.attack = self.generate_attack()
        self.defense = self.generate_defense()

    # turn
    def turn_start(self):
        self.generate_attack()
        self.generate_defense()

    def apply_damage(self, attack):
        effective_damage = max(attack - self.defense, 0)
        self.set_hp(effective_damage)

        return effective_damage

    # generators
    def generate_player_hp(self):
        return np.random.randint(
            self.hp_range["min"],
            self.hp_range["max"] + 1
        )

    def generate_attack(self):
        return np.random.randint(
            self.attack_range["min"],
            self.attack_range["max"] + 1
        )

    def generate_defense(self):
        return np.random.randint(
            self.defense_range["min"],
            self.defense_range["max"] + 1
        )

    # setters
    def set_hp(self, damage):
        self.hp = max(self.hp - damage, 0)

    def set_attack(self, attack):
        self.attack = attack

    def set_defense(self, defense):
        self.defense = defense

    # getters
    def get_hp(self):
        return self.hp

    def get_attack(self):
        return self.attack

    # obs
    def get_obs(self):
        arr = [
            self.max_hp,
            self.hp,
            self.attack,
            self.defense
        ]

        return np.array(arr, dtype=np.int16)
