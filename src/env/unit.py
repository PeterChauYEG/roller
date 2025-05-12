import numpy as np


class Unit:
    def __init__(
        self,
        min_hp: int,
        max_hp: int,
        min_attack: int,
        max_attack: int,
        min_defense: int,
        max_defense: int,
        level: int = 1,
    ):
        self.level = level
        self.hp_range = {"min": min_hp, "max": max_hp}
        self.attack_range = {"min": min_attack, "max": max_attack}
        self.defense_range = {"min": min_defense, "max": max_defense}

        hp = self.__generate_hp()
        self.hp = hp
        self.max_hp = hp

        self.attack = self.__generate_attack()
        self.defense = self.__generate_defense()

    # generators
    def __generate_hp(self) -> int:
        return (
            np.random.randint(self.hp_range["min"], self.hp_range["max"] + 1)
            * self.level
        )

    def __generate_attack(self) -> int:
        return np.random.randint(
            self.attack_range["min"], self.attack_range["max"] + 1
        )

    def __generate_defense(self) -> int:
        return np.random.randint(
            self.defense_range["min"], self.defense_range["max"] + 1
        )

    # turn
    def turn_start(self) -> None:
        self.__generate_attack()
        self.__generate_defense()

    def apply_damage(self, attack: int) -> int:
        effective_damage = max(attack - self.defense, 0)
        self.set_hp(effective_damage)

        return effective_damage

    # setters
    def set_hp(self, damage: int) -> None:
        self.hp = max(self.hp - damage, 0)

    def set_attack(self, attack: int) -> None:
        self.attack = attack

    def set_defense(self, defense: int) -> None:
        self.defense = defense

    # getters
    def get_hp(self) -> int:
        return self.hp

    def get_attack(self) -> int:
        return self.attack

    # obs
    def get_observation(self) -> np.ndarray[np.int16]:
        arr = [self.max_hp, self.hp, self.attack, self.defense]

        return np.array(arr, dtype=np.int16)
