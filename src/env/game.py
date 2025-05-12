import numpy as np

from src.env.data.game import (
    MAX_ENEMY_ATTACK,
    MAX_ENEMY_DEFENSE,
    MAX_ENEMY_HP,
    MAX_PLAYER_ATTACK,
    MAX_PLAYER_DEFENSE,
    MAX_PLAYER_HP,
    MIN_ENEMY_ATTACK,
    MIN_ENEMY_DEFENSE,
    MIN_ENEMY_HP,
    MIN_PLAYER_ATTACK,
    MIN_PLAYER_DEFENSE,
    MIN_PLAYER_HP,
    N_DICES,
    N_DICE_FACES,
    N_MAX_ROLLS,
)
from src.env.data.traits import TRAITS
from src.env.dice import Dice
from src.env.dice_face import DiceFace
from src.env.game_enums import DiceType, WinnerType
from src.env.trait_manager import TraitManager
from src.env.unit import Unit


class Game:

    def __init__(self):
        self.n_max_rolls = N_MAX_ROLLS
        self.n_remaining_rolls = N_MAX_ROLLS
        self.n_dices = N_DICES
        self.n_faces = N_DICE_FACES
        self.traits = TRAITS
        self.enemies_defeated = 0

        self.enemy = Unit(
            MIN_ENEMY_HP,
            MAX_ENEMY_HP,
            MIN_ENEMY_ATTACK,
            MAX_ENEMY_ATTACK,
            MIN_ENEMY_DEFENSE,
            MAX_ENEMY_DEFENSE,
            level=self.enemies_defeated + 1,
        )
        self.player = Unit(
            MIN_PLAYER_HP,
            MAX_PLAYER_HP,
            MIN_PLAYER_ATTACK,
            MAX_PLAYER_ATTACK,
            MIN_PLAYER_DEFENSE,
            MAX_PLAYER_DEFENSE,
        )

        self.trait_manager = TraitManager()
        self.dices = []
        self.roll_results = []
        self.damage_done = [0, 0]

        self.reset()

    # game flow ===============================================
    def reset(self):
        self.damage_done = [0, 0]
        self.enemies_defeated = 0

        self.trait_manager = TraitManager()
        self.dices = self.generate_dices()

        self.player = Unit(
            MIN_PLAYER_HP,
            MAX_PLAYER_HP,
            MIN_PLAYER_ATTACK,
            MAX_PLAYER_ATTACK,
            MIN_PLAYER_DEFENSE,
            MAX_PLAYER_DEFENSE,
        )
        self.enemy = Unit(
            MIN_ENEMY_HP,
            MAX_ENEMY_HP,
            MIN_ENEMY_ATTACK,
            MAX_ENEMY_ATTACK,
            MIN_ENEMY_DEFENSE,
            MAX_ENEMY_DEFENSE,
        )

        return self.new_turn()

    def next_battle(self, rolled):
        self.enemies_defeated += 1

        self.enemy = Unit(
            MIN_ENEMY_HP,
            MAX_ENEMY_HP,
            MIN_ENEMY_ATTACK,
            MAX_ENEMY_ATTACK,
            MIN_ENEMY_DEFENSE,
            MAX_ENEMY_DEFENSE,
        )

        return self.new_turn(
            rolled, winner=WinnerType.NONE, hand_played=True, new_battle=True
        )

    def new_turn(
        self,
        rolled=False,
        winner=WinnerType.NONE,
        hand_played=False,
        new_battle=False,
    ):

        self.n_remaining_rolls = self.n_max_rolls
        self.roll_results = []
        self.roll_all_dices()
        self.reset_rolls()

        prev_damage_done = self.damage_done
        self.damage_done = [0, 0]

        self.enemy.turn_start()

        game_over = winner == WinnerType.ENEMY

        return (
            self.get_observation(prev_damage_done),
            game_over,
            rolled,
            hand_played,
            new_battle,
        )

    def roll_all_dices(self):
        for dice_i in range(self.n_dices):
            face = self.roll_dice(dice_i)
            self.roll_results.append(face)

    def player_turn(self, roll_dices_i):
        game_over = False
        rolled = True
        hand_played = False
        new_battle = False

        should_roll = np.sum(roll_dices_i) > 0
        rolled = should_roll

        if should_roll:
            for i in range(N_DICES):
                if roll_dices_i[i] == 0:
                    continue

                face = self.roll_dice(i)
                self.roll_results[i] = face

            self.consume_roll()

        if self.n_remaining_rolls == 0 or not should_roll:
            self.handle_fight()
            winner = self.get_winner()
            hand_played = True

            if winner == WinnerType.ENEMY:
                game_over = True
                return (
                    self.get_observation(),
                    game_over,
                    rolled,
                    hand_played,
                    new_battle,
                )

            if winner == WinnerType.PLAYER:
                return self.next_battle(rolled)

            return self.new_turn(rolled, winner, hand_played, new_battle)

        return (
            self.get_observation(),
            game_over,
            rolled,
            hand_played,
            new_battle,
        )

    def roll_dice(self, dice_i: int) -> DiceFace:
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
        self.damage_done = [damage_to_player, damage_to_enemy]

    def get_winner(self):
        if self.player.get_hp() <= 0:
            return WinnerType.ENEMY

        if self.enemy.get_hp() <= 0:
            return WinnerType.PLAYER

        return WinnerType.NONE

    # generators ===============================================
    def generate_dices(self):
        dices = []

        for i in range(self.n_dices):
            dices.append(Dice(i, self.trait_manager))

        return dices

    # getters ===============================================
    def get_roll_results_totals_obs(self):
        attack_total, defense_total = self.get_roll_results_totals()

        return np.array(
            [float(attack_total), float(defense_total)], dtype=np.float16
        )

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

        attack_total, defense_total = self.trait_manager.apply_traits(
            attack_total, defense_total, self.roll_results
        )
        return attack_total, defense_total

    def get_all_dices_obs(self):
        all_dice_face_values = []
        all_dice_face_traits = []

        for dice in self.dices:
            dice_face_values = []
            dice_face_traits = []

            for face in dice.get_faces():
                face_value = face.get_value()
                face_trait = face.get_trait()
                dice_face_values.append(face_value)
                dice_face_traits.append(face_trait)

            all_dice_face_values.append(dice_face_values)
            all_dice_face_traits.append(dice_face_traits)

        return (
            np.array(all_dice_face_values, dtype=np.int16).flatten(),
            np.array(all_dice_face_traits, dtype=np.int16).flatten(),
        )

    def get_damage_done_obs(self):
        return np.array(self.damage_done, dtype=np.int16)

    def get_roll_result_obs(self):
        roll_result_values = []
        roll_result_traits = []

        for i, face in enumerate(self.roll_results):
            face_value = face.get_value()
            face_trait = face.get_trait()

            roll_result_values.append(face_value)
            roll_result_traits.append(face_trait)

        return (
            np.array(roll_result_values, dtype=np.int16),
            np.array(roll_result_traits, dtype=np.int16),
        )

    def get_observation(self, prev_damage_done=[0, 0]):
        roll_results_totals = self.get_roll_results_totals_obs()

        self.player.set_attack(roll_results_totals[0])
        self.player.set_defense(roll_results_totals[1])

        enemy = self.enemy.get_observation()
        player = self.player.get_observation()

        roll_result_values, roll_result_traits = self.get_roll_result_obs()
        all_dice_face_values, all_dice_face_traits = self.get_all_dices_obs()
        traits = self.trait_manager.get_observation()

        damage_done = self.get_damage_done_obs()
        if prev_damage_done[0] != 0 or prev_damage_done[1] != 0:
            damage_done = np.array(prev_damage_done, dtype=np.int16)

        return {
            "all_dice_face_traits": all_dice_face_traits,
            "all_dice_face_values": all_dice_face_values,
            "damage_done": damage_done,
            "enemy": enemy,
            "n_remaining_rolls": np.array(
                [self.n_remaining_rolls], dtype=np.int16
            ),
            "player": player,
            "roll_result_traits": roll_result_traits,
            "roll_result_values": roll_result_values,
            "traits": traits,
        }
